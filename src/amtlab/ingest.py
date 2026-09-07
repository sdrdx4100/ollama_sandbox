"""実車/台上ログの取り込みと変速イベントの自動切り出し。

シミュレーションで作ったデータと同じ KPI スキーマに正規化するので、
``modeling`` / ``viz`` / ``calibration`` のコードをそのまま実ログに適用できる。

J1939 ログ(DBC デコード済み)であれば列名を自動検出する。信号の対応表と
サンプルレート診断は :mod:`amtlab.j1939` を参照。

必要な信号(最低限):
    時刻, エンジン回転(SPN 190), 車速(SPN 84), 現在ギヤ(SPN 523)
あると精度が上がる信号:
    シフトインプロセス(574) … 変速イベントを推定でなく直接切り出せる
    選択ギヤ(524) / 実ギヤ比(526) … 変速先とニュートラルが確定する
    インプット/アウトプットシャフト回転(161/191) … 同期・吹け上がりの実測
    クラッチ滑り率(522) … 締結完了の実測
    実トルク率(513) × 参照トルク(544) … トルクホールの実測
    アクセルペダル(91) / 要求トルク率(512) … 運転条件とトルク復帰目標
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from . import j1939
from .features import speed_drop_metrics
from .simulation.plant import SimSettings, _lowpass, simulate_shift
from .simulation.vehicle import KMH_PER_MS, VehicleParams


@dataclass
class SignalMap:
    """ログの列名を標準スキーマに対応付ける。

    各フィールドに列名を書けばその列を使う。指定が無い(または指定した列が
    ログに無い)場合は :func:`amtlab.j1939.detect_columns` で自動検出する。
    """

    # --- 最低限 ---------------------------------------------------------
    time: str | None = "time"
    engine_speed_rpm: str | None = "engine_speed_rpm"
    speed_kmh: str | None = "speed_kmh"
    gear: str | None = "gear"
    # --- J1939 で取れると精度が上がる信号 --------------------------------
    shift_in_process: str | None = None  # SPN 574
    selected_gear: str | None = None  # SPN 524
    gear_ratio: str | None = None  # SPN 526
    input_shaft_rpm: str | None = None  # SPN 161
    output_shaft_rpm: str | None = None  # SPN 191
    clutch_slip_pct: str | None = None  # SPN 522
    accel_pedal_pct: str | None = None  # SPN 91
    demand_torque_pct: str | None = None  # SPN 512
    actual_torque_pct: str | None = None  # SPN 513
    friction_torque_pct: str | None = None  # SPN 514
    reference_torque_nm: str | None = None  # SPN 544
    front_axle_speed_kmh: str | None = None  # SPN 904
    # --- J1939 以外の補助信号 --------------------------------------------
    throttle: str | None = "throttle"
    clutch_position: str | None = None
    shaft_torque: str | None = None
    accel: str | None = None

    #: 指定が無い信号を J1939 の別名から自動検出するか
    auto_detect: bool = True

    @staticmethod
    def from_file(path: str | Path) -> "SignalMap":
        """``{内部名: 列名}`` の YAML / JSON から作る(部分指定で可)。

        自動検出が外れた信号だけを書けばよい。残りは自動検出に任せる。
        """
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {f.name for f in fields(SignalMap)}
        unknown = set(data) - known
        if unknown:
            raise KeyError(
                f"SignalMap に無いキー: {sorted(unknown)} / 使えるキー: {sorted(known)}"
            )
        return SignalMap(**data)

    def required(self) -> list[str]:
        return ["time", "engine_speed_rpm", "speed_kmh"]

    def resolve(self, df: pd.DataFrame) -> dict[str, str]:
        """{内部名: 実際の列名} を解決する。

        明示指定 > 自動検出。明示した列がログに存在しなければ自動検出に委ねる。
        """
        mapping: dict[str, str] = {}
        if self.auto_detect:
            mapping.update(j1939.detect_columns(df))
        for f in fields(self):
            if f.name == "auto_detect":
                continue
            column = getattr(self, f.name)
            if column is not None and column in df.columns:
                mapping[f.name] = column
        missing = [k for k in self.required() if k not in mapping]
        if missing:
            raise KeyError(
                f"ログに必要な信号がありません: {missing} "
                f"(検出できた信号: {sorted(mapping)})"
            )
        if "gear" not in mapping and "shift_in_process" not in mapping:
            raise KeyError(
                "現在ギヤ(SPN 523)もシフトインプロセス(SPN 574)も見つからないため"
                "変速イベントを切り出せません"
            )
        return mapping


@dataclass
class ShiftEvent:
    """検出された 1 変速イベント。"""

    index: int
    from_gear: int
    to_gear: int
    t_start: float  # 変速開始(シフトインプロセス立ち上がり or ギヤ抜き)
    t_end: float  # 締結完了(クラッチすべり収束)
    t_settle: float  # 評価窓の終わり
    t_engage: float = float("nan")  # ギヤ入り(フラグ立ち下がり)
    ratio_to: float = float("nan")  # 変速先の実ギヤ比(ログから取得できた場合)
    source: str = "gear_change"  # 切り出しに使った信号

    @property
    def is_upshift(self) -> bool:
        return self.to_gear > self.from_gear


def standardize(
    df: pd.DataFrame,
    signal_map: SignalMap | None = None,
    resample_hz: float | None = 100.0,
    jerk_cutoff_hz: float = 10.0,
) -> pd.DataFrame:
    """ログを標準スキーマ(等時間間隔・加速度/ジャーク付き)に整形する。"""

    sm = signal_map or SignalMap()
    mapping = sm.resolve(df)
    out = j1939.decode(df, mapping)

    slip_source = out.attrs.get("clutch_slip_source")
    if resample_hz:
        dt = 1.0 / resample_hz
        grid = np.arange(out["time"].iloc[0], out["time"].iloc[-1] + dt * 0.5, dt)
        res = pd.DataFrame({"time": grid})
        src_t = out["time"].to_numpy()
        pos = np.clip(np.searchsorted(src_t, grid), 1, len(src_t) - 1)
        nearest = np.where(grid - src_t[pos - 1] <= src_t[pos] - grid, pos - 1, pos)
        for col in out.columns:
            if col == "time":
                continue
            if col in DISCRETE_CHANNELS:
                # 離散信号は最近傍で補間する(線形補間は幻のギヤ/フラグを生む)
                res[col] = out[col].to_numpy()[nearest]
            else:
                res[col] = np.interp(grid, src_t, out[col].to_numpy())
        for col in DISCRETE_CHANNELS & set(res.columns):
            res[col] = res[col].astype(int)
        out = res

    dt = float(np.median(np.diff(out["time"]))) if len(out) > 1 else 0.01
    _add_driveline_speed(out)
    accel_source, cutoff, jerk_quality = _accel_plan(out, dt, jerk_cutoff_hz)
    if "accel" not in out.columns:
        v = out["speed_kmh"].to_numpy() / KMH_PER_MS
        out["accel"] = np.gradient(_lowpass(v, dt, cutoff), dt)
    out["accel_filt"] = _lowpass(out["accel"].to_numpy(), dt, cutoff)
    out["jerk"] = np.gradient(out["accel_filt"].to_numpy(), dt)
    if "throttle" not in out.columns:
        out["throttle"] = np.nan
    out.attrs["dt"] = dt
    out.attrs["mapping"] = mapping
    out.attrs["clutch_slip_source"] = slip_source or "SPN 522"
    out.attrs["accel_source"] = accel_source
    out.attrs["jerk_quality"] = jerk_quality
    out.attrs["jerk_cutoff_hz"] = cutoff
    return out


def _update_rate(time: np.ndarray, values: np.ndarray) -> float:
    """値が更新される間隔から実効サンプルレートを推定する。"""
    changed = np.flatnonzero(np.diff(values) != 0)
    if len(changed) < 3:
        return 0.0
    dt = float(np.median(np.diff(time[changed + 1])))
    return 1.0 / dt if dt > 0 else 0.0


def _add_driveline_speed(out: pd.DataFrame) -> None:
    """アウトプットシャフト回転(SPN 191)を車速相当に換算した診断用チャネル。

    .. warning::
       これは**車体の速度ではなく駆動系の速度**である。駆動軸のねじり振動
       (シャッフル、2〜8 Hz)がそのまま乗るため、微分して乗り心地の
       ジャークとして扱ってはいけない。振動の可視化・周波数確認に使う。
    """

    if "output_shaft_rpm" not in out.columns or "speed_kmh" not in out.columns:
        return
    rpm = out["output_shaft_rpm"].to_numpy()
    v = out["speed_kmh"].to_numpy()
    usable = np.isfinite(rpm) & np.isfinite(v) & (rpm > 50.0) & (v > 1.0)
    if usable.sum() < 20:
        return
    k = float(np.median(v[usable] / rpm[usable]))
    derived = k * rpm
    error = np.abs(derived[usable] - v[usable]) / np.maximum(v[usable], 1.0)
    if float(np.median(error)) <= 0.05:
        out["driveline_speed_kmh"] = derived


def _accel_plan(
    out: pd.DataFrame, dt: float, jerk_cutoff_hz: float
) -> tuple[str, float, str]:
    """加速度・ジャークの算出方法(信号源・ローパス周波数・信頼度)を決める。

    実測の前後加速度があればそれを使う。無い場合は車速(SPN 84)を微分するが、
    車速の更新周期が低いと階段状の量子化がジャークに化けるため、
    ローパス周波数を実効レートの 1/3 まで下げる。

    ``jerk_quality``:
      ``measured``     加速度センサ実測 — ジャークを定量指標として使える
      ``derived``      50 Hz 以上の車速から算出 — ほぼ使える
      ``low_rate``     車速の更新周期が低い — ジャークは参考値
      ``unusable``     車速の更新が追えない — ジャークは使わないこと
    """

    if "accel" in out.columns:
        return "accel (実測)", jerk_cutoff_hz, "measured"

    time = out["time"].to_numpy()
    column = "speed_kmh"
    rate = _update_rate(time, out["speed_kmh"].to_numpy())
    # 前輪(非駆動輪なら駆動系のねじりを含まない)の方が高レートなら乗り換える
    if "front_axle_speed_kmh" in out.columns:
        front = out["front_axle_speed_kmh"].to_numpy()
        front_rate = _update_rate(time, front)
        usable = np.isfinite(front) & (out["speed_kmh"].to_numpy() > 1.0)
        agree = usable.sum() > 20 and float(
            np.median(
                np.abs(front[usable] - out["speed_kmh"].to_numpy()[usable])
                / np.maximum(out["speed_kmh"].to_numpy()[usable], 1.0)
            )
        ) < 0.02
        if agree and front_rate >= rate * 1.5:
            out["speed_kmh"] = front
            column, rate = "front_axle_speed_kmh", front_rate
    if rate <= 0:
        return "speed_kmh", jerk_cutoff_hz, "unusable"

    cutoff = min(jerk_cutoff_hz, max(rate / 3.0, 0.5), 0.45 / dt)
    if rate >= 50.0:
        quality = "derived"
    elif rate >= 10.0:
        quality = "low_rate"
    else:
        quality = "unusable"
    return f"{column} ({rate:.0f} Hz, lowpass {cutoff:.1f} Hz)", cutoff, quality


#: 補間してはいけない離散チャネル
DISCRETE_CHANNELS = {"gear", "selected_gear", "shift_in_process"}


def _lock_time(
    trace: pd.DataFrame, from_index: int, ratio_to: float,
    lock_tol_rpm: float, lock_tol_pct: float,
) -> float:
    """ギヤが入った後、クラッチが締結し切った時刻を返す。

    優先順位: クラッチ滑り率(SPN 522) > エンジン回転 vs インプットシャフト回転
    > エンジン回転 vs アウトプットシャフト回転 × 実ギヤ比 > 車速からの換算。
    """
    time = trace["time"].to_numpy()
    idx = np.arange(len(time))
    candidates: np.ndarray | None = None

    if "clutch_slip_pct" in trace.columns:
        slip = trace["clutch_slip_pct"].to_numpy()
        candidates = np.abs(slip) < lock_tol_pct
    elif "input_shaft_rpm" in trace.columns:
        slip = trace["engine_speed_rpm"].to_numpy() - trace["input_shaft_rpm"].to_numpy()
        candidates = np.abs(slip) < lock_tol_rpm
    elif "output_shaft_rpm" in trace.columns and np.isfinite(ratio_to):
        target = trace["output_shaft_rpm"].to_numpy() * ratio_to
        candidates = np.abs(trace["engine_speed_rpm"].to_numpy() - target) < lock_tol_rpm

    if candidates is None:
        return float("nan")
    hit = np.flatnonzero(candidates & (idx >= from_index))
    return float(time[hit[0]]) if len(hit) else float("nan")


def _ratio_after(trace: pd.DataFrame, index: int, window: int = 50) -> float:
    """ギヤが入った直後の実ギヤ比(SPN 526)= 変速先ギヤの**変速機内**の比。

    SPN 526 は入力軸回転 / 出力軸回転なので終減速比を含まない。車速からの
    換算に使う総減速比(``VehicleParams.transmission.total_ratio``)とは
    別物なので混同しないこと。
    """
    if "gear_ratio" not in trace.columns:
        return float("nan")
    seg = trace["gear_ratio"].to_numpy()[index : index + window]
    seg = seg[np.isfinite(seg) & (np.abs(seg) > 1e-3)]
    return float(np.median(seg)) if len(seg) else float("nan")


def _events_from_flag(
    trace: pd.DataFrame, settle_time: float, lock_tol_rpm: float, lock_tol_pct: float,
) -> list[ShiftEvent]:
    """SPN 574(シフトインプロセス)の立ち上がり/立ち下がりで切り出す。"""

    flag = trace["shift_in_process"].to_numpy().astype(int)
    gear = trace["gear"].to_numpy() if "gear" in trace.columns else None
    selected = trace["selected_gear"].to_numpy() if "selected_gear" in trace.columns else None
    time = trace["time"].to_numpy()

    edges = np.diff(flag)
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if flag[0] == 1:  # ログ開始時点で変速中の場合は捨てる
        ends = ends[ends > starts[0]] if len(starts) else ends[1:]
    events: list[ShiftEvent] = []
    for i0 in starts:
        after = ends[ends > i0]
        if not len(after):
            break  # 最後の変速が完了していない
        i1 = int(after[0])

        g_from = _last_engaged(gear, i0) if gear is not None else 0
        g_to = _first_engaged(gear, i1) if gear is not None else 0
        if selected is not None and g_to <= 0:
            during = selected[i0:i1]
            during = during[during > 0]
            g_to = int(np.bincount(during.astype(int)).argmax()) if len(during) else 0
        if g_from <= 0 or g_to <= 0 or g_from == g_to:
            continue  # 変速中断・ニュートラル抜けなど

        ratio_to = _ratio_after(trace, i1)
        t_lock = _lock_time(trace, i1, ratio_to, lock_tol_rpm, lock_tol_pct)
        t_end = t_lock if np.isfinite(t_lock) else float(time[i1])
        events.append(
            ShiftEvent(
                index=len(events),
                from_gear=int(g_from),
                to_gear=int(g_to),
                t_start=float(time[i0]),
                t_end=t_end,
                t_settle=min(t_end + settle_time, float(time[-1])),
                t_engage=float(time[i1]),
                ratio_to=ratio_to,
                source="shift_in_process",
            )
        )
    return events


def _last_engaged(gear: np.ndarray, index: int) -> int:
    """index より前で最後に噛んでいたギヤ。"""
    prior = gear[:index]
    engaged = np.flatnonzero(prior > 0)
    return int(prior[engaged[-1]]) if len(engaged) else 0


def _first_engaged(gear: np.ndarray, index: int) -> int:
    """index 以降で最初に噛んだギヤ。"""
    later = gear[index:]
    engaged = np.flatnonzero(later > 0)
    return int(later[engaged[0]]) if len(engaged) else 0


def _events_from_gear(
    trace: pd.DataFrame, vehicle: VehicleParams, settle_time: float,
    lock_tol_rpm: float, lock_tol_pct: float,
) -> list[ShiftEvent]:
    """ギヤ信号の遷移から切り出す(シフトインプロセスが無い場合)。"""

    gear = trace["gear"].to_numpy()
    time = trace["time"].to_numpy()
    events: list[ShiftEvent] = []

    change_idx = np.flatnonzero(np.diff(gear) != 0)
    idx = 0
    n = len(change_idx)
    while idx < n:
        i = int(change_idx[idx])
        g_from = int(gear[i])
        if g_from <= 0:
            idx += 1
            continue
        j = idx
        g_to = int(gear[i + 1])
        while g_to <= 0 and j + 1 < n:
            j += 1
            g_to = int(gear[int(change_idx[j]) + 1])
        if g_to <= 0 or g_to == g_from:
            idx = j + 1
            continue

        engage_idx = int(change_idx[j]) + 1
        ratio_to = _ratio_after(trace, engage_idx)  # SPN 526(無ければ NaN)
        t_lock = _lock_time(trace, engage_idx, ratio_to, lock_tol_rpm, lock_tol_pct)
        if not np.isfinite(t_lock):
            # 車速から入力軸回転を換算して締結を判定する
            w_in_rpm = (
                trace["speed_kmh"].to_numpy() / KMH_PER_MS / vehicle.wheel_radius * ratio_to
            ) * (60.0 / (2.0 * np.pi))
            slip = np.abs(trace["engine_speed_rpm"].to_numpy() - w_in_rpm)
            locked = np.flatnonzero((slip < lock_tol_rpm) & (np.arange(len(slip)) >= engage_idx))
            t_lock = (
                float(time[locked[0]])
                if len(locked)
                else float(time[min(engage_idx, len(time) - 1)])
            )
        events.append(
            ShiftEvent(
                index=len(events),
                from_gear=g_from,
                to_gear=g_to,
                t_start=float(time[i]),
                t_end=float(t_lock),
                t_settle=min(float(t_lock) + settle_time, float(time[-1])),
                t_engage=float(time[engage_idx]),
                ratio_to=ratio_to,
                source="gear_change",
            )
        )
        idx = j + 1
    return events


def detect_shift_events(
    trace: pd.DataFrame,
    vehicle: VehicleParams | None = None,
    settle_time: float = 0.6,
    lock_tol_rpm: float = 60.0,
    lock_tol_pct: float = 1.0,
) -> list[ShiftEvent]:
    """変速イベントを切り出す。

    SPN 574(シフトインプロセス)があればそれを使う。無ければギヤ信号の
    遷移(ニュートラル経由に対応)から推定する。
    """

    veh = vehicle or VehicleParams()
    if "shift_in_process" in trace.columns and int(trace["shift_in_process"].sum()) > 0:
        return _events_from_flag(trace, settle_time, lock_tol_rpm, lock_tol_pct)
    if "gear" not in trace.columns:
        return []
    return _events_from_gear(trace, veh, settle_time, lock_tol_rpm, lock_tol_pct)


#: クラッチが締結しているとみなすすべり率 [%]
CLUTCH_ENGAGED_PCT = 1.0
#: クラッチが完全に切れているとみなすすべり率 [%]
CLUTCH_RELEASED_PCT = 90.0


def clutch_metrics(
    trace: pd.DataFrame,
    event: ShiftEvent,
    engaged_pct: float = CLUTCH_ENGAGED_PCT,
    released_pct: float = CLUTCH_RELEASED_PCT,
    lead_time: float = 0.2,
    hold_time: float = 0.05,
) -> dict[str, float]:
    """変速中のクラッチ ON→OFF→ON の切り方を測る。

    「変速は短いのにクラッチはしっかり切れている」といった操作の質を
    数値化するための指標。すべり率(SPN 522)の 1 サイクルを

        締結 → 切り始め(release) → 全切り(open) → 繋ぎ(engage) → 締結

    に分解する。

    Returns
    -------
    clutch_cycle_time_s
        ON→OFF→ON の全体時間。「ギヤチェンジ中の ON-OFF の時間」。
    clutch_release_time_s / clutch_engage_time_s
        切り始めから全切りまで / 全切り解除から再締結まで。
    clutch_open_time_s
        全切り(すべり率が released_pct 以上)の時間。
    clutch_full_release_ratio
        サイクル時間に占める全切り時間の割合。**短時間でもしっかり切って
        いるか**を表す(1 に近いほど深く切っている)。
    clutch_mean_slip_pct
        サイクル平均のすべり率。すべり面積 ÷ サイクル時間。
    clutch_release_rate_pct_s / clutch_engage_rate_pct_s
        切り / 繋ぎのアクチュエータ速度。
    """

    if "clutch_slip_pct" not in trace.columns:
        return {}

    source = str(trace.attrs.get("clutch_slip_source", "SPN 522"))
    dt = float(trace.attrs.get("dt", np.median(np.diff(trace["time"]))))
    window = trace[
        (trace["time"] >= event.t_start - lead_time) & (trace["time"] <= event.t_settle)
    ]
    if len(window) < 5:
        return {}

    time = window["time"].to_numpy()
    slip = np.abs(np.nan_to_num(window["clutch_slip_pct"].to_numpy(), nan=0.0))
    above = slip > engaged_pct
    if not above.any():
        return {"clutch_engaged_throughout": 1.0, "clutch_slip_source": source}

    start = int(np.argmax(above))
    # 締結に戻った点: engaged 以下が hold_time 続いた最初の位置(ノイズ耐性)
    hold = max(int(round(hold_time / dt)), 1)
    end = len(slip) - 1
    below = ~above
    for i in range(start + 1, len(slip) - hold + 1):
        if below[i : i + hold].all():
            end = i
            break

    cycle = slip[start : end + 1]
    cycle_time = float(time[end] - time[start])
    open_mask = cycle >= released_pct
    open_time = float(open_mask.sum() * dt)
    integral = float(np.trapezoid(cycle, dx=dt))

    metrics = {
        "clutch_cycle_time_s": cycle_time,
        "clutch_open_time_s": open_time,
        "clutch_full_release_ratio": open_time / cycle_time if cycle_time > 0 else np.nan,
        "clutch_mean_slip_pct": integral / cycle_time if cycle_time > 0 else np.nan,
        "clutch_peak_slip_pct": float(np.max(cycle)),
        "clutch_slip_integral_pct_s": integral,
        "clutch_engaged_throughout": 0.0,
        "clutch_slip_source": source,
    }

    if open_mask.any():
        first_open = int(np.argmax(open_mask))
        last_open = len(open_mask) - 1 - int(np.argmax(open_mask[::-1]))
        release_time = float(time[start + first_open] - time[start])
        engage_time = float(time[end] - time[start + last_open])
        metrics.update(
            {
                "clutch_release_time_s": release_time,
                "clutch_engage_time_s": engage_time,
                "clutch_release_rate_pct_s": (
                    float(cycle[first_open] - cycle[0]) / release_time
                    if release_time > 0 else np.nan
                ),
                "clutch_engage_rate_pct_s": (
                    float(cycle[last_open]) / engage_time if engage_time > 0 else np.nan
                ),
            }
        )
    else:  # 全切りに達していない(半クラのまま変速している)
        metrics.update(
            {
                "clutch_release_time_s": np.nan,
                "clutch_engage_time_s": np.nan,
                "clutch_release_rate_pct_s": np.nan,
                "clutch_engage_rate_pct_s": np.nan,
            }
        )
    return metrics


def clutch_profile(
    trace: pd.DataFrame,
    event: ShiftEvent,
    n_points: int = 160,
    lead_time: float = 0.2,
    span: float = 2.5,
) -> pd.DataFrame:
    """変速開始を 0 秒に揃えたクラッチすべり率の波形(重ね描き用)。

    時系列そのものは捨てても、この短い区間だけ残しておけば
    「切り方の形」をファイル横断で比較できる。時間軸は
    ``-lead_time`` 〜 ``span`` の**固定グリッド**なので、イベント長が
    違っても素直に重ねられる(区間外は NaN)。
    """

    if "clutch_slip_pct" not in trace.columns:
        return pd.DataFrame()
    window = trace[
        (trace["time"] >= event.t_start - lead_time) & (trace["time"] <= event.t_settle)
    ]
    if len(window) < 5:
        return pd.DataFrame()

    rel = window["time"].to_numpy() - event.t_start
    grid = np.linspace(-lead_time, span, n_points)
    inside = (grid >= rel[0]) & (grid <= rel[-1])
    out = pd.DataFrame({"rel_time_s": grid})
    for column in ("clutch_slip_pct", "engine_speed_rpm", "speed_kmh"):
        if column in window.columns:
            values = np.interp(grid, rel, window[column].to_numpy())
            out[column] = np.where(inside, values, np.nan)
    out["event_id"] = event.index
    out["current_gear"] = event.from_gear
    out["to_gear"] = event.to_gear
    return out


def event_kpis(trace: pd.DataFrame, event: ShiftEvent,
               vehicle: VehicleParams | None = None) -> dict[str, float]:
    """ログの 1 イベントからシミュレーションと同じ KPI を計算する。

    測定できている信号を優先して使い、無い分だけ推定で補う。
    どの信号を使ったかは ``torque_source`` / ``flare_source`` に記録する。
    """

    veh = vehicle or VehicleParams()
    dt = float(trace.attrs.get("dt", np.median(np.diff(trace["time"]))))
    win = trace[(trace["time"] >= event.t_start) & (trace["time"] <= event.t_settle)]
    pre = trace[(trace["time"] >= event.t_start - 0.15) & (trace["time"] < event.t_start)]
    if len(win) < 3:
        return {}

    jerk = win["jerk"].to_numpy()
    accel = win["accel"].to_numpy()
    pre_accel = float(pre["accel"].mean()) if len(pre) else float(accel[0])
    pre_speed = float(pre["speed_kmh"].mean()) if len(pre) else float(win["speed_kmh"].iloc[0])

    # --- 吹け上がり: 変速先ギヤでの入力軸相当回転を基準にする ----------------
    ratio_to = event.ratio_to
    if "output_shaft_rpm" in win.columns and np.isfinite(ratio_to):
        target_rpm = win["output_shaft_rpm"].to_numpy() * ratio_to
        flare_source = "output_shaft_rpm x gear_ratio"
    else:
        # 車速からの換算には終減速比を含む総減速比を使う(SPN 526 ではない)
        total_ratio = veh.transmission.total_ratio(event.to_gear)
        target_rpm = (
            win["speed_kmh"].to_numpy() / KMH_PER_MS / veh.wheel_radius * total_ratio
        ) * (60.0 / (2.0 * np.pi))
        flare_source = "speed_kmh (推定)"
    start_rpm = float(pre["engine_speed_rpm"].mean()) if len(pre) else float(
        win["engine_speed_rpm"].iloc[0]
    )
    flare = win["engine_speed_rpm"].to_numpy() - np.maximum(target_rpm, start_rpm)

    speed = speed_drop_metrics(
        win["time"].to_numpy(), win["speed_kmh"].to_numpy(), pre_accel
    )

    kpis = {
        "event_id": float(event.index),
        "current_gear": float(event.from_gear),
        "from_gear": float(event.from_gear),
        "to_gear": float(event.to_gear),
        "is_upshift": float(event.is_upshift),
        "t_start": float(event.t_start),
        "t_end": float(event.t_end),
        "speed_kmh": pre_speed,
        "throttle": float(pre["throttle"].mean()) if "throttle" in pre else np.nan,
        "shift_time_s": float(event.t_end - event.t_start),
        **speed,
        "jerk_rms": float(np.sqrt(np.mean(jerk**2))),
        "jerk_peak": float(np.max(np.abs(jerk))),
        "accel_drop": float(pre_accel - np.min(accel)),
        "min_accel": float(np.min(accel)),
        "engine_flare_rpm": float(max(np.max(flare), 0.0)),
        "neutral_time_s": float((win["gear"].to_numpy() <= 0).sum() * dt)
        if "gear" in win.columns else np.nan,
        "completed": 1.0,
    }

    if np.isfinite(event.t_engage):
        kpis["gear_engage_time_s"] = float(event.t_engage - event.t_start)
        kpis["clutch_close_time_s"] = float(event.t_end - event.t_engage)

    # --- トルクホール: 実測トルクがあれば使う ------------------------------
    torque_col = next(
        (c for c in ("est_wheel_torque_nm", "shaft_torque", "engine_torque_nm")
         if c in win.columns),
        None,
    )
    if torque_col and len(pre):
        pre_torque = float(pre[torque_col].mean())
        thr = 0.2 * pre_torque
        kpis["pre_shaft_torque_nm"] = pre_torque
        kpis["torque_source"] = torque_col
        kpis["torque_interrupt_s"] = (
            float((win[torque_col].to_numpy() < thr).sum() * dt)
            if pre_torque > 0
            else np.nan
        )
    if "demand_torque_nm" in win.columns and "engine_torque_nm" in win.columns:
        # トルク復帰の遅れ: 要求に対して実トルクが 90% まで戻るまでの時間
        demand = win["demand_torque_nm"].to_numpy()
        actual = win["engine_torque_nm"].to_numpy()
        time = win["time"].to_numpy()
        recovered = np.flatnonzero((demand > 1.0) & (actual >= 0.9 * demand))
        after_engage = time[recovered] if len(recovered) else np.array([])
        after_engage = after_engage[after_engage >= event.t_engage] if np.isfinite(
            event.t_engage
        ) else after_engage
        kpis["torque_recovery_s"] = (
            float(after_engage[0] - event.t_start) if len(after_engage) else np.nan
        )

    if "clutch_slip_pct" in win.columns:
        # ギヤが入ってからの滑り = 実際にトルクを伝えながら滑っている時間
        after = win["time"].to_numpy() >= (
            event.t_engage if np.isfinite(event.t_engage) else event.t_start
        )
        slip = np.abs(win["clutch_slip_pct"].to_numpy())
        kpis["clutch_slip_time_s"] = float((slip[after] > 1.0).sum() * dt)
        kpis["clutch_slip_peak_pct"] = float(np.nanmax(slip[after])) if after.any() else np.nan
        kpis.update(clutch_metrics(trace, event))

    kpis["flare_source"] = flare_source
    kpis["detection_source"] = event.source
    kpis["jerk_quality"] = str(trace.attrs.get("jerk_quality", "unknown"))
    return kpis


def analyze_log(
    source: str | Path | pd.DataFrame,
    signal_map: SignalMap | None = None,
    vehicle: VehicleParams | None = None,
    resample_hz: float | None = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ログを読み込み、(整形済み時系列, イベント別 KPI 表) を返す。

    ``source`` はパス(CSV / parquet / feather)でも DataFrame でもよい。
    """

    if isinstance(source, pd.DataFrame):
        df = source
    else:
        from .batch import read_log  # 循環 import を避けるため関数内で読む

        df = read_log(source)
    trace = standardize(df, signal_map, resample_hz=resample_hz)
    events = detect_shift_events(trace, vehicle)
    rows = [event_kpis(trace, ev, vehicle) for ev in events]
    kpi_df = pd.DataFrame([r for r in rows if r])
    return trace, kpi_df


# ----------------------------------------------------------------------
def make_demo_log(
    n_shifts: int = 4,
    start_speed_kmh: float = 25.0,
    start_gear: int = 1,
    sample_hz: float = 100.0,
    seed: int = 0,
    controls: list | None = None,
) -> pd.DataFrame:
    """シミュレーションを連結して、実ログ相当の連続時系列を作る。

    各セグメントの終端車速・ギヤを次のセグメントの初期値に引き継ぐので、
    信号が連続した 1 本の加速ログになる。取り込み機能(``analyze_log``)の
    動作確認・デモに使う。
    """

    from .simulation.controller import ShiftControlParams
    from .simulation.plant import ShiftScenario

    rng = np.random.default_rng(seed)
    settings = SimSettings(post_time=1.0)
    step = max(int(round((1.0 / sample_hz) / settings.dt)), 1)

    frames: list[pd.DataFrame] = []
    speed = float(start_speed_kmh)
    gear = int(start_gear)
    t_offset = 0.0
    n_gears = VehicleParams().transmission.n_gears()

    for i in range(n_shifts):
        if gear + 1 > n_gears:
            break
        control = controls[i] if controls else ShiftControlParams.sample(rng)
        scenario = ShiftScenario(
            gear, gear + 1, speed_kmh=speed, throttle=float(rng.uniform(0.4, 0.9))
        )
        res = simulate_shift(control, scenario, settings=settings)
        tr = res.trace.iloc[::step][
            ["time", "engine_speed_rpm", "speed_kmh", "gear", "shaft_torque"]
        ].copy()
        tr["throttle"] = scenario.throttle
        tr["time"] = tr["time"] - tr["time"].iloc[0] + t_offset
        t_offset = float(tr["time"].iloc[-1]) + 1.0 / sample_hz
        speed = float(tr["speed_kmh"].iloc[-1])
        gear += 1
        frames.append(tr)

    log = pd.concat(frames, ignore_index=True)
    log["engine_speed_rpm"] = log["engine_speed_rpm"].round(1)
    return log


# ----------------------------------------------------------------------
#: J1939 デモログの列名(標準的な DBC の短縮形。``Transmission`` → ``Trans``、
#: ``Engine`` → ``Eng`` など)
J1939_DEMO_COLUMNS = {
    "time": "Timestamp",
    "accel_pedal_pct": "EEC2::AccelPedalPos1",
    "demand_torque_pct": "EEC1::DriversDemandEngPercentTorque",
    "actual_torque_pct": "EEC1::ActualEngPercentTorque",
    "reference_torque_nm": "EC1::EngReferenceTorque",
    "engine_speed_rpm": "EEC1::EngSpeed",
    "speed_kmh": "CCVS1::WheelBasedVehicleSpeed",
    "front_axle_speed_kmh": "EBC2::FrontAxleSpeed",
    "input_shaft_rpm": "ETC1::TransInputShaftSpeed",
    "output_shaft_rpm": "ETC1::TransOutputShaftSpeed",
    "clutch_slip_pct": "ETC1::PercentClutchSlip",
    "shift_in_process": "ETC1::TransShiftInProcess",
    "gear": "ETC2::TransCurrentGear",
    "selected_gear": "ETC2::TransSelectedGear",
    "gear_ratio": "ETC2::TransActualGearRatio",
}

#: 略さない書き方のロガーもあるので、こちらも生成できるようにしておく
J1939_DEMO_COLUMNS_LONG = {
    "time": "Timestamp",
    "accel_pedal_pct": "EEC2::AcceleratorPedalPosition1",
    "demand_torque_pct": "EEC1::DriversDemandEnginePercentTorque",
    "actual_torque_pct": "EEC1::ActualEnginePercentTorque",
    "reference_torque_nm": "EC1::EngineReferenceTorque",
    "engine_speed_rpm": "EEC1::EngineSpeed",
    "speed_kmh": "CCVS1::WheelBasedVehicleSpeed",
    "front_axle_speed_kmh": "EBC2::FrontAxleSpeed",
    "input_shaft_rpm": "ETC1::TransmissionInputShaftSpeed",
    "output_shaft_rpm": "ETC1::TransmissionOutputShaftSpeed",
    "clutch_slip_pct": "ETC1::PercentClutchSlip",
    "shift_in_process": "ETC1::TransmissionShiftInProcess",
    "gear": "ETC2::TransmissionCurrentGear",
    "selected_gear": "ETC2::TransmissionSelectedGear",
    "gear_ratio": "ETC2::TransmissionActualGearRatio",
}

#: 各信号の代表的な更新周期 [Hz](PGN ごとに違う)
J1939_DEMO_RATES = {
    "accel_pedal_pct": 20.0,
    "demand_torque_pct": 50.0,
    "actual_torque_pct": 50.0,
    "reference_torque_nm": 1.0,
    "engine_speed_rpm": 50.0,
    "speed_kmh": 10.0,
    "front_axle_speed_kmh": 20.0,
    "input_shaft_rpm": 50.0,
    "output_shaft_rpm": 50.0,
    "clutch_slip_pct": 50.0,
    "shift_in_process": 50.0,
    "gear": 20.0,
    "selected_gear": 20.0,
    "gear_ratio": 20.0,
}


def _hold(time: np.ndarray, values: np.ndarray, rate_hz: float) -> np.ndarray:
    """指定周期でサンプル&ホールドする(CAN の更新周期を模擬)。"""
    if rate_hz <= 0:
        return values
    step = max(int(round((1.0 / rate_hz) / max(np.median(np.diff(time)), 1e-9))), 1)
    idx = (np.arange(len(values)) // step) * step
    return values[np.clip(idx, 0, len(values) - 1)]


def make_j1939_demo_log(
    n_shifts: int = 4,
    start_speed_kmh: float = 25.0,
    start_gear: int = 1,
    sample_hz: float = 100.0,
    seed: int = 0,
    naming: str = "short",
    controls: list | None = None,
) -> pd.DataFrame:
    """シミュレーション結果を **J1939 の信号名・単位・更新周期** に変換した
    デモログを作る。取り込み経路(自動検出 → イベント切り出し)の確認用。

    ``naming="short"`` は標準的な DBC の短縮形(``ETC1::TransShiftInProcess``)、
    ``"long"`` は略さない形(``ETC1::TransmissionShiftInProcess``)。

    実車ログを模して次を再現する。

      * PGN ごとに異なる更新周期(車速 10 Hz、EEC1 50 Hz など)のホールド
      * 車速の 1/256 km/h 量子化
      * トルクは参照トルク(SPN 544)に対する % で持つ
      * ニュートラルでは現在ギヤ 0・実ギヤ比 0
      * シフトインプロセス(SPN 574)はギヤが入った時点で 0 に戻る
        (クラッチ締結とトルク復帰はフラグが落ちた後も続く)
    """

    from .simulation.controller import ShiftControlParams
    from .simulation.plant import ShiftScenario

    rng = np.random.default_rng(seed)
    veh = VehicleParams()
    eng, trm = veh.engine, veh.transmission
    settings = SimSettings(post_time=1.0)
    step = max(int(round((1.0 / sample_hz) / settings.dt)), 1)
    reference_torque = float(max(eng.wot_torque))

    frames: list[pd.DataFrame] = []
    speed = float(start_speed_kmh)
    gear = int(start_gear)
    t_offset = 0.0

    for i in range(n_shifts):
        if gear + 1 > trm.n_gears():
            break
        control = controls[i % len(controls)] if controls else ShiftControlParams.sample(rng)
        scenario = ShiftScenario(
            gear, gear + 1, speed_kmh=speed, throttle=float(rng.uniform(0.4, 0.9))
        )
        res = simulate_shift(control, scenario, settings=settings)
        tr = res.trace.iloc[::step].reset_index(drop=True)

        w_e = tr["engine_speed_rpm"].to_numpy()
        in_gear = tr["gear"].to_numpy() > 0
        # 出力軸回転 = 車輪回転 x 終減速比
        out_rpm = tr["wheel_speed"].to_numpy() * trm.final_drive * 60.0 / (2.0 * np.pi)
        gearbox_ratio = np.where(
            in_gear,
            [trm.gear_ratios[int(g) - 1] if g > 0 else 0.0 for g in tr["gear"]],
            0.0,
        )
        input_rpm = np.where(in_gear, out_rpm * gearbox_ratio, np.nan)
        # ニュートラル中の入力軸はクラッチが切れて自由回転する: 前後を線形に繋ぐ
        input_series = pd.Series(input_rpm).interpolate(limit_direction="both")
        # すべり率: 回転差ベースと、クラッチストロークから決まる「切れ具合」の
        # 大きい方。実車の TCU もクラッチが離れていれば 100% を出す。
        speed_slip = np.clip(
            np.abs(w_e - input_series.to_numpy()) / np.maximum(w_e, 1.0) * 100.0, 0.0, 100.0
        )
        kiss = trm.clutch_kiss_point
        engagement = np.clip(
            (tr["clutch_position"].to_numpy() - kiss) / (1.0 - kiss), 0.0, 1.0
        )
        slip_pct = np.maximum(speed_slip, (1.0 - engagement) * 100.0)

        demand = np.array(
            [eng.steady_torque(scenario.throttle, w / 9.5493) for w in w_e]
        )
        phase = tr["phase"].to_numpy()
        shifting = np.isin(
            phase, ["torque_reduce", "clutch_open", "gear_out", "speed_sync", "gear_in"]
        ).astype(int)
        selected = np.where(shifting == 1, scenario.to_gear, tr["gear"].to_numpy())

        seg = pd.DataFrame(
            {
                "time": tr["time"].to_numpy() - tr["time"].iloc[0] + t_offset,
                "accel_pedal_pct": np.full(len(tr), scenario.throttle * 100.0),
                "demand_torque_pct": demand / reference_torque * 100.0,
                "actual_torque_pct": tr["engine_torque"].to_numpy() / reference_torque * 100.0,
                "reference_torque_nm": np.full(len(tr), reference_torque),
                "engine_speed_rpm": w_e,
                "speed_kmh": tr["speed_kmh"].to_numpy(),
                "front_axle_speed_kmh": tr["speed_kmh"].to_numpy(),
                "input_shaft_rpm": input_series.to_numpy(),
                "output_shaft_rpm": out_rpm,
                "clutch_slip_pct": slip_pct,
                "shift_in_process": shifting,
                "gear": tr["gear"].to_numpy().astype(int),
                "selected_gear": selected.astype(int),
                "gear_ratio": gearbox_ratio,
            }
        )
        t_offset = float(seg["time"].iloc[-1]) + 1.0 / sample_hz
        speed = float(seg["speed_kmh"].iloc[-1])
        gear += 1
        frames.append(seg)

    log = pd.concat(frames, ignore_index=True)

    # PGN ごとの更新周期でホールド + 分解能の量子化
    time = log["time"].to_numpy()
    for key, rate in J1939_DEMO_RATES.items():
        log[key] = _hold(time, log[key].to_numpy(), rate)
    log["speed_kmh"] = np.round(log["speed_kmh"] * 256.0) / 256.0  # 1/256 km/h
    log["front_axle_speed_kmh"] = np.round(log["front_axle_speed_kmh"] * 256.0) / 256.0
    log["engine_speed_rpm"] = np.round(log["engine_speed_rpm"] * 8.0) / 8.0  # 0.125 rpm
    log["input_shaft_rpm"] = np.round(log["input_shaft_rpm"] * 8.0) / 8.0
    log["output_shaft_rpm"] = np.round(log["output_shaft_rpm"] * 8.0) / 8.0
    log["gear_ratio"] = np.round(log["gear_ratio"], 3)  # 0.001
    for col in ("accel_pedal_pct",):
        log[col] = np.round(log[col] / 0.4) * 0.4  # 0.4 %/bit
    for col in ("demand_torque_pct", "actual_torque_pct"):
        log[col] = np.round(log[col])  # 1 %/bit

    columns = J1939_DEMO_COLUMNS if naming == "short" else J1939_DEMO_COLUMNS_LONG
    return log.rename(columns=columns)
