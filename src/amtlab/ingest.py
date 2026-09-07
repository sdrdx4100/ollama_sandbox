"""実車/台上ログの取り込みと変速イベントの自動切り出し。

シミュレーションで作ったデータと同じ KPI スキーマに正規化するので、
``modeling`` / ``viz`` / ``calibration`` のコードをそのまま実ログに適用できる。

必要な信号(最低限):
    時刻, エンジン回転, 車速, 選択ギヤ
あると精度が上がる信号:
    スロットル(アクセル開度), クラッチストローク, 駆動軸トルク, 前後加速度
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .simulation.plant import SimSettings, _lowpass, simulate_shift
from .simulation.vehicle import KMH_PER_MS, VehicleParams


@dataclass
class SignalMap:
    """ログの列名を標準スキーマに対応付ける。"""

    time: str = "time"
    engine_speed_rpm: str = "engine_speed_rpm"
    speed_kmh: str = "speed_kmh"
    gear: str = "gear"
    throttle: str | None = "throttle"
    clutch_position: str | None = None
    shaft_torque: str | None = None
    accel: str | None = None

    def required(self) -> list[str]:
        return [self.time, self.engine_speed_rpm, self.speed_kmh, self.gear]


@dataclass
class ShiftEvent:
    """検出された 1 変速イベント。"""

    index: int
    from_gear: int
    to_gear: int
    t_start: float
    t_end: float
    t_settle: float

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
    missing = [c for c in sm.required() if c not in df.columns]
    if missing:
        raise KeyError(f"ログに必要な列がありません: {missing}")

    out = pd.DataFrame(
        {
            "time": df[sm.time].astype(float).to_numpy(),
            "engine_speed_rpm": df[sm.engine_speed_rpm].astype(float).to_numpy(),
            "speed_kmh": df[sm.speed_kmh].astype(float).to_numpy(),
            "gear": df[sm.gear].fillna(0).astype(int).to_numpy(),
        }
    )
    for attr, col in (
        ("throttle", "throttle"),
        ("clutch_position", "clutch_position"),
        ("shaft_torque", "shaft_torque"),
        ("accel", "accel"),
    ):
        src = getattr(sm, attr)
        if src and src in df.columns:
            out[col] = df[src].astype(float).to_numpy()

    out = out.sort_values("time").reset_index(drop=True)
    out["time"] -= out["time"].iloc[0]

    if resample_hz:
        dt = 1.0 / resample_hz
        grid = np.arange(out["time"].iloc[0], out["time"].iloc[-1] + dt * 0.5, dt)
        res = pd.DataFrame({"time": grid})
        for col in out.columns:
            if col == "time":
                continue
            if col == "gear":
                # ギヤは離散信号なので最近傍で補間する(線形補間は幻のギヤを生む)
                src_t = out["time"].to_numpy()
                pos = np.clip(np.searchsorted(src_t, grid), 1, len(src_t) - 1)
                nearest = np.where(
                    grid - src_t[pos - 1] <= src_t[pos] - grid, pos - 1, pos
                )
                res[col] = out[col].to_numpy()[nearest].astype(int)
            else:
                res[col] = np.interp(grid, out["time"], out[col])
        out = res

    dt = float(np.median(np.diff(out["time"]))) if len(out) > 1 else 0.01
    if "accel" not in out.columns:
        v = out["speed_kmh"].to_numpy() / KMH_PER_MS
        out["accel"] = np.gradient(_lowpass(v, dt, jerk_cutoff_hz), dt)
    out["accel_filt"] = _lowpass(out["accel"].to_numpy(), dt, jerk_cutoff_hz)
    out["jerk"] = np.gradient(out["accel_filt"].to_numpy(), dt)
    if "throttle" not in out.columns:
        out["throttle"] = np.nan
    out.attrs["dt"] = dt
    return out


def detect_shift_events(
    trace: pd.DataFrame,
    vehicle: VehicleParams | None = None,
    settle_time: float = 0.6,
    lock_tol_rpm: float = 60.0,
) -> list[ShiftEvent]:
    """ギヤ信号の遷移から変速イベントを切り出す(ニュートラル経由に対応)。"""

    veh = vehicle or VehicleParams()
    gear = trace["gear"].to_numpy()
    time = trace["time"].to_numpy()
    events: list[ShiftEvent] = []

    # ギヤが変化する時刻(0 = ニュートラルは遷移中とみなす)
    change_idx = np.flatnonzero(np.diff(gear) != 0)
    idx = 0
    n = len(change_idx)
    while idx < n:
        i = int(change_idx[idx])
        g_from = int(gear[i])
        if g_from <= 0:
            idx += 1
            continue
        # ニュートラルを挟む場合は次に有効ギヤが入るまで進める
        j = idx
        g_to = int(gear[i + 1])
        while g_to <= 0 and j + 1 < n:
            j += 1
            g_to = int(gear[int(change_idx[j]) + 1])
        if g_to <= 0 or g_to == g_from:
            idx = j + 1
            continue

        t_start = float(time[i])
        engage_idx = int(change_idx[j]) + 1
        # 締結完了 = 入力軸回転とエンジン回転が一致した時点
        ratio = veh.transmission.total_ratio(g_to)
        w_in_rpm = (
            trace["speed_kmh"].to_numpy() / KMH_PER_MS / veh.wheel_radius * ratio
        ) * (60.0 / (2.0 * np.pi))
        slip = np.abs(trace["engine_speed_rpm"].to_numpy() - w_in_rpm)
        locked = np.flatnonzero((slip < lock_tol_rpm) & (np.arange(len(slip)) >= engage_idx))
        t_end = (
            float(time[locked[0]])
            if len(locked)
            else float(time[min(engage_idx, len(time) - 1)])
        )
        events.append(
            ShiftEvent(
                index=len(events),
                from_gear=g_from,
                to_gear=g_to,
                t_start=t_start,
                t_end=t_end,
                t_settle=min(t_end + settle_time, float(time[-1])),
            )
        )
        idx = j + 1
    return events


def event_kpis(trace: pd.DataFrame, event: ShiftEvent,
               vehicle: VehicleParams | None = None) -> dict[str, float]:
    """ログの 1 イベントからシミュレーションと同じ KPI を計算する。"""

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

    ratio_to = veh.transmission.total_ratio(event.to_gear)
    w_in_rpm = (
        win["speed_kmh"].to_numpy() / KMH_PER_MS / veh.wheel_radius * ratio_to
    ) * (60.0 / (2.0 * np.pi))
    start_rpm = float(pre["engine_speed_rpm"].mean()) if len(pre) else float(
        win["engine_speed_rpm"].iloc[0]
    )
    flare = win["engine_speed_rpm"].to_numpy() - np.maximum(w_in_rpm, start_rpm)

    kpis = {
        "event_id": float(event.index),
        "from_gear": float(event.from_gear),
        "to_gear": float(event.to_gear),
        "is_upshift": float(event.is_upshift),
        "t_start": float(event.t_start),
        "t_end": float(event.t_end),
        "speed_kmh": pre_speed,
        "throttle": float(pre["throttle"].mean()) if "throttle" in pre else np.nan,
        "shift_time_s": float(event.t_end - event.t_start),
        "jerk_rms": float(np.sqrt(np.mean(jerk**2))),
        "jerk_peak": float(np.max(np.abs(jerk))),
        "accel_drop": float(pre_accel - np.min(accel)),
        "min_accel": float(np.min(accel)),
        "speed_loss_kmh": max(float(pre_speed - np.min(win["speed_kmh"].to_numpy())), 0.0),
        "engine_flare_rpm": float(max(np.max(flare), 0.0)),
        "neutral_time_s": float((win["gear"].to_numpy() <= 0).sum() * dt),
        "completed": 1.0,
    }
    if "shaft_torque" in win.columns and len(pre):
        pre_torque = float(pre["shaft_torque"].mean())
        thr = 0.2 * pre_torque
        kpis["pre_shaft_torque_nm"] = pre_torque
        kpis["torque_interrupt_s"] = float(
            (win["shaft_torque"].to_numpy() < thr).sum() * dt
        ) if pre_torque > 0 else np.nan
    return kpis


def analyze_log(
    source: str | Path | pd.DataFrame,
    signal_map: SignalMap | None = None,
    vehicle: VehicleParams | None = None,
    resample_hz: float | None = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ログを読み込み、(整形済み時系列, イベント別 KPI 表) を返す。"""

    df = pd.read_csv(source) if not isinstance(source, pd.DataFrame) else source
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
