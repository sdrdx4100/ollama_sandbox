"""変速イベントの時系列から特徴量(変速品質 KPI)を抽出する。

解析の主眼は「トルクホールがドライバに何をするか」なので、次の 2 つを中心に置く。

  * ``shift_time_s``     変速時間(トルクダウン開始 〜 トルク復帰完了)
  * ``speed_drop_kmh``   変速前後の車速の落ち込み幅(ピーク → 谷)

補助指標として、トルク抜け時間・ジャーク・クラッチすべり仕事・吹け上がり、
および「変速しなかった場合」との差である ``speed_loss_kmh`` を持つ。

なお ``from_gear`` は変速前の段 = **カレントギア(現在ギア)**、``to_gear`` は
変速先の段を指す。ギヤ別の傾向は :func:`gear_summary` で集計する。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .simulation.plant import ShiftResult
from .simulation.vehicle import RPM_PER_RADS

#: KPI の正規化スケール(代表的な良品レベル。小さいほど良い KPI のみ)
KPI_SCALES: dict[str, float] = {
    "shift_time_s": 1.0,
    "speed_drop_kmh": 3.0,
    "speed_loss_kmh": 3.0,
    "jerk_rms": 6.0,
    "jerk_peak": 40.0,
    "clutch_energy_j": 500.0,
    "torque_interrupt_s": 1.0,
    "engine_flare_rpm": 300.0,
}


@dataclass(frozen=True)
class ObjectiveSpec:
    """最適化で使う目的 KPI・正規化スケール・重み。

    既定は「変速時間」と「変速による車速損失」を主目的、変速ショック(ジャーク)を
    抑え役の第 3 目的とした構成。``configs/*.yaml`` から差し替えられる。

    Note
    ----
    車速の落ち込みには ``speed_drop_kmh``(実測のピーク→谷)と
    ``speed_loss_kmh``(変速しなかった場合との差)があるが、**目的関数には
    後者を使う**。前者は全開アップシフトのように「減速はしないが伸びが止まる」
    条件で恒等的に 0 になり、変速が遅くなっても改善したように見えてしまうため
    (``docs/methodology.md`` 参照)。落ち込み幅そのものは記述指標として
    全ての表・図に残している。
    """

    keys: tuple[str, ...] = ("shift_time_s", "speed_loss_kmh", "jerk_rms")
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "shift_time_s": 0.40,
            "speed_loss_kmh": 0.40,
            "jerk_rms": 0.20,
        }
    )
    scales: dict[str, float] = field(default_factory=lambda: dict(KPI_SCALES))

    def __post_init__(self) -> None:
        missing = [k for k in self.keys if k not in self.weights]
        if missing:
            raise KeyError(f"重みが未定義の目的 KPI: {missing}")
        unknown = [k for k in self.weights if k not in self.scales]
        if unknown:
            raise KeyError(f"正規化スケールが未定義の KPI: {unknown}")

    def scale(self, key: str) -> float:
        return float(self.scales.get(key, 1.0))

    @staticmethod
    def from_config(keys=None, weights=None) -> "ObjectiveSpec":
        """設定 (YAML) からの生成。``keys`` 省略時は重みのキー順を使う。"""
        if weights is None and keys is None:
            return ObjectiveSpec()
        weights = dict(weights) if weights else None
        if keys is None:
            keys = tuple(weights)
        keys = tuple(keys)
        if weights is None:
            base = ObjectiveSpec().weights
            weights = {k: base.get(k, 1.0 / len(keys)) for k in keys}
        return ObjectiveSpec(keys=keys, weights=weights)


#: 既定の目的仕様(後方互換のための別名も公開する)
DEFAULT_OBJECTIVES = ObjectiveSpec()
OBJECTIVE_KPIS = DEFAULT_OBJECTIVES.keys
OBJECTIVE_SCALES = DEFAULT_OBJECTIVES.scales
DEFAULT_WEIGHTS = DEFAULT_OBJECTIVES.weights


def _window(trace: pd.DataFrame, t0: float, t1: float) -> pd.DataFrame:
    return trace[(trace["time"] >= t0) & (trace["time"] <= t1)]


def speed_drop_metrics(
    time: np.ndarray, speed_kmh: np.ndarray, pre_accel: float
) -> dict[str, float]:
    """車速の落ち込みを 2 通りで測る。

    ``speed_drop_kmh``
        実測波形のピーク → 谷の落ち込み幅。ログからそのまま読める量。
    ``speed_loss_kmh``
        変速前の加速度を延長した「変速しなかった場合」の車速との最大差。
        惰行中のダウンシフトのように元々減速している場合でも、
        変速そのものが奪った車速だけを取り出せる。
    ``speed_recovery_s``
        谷から落ち込み前のピーク車速へ戻るまでの時間(窓内で戻らなければ NaN)。
    """

    if len(speed_kmh) < 2:
        return {"speed_drop_kmh": np.nan, "speed_drop_pct": np.nan,
                "speed_loss_kmh": np.nan, "speed_recovery_s": np.nan}

    i_min = int(np.argmin(speed_kmh))
    peak = float(np.max(speed_kmh[: i_min + 1])) if i_min > 0 else float(speed_kmh[0])
    trough = float(speed_kmh[i_min])
    drop = max(peak - trough, 0.0)

    # 変速しなかった場合の車速(変速前加速度を線形に延長)
    reference = speed_kmh[0] + pre_accel * 3.6 * (time - time[0])
    loss = float(np.max(reference - speed_kmh))

    recovered = np.flatnonzero(speed_kmh[i_min:] >= peak)
    recovery = float(time[i_min + recovered[0]] - time[i_min]) if len(recovered) else np.nan

    return {
        "speed_drop_kmh": drop,
        "speed_drop_pct": drop / peak * 100.0 if peak > 0 else np.nan,
        "speed_loss_kmh": max(loss, 0.0),
        "speed_recovery_s": recovery,
    }


def extract_kpis(result: ShiftResult, settle_time: float = 0.6) -> dict[str, float]:
    """1 変速イベントの KPI を辞書で返す。"""

    tr = result.trace
    t0 = result.shift_start_time
    t1 = result.shift_end_time
    dt = result.settings.dt
    win = _window(tr, t0, t1 + settle_time)
    pre = _window(tr, max(t0 - 0.15, tr["time"].iloc[0]), t0)

    pre_torque = float(pre["shaft_torque"].mean()) if len(pre) else 0.0
    pre_accel = float(pre["accel"].mean()) if len(pre) else 0.0
    pre_speed = float(pre["speed_kmh"].mean()) if len(pre) else float(tr["speed_kmh"].iloc[0])

    jerk = win["jerk"].to_numpy()
    accel = win["accel"].to_numpy()

    # トルク抜け: 変速前トルクの 20% を下回っている時間
    thr = 0.2 * pre_torque
    if pre_torque > 0:
        interrupt = float((win["shaft_torque"].to_numpy() < thr).sum() * dt)
    else:
        interrupt = float((win["shaft_torque"].to_numpy() > thr).sum() * dt)

    # クラッチすべり仕事
    slip = win["slip"].to_numpy()
    ct = win["clutch_torque"].to_numpy()
    power = np.abs(np.nan_to_num(ct * slip, nan=0.0))
    clutch_energy = float(np.trapezoid(power, dx=dt))
    slipping = np.abs(np.nan_to_num(slip, nan=0.0)) > result.lock_slip_tol
    slip_time = float((slipping & (np.abs(ct) > 1.0)).sum() * dt)

    # 吹け上がり: 変速中のエンジン回転が「開始回転 or 同期先回転」をどれだけ超えたか
    target_rpm = win["wheel_speed"].to_numpy() * result.ratio_to * RPM_PER_RADS
    start_rpm = float(pre["engine_speed_rpm"].mean()) if len(pre) else float(
        tr["engine_speed_rpm"].iloc[0]
    )
    reference = np.maximum(target_rpm, start_rpm)
    during_shift = win["phase"].isin(
        ["gear_out", "speed_sync", "gear_in", "clutch_close"]
    ).to_numpy()
    flare = win["engine_speed_rpm"].to_numpy() - reference
    engine_flare_rpm = float(np.max(flare[during_shift])) if during_shift.any() else 0.0

    speed = speed_drop_metrics(
        win["time"].to_numpy(), win["speed_kmh"].to_numpy(), pre_accel
    )

    phase_time = win.groupby("phase")["time"].agg(lambda s: (len(s) * dt))

    kpis = {
        "shift_time_s": float(t1 - t0),
        "current_gear": float(result.scenario.from_gear),
        "torque_interrupt_s": interrupt,
        "jerk_rms": float(np.sqrt(np.mean(jerk**2))) if len(jerk) else np.nan,
        "jerk_peak": float(np.max(np.abs(jerk))) if len(jerk) else np.nan,
        "accel_drop": float(pre_accel - np.min(accel)) if len(accel) else np.nan,
        "min_accel": float(np.min(accel)) if len(accel) else np.nan,
        **speed,
        "pre_speed_kmh": pre_speed,
        "clutch_energy_j": clutch_energy,
        "clutch_slip_time_s": slip_time,
        "engine_flare_rpm": max(engine_flare_rpm, 0.0),
        "sync_time_s": float(phase_time.get("speed_sync", 0.0)),
        "clutch_close_time_s": float(phase_time.get("clutch_close", 0.0)),
        "pre_shaft_torque_nm": pre_torque,
        "completed": float(result.completed),
    }
    kpis["quality_score"] = scalar_objective(kpis)
    return kpis


def scalar_objective(
    kpis: dict[str, float], objectives: ObjectiveSpec | None = None
) -> float:
    """目的 KPI を正規化して重み付き和にしたスカラー目的関数(小さいほど良い)。"""

    spec = objectives or DEFAULT_OBJECTIVES
    score = 0.0
    for key in spec.keys:
        value = kpis.get(key, np.nan)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return float("inf")
        score += spec.weights[key] * float(value) / spec.scale(key)
    if not kpis.get("completed", 1.0):
        score += 1.0  # 変速未完了はペナルティ
    return float(score)


def objective_vector(
    kpis: dict[str, float], objectives: ObjectiveSpec | None = None
) -> tuple[float, ...]:
    """多目的最適化用の目的値ベクトル。"""
    spec = objectives or DEFAULT_OBJECTIVES
    return tuple(float(kpis[k]) for k in spec.keys)


def result_to_row(result: ShiftResult) -> dict[str, float]:
    """1 実験 = 1 行(条件 + 適合値 + KPI)に変換する。"""
    row: dict[str, float] = {}
    row.update(result.scenario.as_dict())
    row.update(result.control.as_dict())
    row.update(extract_kpis(result))
    return row


def results_to_frame(results) -> pd.DataFrame:
    """複数の ShiftResult を解析用 DataFrame にまとめる。"""
    return pd.DataFrame([result_to_row(r) for r in results])


#: カレントギア(現在ギア)別に集計する KPI
GEAR_SUMMARY_KPIS = (
    "shift_time_s",
    "speed_drop_kmh",
    "speed_loss_kmh",
    "torque_interrupt_s",
    "jerk_rms",
)


def gear_summary(
    df: pd.DataFrame, kpis: tuple[str, ...] = GEAR_SUMMARY_KPIS
) -> pd.DataFrame:
    """カレントギア(``from_gear``)× 変速方向で KPI を集計する。

    「どの段の変速が一番効いているか」を一目で見るための表。
    """

    d = df.copy()
    d["current_gear"] = d["from_gear"].astype(int)
    d["direction"] = np.where(d["is_upshift"] == 1, "upshift", "downshift")
    cols = [k for k in kpis if k in d.columns]
    out = (
        d.groupby(["current_gear", "direction"], observed=True)[cols]
        .agg(["count", "mean", "std", "max"])
        .round(3)
    )
    out.columns = [f"{kpi}_{stat}" for kpi, stat in out.columns]
    return out.reset_index()


def gear_long_format(df: pd.DataFrame, kpis: tuple[str, ...] = GEAR_SUMMARY_KPIS) -> pd.DataFrame:
    """seaborn 用の長形式(カレントギア別 KPI)。"""
    d = df.copy()
    d["current_gear"] = d["from_gear"].astype(int)
    d["direction"] = np.where(d["is_upshift"] == 1, "upshift", "downshift")
    value_vars = [k for k in kpis if k in d.columns]
    return d.melt(
        id_vars=["current_gear", "direction", "speed_kmh", "throttle"],
        value_vars=value_vars,
        var_name="kpi",
        value_name="value",
    )
