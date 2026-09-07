"""実験計画(DoE)とバッチシミュレーションによる解析用データセット生成。

実車の適合データ(台上・実走ログ)に相当するものを、プラントモデルから
生成する。1 行 = 1 変速イベント = 「運転条件 + 適合値 + 変速品質 KPI」。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .features import result_to_row
from .simulation.controller import ShiftControlParams
from .simulation.plant import ShiftScenario, SimSettings, simulate_shift
from .simulation.vehicle import KMH_PER_MS, VehicleParams, rpm_to_rads


@dataclass
class ScenarioSpace:
    """運転条件のサンプリング空間。"""

    upshift_ratio: float = 0.65  # アップシフトの割合
    upshift_rpm: tuple[float, float] = (1800.0, 5200.0)
    downshift_rpm: tuple[float, float] = (1200.0, 3400.0)
    throttle: tuple[float, float] = (0.10, 1.00)
    grade_pct: tuple[float, float] = (-6.0, 8.0)
    payload_kg: tuple[float, float] = (0.0, 400.0)
    max_engine_rpm: float = 6200.0
    min_engine_rpm: float = 900.0
    vehicle: VehicleParams = field(default_factory=VehicleParams)

    def sample(self, rng: np.random.Generator) -> ShiftScenario:
        trm = self.vehicle.transmission
        n = trm.n_gears()
        for _ in range(64):
            upshift = rng.random() < self.upshift_ratio
            if upshift:
                g_from = int(rng.integers(1, n))  # 1..n-1
                g_to = g_from + 1
                rpm = float(rng.uniform(*self.upshift_rpm))
            else:
                g_from = int(rng.integers(2, n + 1))  # 2..n
                g_to = g_from - 1
                rpm = float(rng.uniform(*self.downshift_rpm))
            v = (
                rpm_to_rads(rpm)
                * self.vehicle.wheel_radius
                / trm.total_ratio(g_from)
            )
            rpm_to = v * trm.total_ratio(g_to) / self.vehicle.wheel_radius * 9.5493
            if not (self.min_engine_rpm <= rpm_to <= self.max_engine_rpm):
                continue
            return ShiftScenario(
                from_gear=g_from,
                to_gear=g_to,
                speed_kmh=float(v * KMH_PER_MS),
                throttle=float(rng.uniform(*self.throttle)),
                grade_pct=float(rng.uniform(*self.grade_pct)),
                payload_kg=float(rng.uniform(*self.payload_kg)),
            )
        raise RuntimeError("有効な運転条件をサンプリングできませんでした")


def _run_one(control: ShiftControlParams, scenario: ShiftScenario,
             vehicle: VehicleParams, settings: SimSettings) -> dict[str, float]:
    result = simulate_shift(control, scenario, vehicle=vehicle, settings=settings)
    return result_to_row(result)


def run_batch(
    controls: list[ShiftControlParams],
    scenarios: list[ShiftScenario],
    vehicle: VehicleParams | None = None,
    settings: SimSettings | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """(適合値, 条件) の組をまとめてシミュレーションし、DataFrame を返す。"""

    if len(controls) != len(scenarios):
        raise ValueError("controls と scenarios の長さが一致していません")
    vehicle = vehicle or VehicleParams()
    settings = settings or SimSettings()
    rows = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_run_one)(c, s, vehicle, settings) for c, s in zip(controls, scenarios)
    )
    return pd.DataFrame(rows)


def build_dataset(
    n_samples: int = 400,
    seed: int = 0,
    space: ScenarioSpace | None = None,
    vehicle: VehicleParams | None = None,
    settings: SimSettings | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """ランダム DoE で解析用データセットを生成する。

    適合値・運転条件ともに一様乱数でサンプリングするため、感度解析と
    代理モデル学習の双方に使える。
    """

    rng = np.random.default_rng(seed)
    space = space or ScenarioSpace(vehicle=vehicle or VehicleParams())
    scenarios = [space.sample(rng) for _ in range(n_samples)]
    controls = [ShiftControlParams.sample(rng) for _ in range(n_samples)]
    df = run_batch(controls, scenarios, vehicle=vehicle, settings=settings, n_jobs=n_jobs)
    df.insert(0, "event_id", np.arange(len(df)))
    return df


#: 説明変数(運転条件)
CONDITION_FEATURES = [
    "from_gear",
    "to_gear",
    "is_upshift",
    "speed_kmh",
    "throttle",
    "grade_pct",
    "payload_kg",
]

#: 説明変数(適合値)
CONTROL_FEATURES = list(ShiftControlParams().as_dict().keys())

FEATURE_COLUMNS = CONDITION_FEATURES + CONTROL_FEATURES
