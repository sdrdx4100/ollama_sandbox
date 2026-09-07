"""車両・パワートレインの物理パラメータとマップ。

単位系は SI 系で統一する(トルク[Nm], 角速度[rad/s], 速度[m/s])。
表示・ログ用に rpm / km/h への変換ヘルパを用意する。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RPM_PER_RADS = 60.0 / (2.0 * np.pi)
KMH_PER_MS = 3.6
GRAVITY = 9.80665


def rads_to_rpm(w: float | np.ndarray) -> float | np.ndarray:
    return w * RPM_PER_RADS


def rpm_to_rads(rpm: float | np.ndarray) -> float | np.ndarray:
    return rpm / RPM_PER_RADS


@dataclass
class EngineParams:
    """簡易エンジンモデル(定常トルクマップ + 一次遅れのトルク応答)。"""

    # 全開トルク特性 (rpm -> Nm)
    wot_rpm: tuple[float, ...] = (800, 1500, 2500, 3500, 4500, 5500, 6500)
    wot_torque: tuple[float, ...] = (95, 150, 185, 190, 180, 160, 130)
    # 引きずり(エンジンブレーキ)トルク: -(a + b * rpm)
    drag_offset: float = 10.0
    drag_slope: float = 0.008
    idle_rpm: float = 750.0
    max_rpm: float = 6800.0
    inertia: float = 0.18  # J_e [kg m^2] クランク + クラッチドライブ側
    torque_time_constant: float = 0.06  # トルク応答一次遅れ [s]

    def wot(self, w: float) -> float:
        """指定角速度における全開トルク [Nm]。"""
        rpm = float(np.clip(rads_to_rpm(w), self.wot_rpm[0], self.wot_rpm[-1]))
        return float(np.interp(rpm, self.wot_rpm, self.wot_torque))

    def drag(self, w: float) -> float:
        """スロットル全閉時のトルク [Nm](負値)。"""
        rpm = max(float(rads_to_rpm(w)), 0.0)
        return -(self.drag_offset + self.drag_slope * rpm)

    def torque_limits(self, w: float) -> tuple[float, float]:
        """指定角速度で実現可能なトルク範囲 (下限, 上限)。"""
        return self.drag(w), self.wot(w)

    def steady_torque(self, throttle: float, w: float) -> float:
        """スロットル開度から定常トルクを補間する。"""
        lo, hi = self.torque_limits(w)
        th = float(np.clip(throttle, 0.0, 1.0))
        return lo + (hi - lo) * th


@dataclass
class TransmissionParams:
    """AMT(自動MT: 有段ギヤ + 乾式単板クラッチ)のパラメータ。"""

    gear_ratios: tuple[float, ...] = (3.55, 2.05, 1.35, 1.00, 0.80, 0.65)
    final_drive: float = 4.10
    efficiency: float = 0.94
    input_inertia: float = 0.03  # 入力軸 [kg m^2]
    # クラッチ
    clutch_capacity: float = 320.0  # 完全締結時の伝達容量 [Nm]
    clutch_kiss_point: float = 0.30  # ストローク比 これ以下は容量ゼロ
    clutch_curve_exponent: float = 1.6  # 容量の非線形性
    lock_slip_tol: float = 1.5  # ロック判定のすべり閾値 [rad/s]
    # アクチュエータ実時間
    gear_out_time: float = 0.05  # ギヤ抜き [s]
    gear_in_time: float = 0.07  # ギヤ入れ(シンクロ)[s]

    def n_gears(self) -> int:
        return len(self.gear_ratios)

    def has_gear(self, gear: int) -> bool:
        """その段が存在するか(ログのギヤ位置が段数を超える場合がある)。"""
        return isinstance(gear, (int, np.integer)) and 1 <= int(gear) <= self.n_gears()

    def total_ratio(self, gear: int) -> float:
        """gear は 1 始まり。総減速比 i_g * i_f。"""
        if not self.has_gear(gear):
            raise IndexError(
                f"{gear} 速はこの変速機({self.n_gears()} 段)にありません。"
                "車両プリセットを合わせてください(例: truck_12speed())"
            )
        return self.gear_ratios[gear - 1] * self.final_drive

    def clutch_capacity_at(self, position: float) -> float:
        """クラッチストローク比(0=解放, 1=締結)に対する伝達容量 [Nm]。"""
        x = float(np.clip(position, 0.0, 1.0))
        if x <= self.clutch_kiss_point:
            return 0.0
        u = (x - self.clutch_kiss_point) / (1.0 - self.clutch_kiss_point)
        return self.clutch_capacity * float(u ** self.clutch_curve_exponent)


@dataclass
class DrivelineParams:
    """駆動系ねじり(ドライブシャフト)2慣性モデル。"""

    stiffness: float = 4000.0  # k [Nm/rad]
    damping: float = 30.0  # c [Nm s/rad]
    wheel_inertia: float = 1.40  # 4輪 + ブレーキ [kg m^2]


@dataclass
class VehicleParams:
    """車体パラメータ + 走行抵抗。"""

    mass: float = 1500.0  # 車両重量 [kg]
    payload: float = 0.0  # 積載 [kg]
    wheel_radius: float = 0.31  # 動半径 [m]
    drag_area: float = 0.68  # Cd*A [m^2]
    air_density: float = 1.20
    rolling_resistance: float = 0.012
    grade: float = 0.0  # 勾配 [rad]

    engine: EngineParams = field(default_factory=EngineParams)
    transmission: TransmissionParams = field(default_factory=TransmissionParams)
    driveline: DrivelineParams = field(default_factory=DrivelineParams)

    @property
    def total_mass(self) -> float:
        return self.mass + self.payload

    def road_load(self, v: float) -> float:
        """走行抵抗 [N](空気 + 転がり + 勾配)。"""
        aero = 0.5 * self.air_density * self.drag_area * v * abs(v)
        roll = self.total_mass * GRAVITY * self.rolling_resistance * np.tanh(v / 0.5)
        grade = self.total_mass * GRAVITY * np.sin(self.grade)
        return float(aero + roll + grade)

    def wheel_speed_for(self, v: float) -> float:
        """車速 [m/s] から車輪角速度 [rad/s]。"""
        return v / self.wheel_radius

    def engine_speed_for(self, v: float, gear: int) -> float:
        """ロック締結時のエンジン角速度 [rad/s]。"""
        return self.wheel_speed_for(v) * self.transmission.total_ratio(gear)

    def wheel_side_inertia(self, gear: int | None) -> float:
        """車輪側に換算した回転慣性 [kg m^2]。gear=None はニュートラル。"""
        j = self.driveline.wheel_inertia
        if gear is not None:
            j += self.transmission.input_inertia * self.transmission.total_ratio(gear) ** 2
        return j


# ----------------------------------------------------------------------
# 車両プリセット
# ----------------------------------------------------------------------
def passenger_6speed() -> VehicleParams:
    """乗用車 + 6 速 AMT(既定値)。"""
    return VehicleParams()


def truck_12speed() -> VehicleParams:
    """大型トラック(トラクタ + セミトレーラ)+ 12 速 AMT。

    欧州系 12 速 AMT(オーバードライブ無し、トップ 1.00)を想定した代表値。
    実車に当てる場合は、少なくとも次を実測で同定すること。

      * 変速機の各段比と終減速比(SPN 526 のログから確認できる)
      * 車両総重量(積車 / 空車で大きく変わる)
      * 駆動系のねじり剛性・減衰(シャッフル周波数を合わせる)
      * クラッチ伝達容量とアクチュエータ応答
    """

    engine = EngineParams(
        # 13 L ディーゼル: 低回転で高トルク、常用域が狭い
        wot_rpm=(600, 900, 1200, 1400, 1600, 1800, 2100),
        wot_torque=(1200, 2300, 2300, 2300, 2050, 1800, 1200),
        drag_offset=60.0,
        drag_slope=0.05,
        idle_rpm=600.0,
        max_rpm=2100.0,
        inertia=2.6,  # フライホイールが大きい
        torque_time_constant=0.10,
    )
    transmission = TransmissionParams(
        gear_ratios=(14.94, 11.73, 9.04, 7.09, 5.54, 4.35,
                     3.44, 2.70, 2.08, 1.63, 1.27, 1.00),
        final_drive=3.40,
        efficiency=0.96,
        input_inertia=0.50,
        clutch_capacity=3000.0,
        clutch_kiss_point=0.35,
        clutch_curve_exponent=1.6,
        lock_slip_tol=2.0,
        gear_out_time=0.10,
        gear_in_time=0.15,
    )
    driveline = DrivelineParams(stiffness=60000.0, damping=1200.0, wheel_inertia=40.0)
    return VehicleParams(
        mass=16000.0,  # 空車(トラクタ + 空トレーラ)
        payload=0.0,
        wheel_radius=0.52,  # 315/80R22.5 相当
        drag_area=6.0,  # Cd*A(トラクタ + セミトレーラ)
        rolling_resistance=0.006,
        engine=engine,
        transmission=transmission,
        driveline=driveline,
    )


#: 名前で引ける車両プリセット
VEHICLE_PRESETS = {
    "passenger6": passenger_6speed,
    "truck12": truck_12speed,
}


def get_vehicle(name: str | VehicleParams | None) -> VehicleParams:
    """プリセット名から車両パラメータを取得する。"""
    if name is None:
        return VehicleParams()
    if isinstance(name, VehicleParams):
        return name
    if name not in VEHICLE_PRESETS:
        raise KeyError(
            f"未知の車両プリセット: {name} (選択肢: {sorted(VEHICLE_PRESETS)})"
        )
    return VEHICLE_PRESETS[name]()
