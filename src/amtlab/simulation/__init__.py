"""AMT 車両のプラント/制御シミュレーション。"""

from .controller import (
    CONTROL_BOUNDS,
    ShiftController,
    ShiftControlParams,
    ShiftPhase,
    scaled_bounds,
)
from .plant import ShiftResult, ShiftScenario, SimSettings, simulate_shift
from .vehicle import (
    VEHICLE_PRESETS,
    DrivelineParams,
    EngineParams,
    TransmissionParams,
    VehicleParams,
    get_vehicle,
    passenger_6speed,
    rads_to_rpm,
    rpm_to_rads,
    truck_12speed,
)

__all__ = [
    "CONTROL_BOUNDS",
    "VEHICLE_PRESETS",
    "get_vehicle",
    "passenger_6speed",
    "scaled_bounds",
    "truck_12speed",
    "DrivelineParams",
    "EngineParams",
    "ShiftControlParams",
    "ShiftController",
    "ShiftPhase",
    "ShiftResult",
    "ShiftScenario",
    "SimSettings",
    "TransmissionParams",
    "VehicleParams",
    "rads_to_rpm",
    "rpm_to_rads",
    "simulate_shift",
]
