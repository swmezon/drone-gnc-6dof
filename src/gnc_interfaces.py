from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.reference_frames import Frame
from src.spacecraft_state import SpacecraftState


@dataclass(frozen=True)
class SensorMeasurement:
    timestamp_s: float
    value: np.ndarray
    covariance: np.ndarray
    frame: Frame
    sensor_name: str
    valid: bool = True


@dataclass(frozen=True)
class BodyWrenchCommand:
    force_B: np.ndarray
    torque_B: np.ndarray

    def __post_init__(self) -> None:
        force_B = np.asarray(self.force_B, dtype=float)
        torque_B = np.asarray(self.torque_B, dtype=float)

        if force_B.shape != (3,):
            raise ValueError("force_B must have shape (3,).")
        if torque_B.shape != (3,):
            raise ValueError("torque_B must have shape (3,).")

        object.__setattr__(self, "force_B", force_B)
        object.__setattr__(self, "torque_B", torque_B)

    def to_vector(self) -> np.ndarray:
        return np.concatenate([self.force_B, self.torque_B])


class SensorInterface(Protocol):
    def measure(
        self,
        timestamp_s: float,
        truth_state: SpacecraftState,
    ) -> SensorMeasurement:
        ...


class ActuatorInterface(Protocol):
    def realize(self, command: BodyWrenchCommand) -> BodyWrenchCommand:
        ...

