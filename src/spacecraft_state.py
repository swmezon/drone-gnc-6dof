from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.quaternion_kinematics import normalize_quaternion


SPACECRAFT_STATE_SIZE = 13


@dataclass
class SpacecraftState:
    position_I: np.ndarray
    velocity_I: np.ndarray
    q_BI: np.ndarray
    omega_B: np.ndarray

    def __post_init__(self) -> None:
        self.position_I = np.asarray(self.position_I, dtype=float)
        self.velocity_I = np.asarray(self.velocity_I, dtype=float)
        self.q_BI = np.asarray(self.q_BI, dtype=float)
        self.omega_B = np.asarray(self.omega_B, dtype=float)

        if self.position_I.shape != (3,):
            raise ValueError("position_I must have shape (3,).")
        if self.velocity_I.shape != (3,):
            raise ValueError("velocity_I must have shape (3,).")
        if self.q_BI.shape != (4,):
            raise ValueError("q_BI must have shape (4,).")
        if self.omega_B.shape != (3,):
            raise ValueError("omega_B must have shape (3,).")

        self.q_BI = normalize_quaternion(self.q_BI)

    def to_vector(self) -> np.ndarray:
        """Convert the named state fields into the 13-number vector used by RK4."""
        return np.concatenate(
            [self.position_I, self.velocity_I, self.q_BI, self.omega_B]
        )

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "SpacecraftState":
        """Convert a 13-number numerical vector back into named state fields."""
        vector = np.asarray(vector, dtype=float)

        if vector.shape != (SPACECRAFT_STATE_SIZE,):
            raise ValueError(
                f"Spacecraft state must have shape ({SPACECRAFT_STATE_SIZE},), "
                f"got {vector.shape}."
            )

        return cls(
            position_I=vector[0:3],
            velocity_I=vector[3:6],
            q_BI=vector[6:10],
            omega_B=vector[10:13],
        )

