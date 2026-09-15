from __future__ import annotations

from enum import Enum

import numpy as np


class Frame(str, Enum):
    INERTIAL = "I"
    BODY = "B"
    TARGET = "T"


def relative_translation(
    spacecraft_position_I: np.ndarray,
    spacecraft_velocity_I: np.ndarray,
    target_position_I: np.ndarray,
    target_velocity_I: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return spacecraft-minus-target relative position and relative velocity."""
    spacecraft_position_I = np.asarray(spacecraft_position_I, dtype=float)
    spacecraft_velocity_I = np.asarray(spacecraft_velocity_I, dtype=float)
    target_position_I = np.asarray(target_position_I, dtype=float)
    target_velocity_I = np.asarray(target_velocity_I, dtype=float)

    vectors = (
        spacecraft_position_I,
        spacecraft_velocity_I,
        target_position_I,
        target_velocity_I,
    )

    if any(vector.shape != (3,) for vector in vectors):
        raise ValueError("All translation vectors must have shape (3,).")

    relative_position_I = spacecraft_position_I - target_position_I
    relative_velocity_I = spacecraft_velocity_I - target_velocity_I

    return relative_position_I, relative_velocity_I


def inertial_vector_to_target(vector_I: np.ndarray, R_TI: np.ndarray) -> np.ndarray:
    """Express an inertial-frame vector in target-frame coordinates."""
    vector_I = np.asarray(vector_I, dtype=float)
    R_TI = np.asarray(R_TI, dtype=float)

    if vector_I.shape != (3,):
        raise ValueError("vector_I must have shape (3,).")
    if R_TI.shape != (3, 3):
        raise ValueError("R_TI must have shape (3, 3).")

    return R_TI.T @ vector_I

