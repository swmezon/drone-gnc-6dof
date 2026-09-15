from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.quaternion_kinematics import quaternion_derivative, quaternion_to_rotation_matrix
from src.spacecraft_state import SPACECRAFT_STATE_SIZE, SpacecraftState


BODY_WRENCH_SIZE = 6


@dataclass(frozen=True)
class SpacecraftParams:
    mass_kg: float
    inertia_B_kgm2: np.ndarray

    def __post_init__(self) -> None:
        inertia = np.asarray(self.inertia_B_kgm2, dtype=float)

        if self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be positive.")
        if inertia.shape != (3, 3):
            raise ValueError("inertia_B_kgm2 must have shape (3, 3).")
        if not np.allclose(inertia, inertia.T, atol=1e-12, rtol=0.0):
            raise ValueError("inertia_B_kgm2 must be symmetric.")
        if np.any(np.linalg.eigvalsh(inertia) <= 0.0):
            raise ValueError("inertia_B_kgm2 must be positive definite.")

        object.__setattr__(self, "inertia_B_kgm2", inertia)


def spacecraft_dynamics(
    t: float,
    state_vector: np.ndarray,
    body_wrench: np.ndarray,
    params: SpacecraftParams,
) -> np.ndarray:
    """Evaluate free-space 6-DOF rigid-body spacecraft dynamics."""
    del t

    state_vector = np.asarray(state_vector, dtype=float)
    body_wrench = np.asarray(body_wrench, dtype=float)

    if state_vector.shape != (SPACECRAFT_STATE_SIZE,):
        raise ValueError(
            f"state_vector must have shape ({SPACECRAFT_STATE_SIZE},)."
        )
    if body_wrench.shape != (BODY_WRENCH_SIZE,):
        raise ValueError(f"body_wrench must have shape ({BODY_WRENCH_SIZE},).")

    state = SpacecraftState.from_vector(state_vector)
    force_B = body_wrench[0:3]
    torque_B = body_wrench[3:6]

    R_BI = quaternion_to_rotation_matrix(state.q_BI)

    position_dot_I = state.velocity_I
    force_I = R_BI @ force_B
    velocity_dot_I = force_I / params.mass_kg

    q_dot_BI = quaternion_derivative(state.q_BI, state.omega_B)

    angular_momentum_B = params.inertia_B_kgm2 @ state.omega_B
    gyroscopic_term_B = np.cross(state.omega_B, angular_momentum_B)
    omega_dot_B = np.linalg.solve(
        params.inertia_B_kgm2,
        torque_B - gyroscopic_term_B,
    )

    return np.concatenate(
        [position_dot_I, velocity_dot_I, q_dot_BI, omega_dot_B]
    )

