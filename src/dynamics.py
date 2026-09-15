from __future__ import annotations

import numpy as np

from src.parameters import QuadrotorParams
from src.rotations import body_to_inertial, euler_rate_matrix

STATE_SIZE = 12
CONTROL_SIZE = 4


def quadrotor_dynamics(
    t: float,
    state: np.ndarray,
    control: np.ndarray,
    params: QuadrotorParams,
) -> np.ndarray:
    """Evaluate the nonlinear 6-DOF rigid-body state derivative.

    State ordering
    --------------
    [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]

    Coordinate/sign convention
    --------------------------
    * Inertial frame is right-handed with +z upward.
    * Gravity acts in inertial -z.
    * Positive collective thrust acts along body +z.
    * R_BI maps body-frame vectors into the inertial frame.

    Control ordering
    ----------------
    [T, tau_x, tau_y, tau_z]
      T      : collective thrust [N]
      tau_*  : body torques [N*m]
    """
    del t  # Autonomous dynamics; retained in the signature for integrators.

    state = np.asarray(state, dtype=float)
    control = np.asarray(control, dtype=float)

    if state.shape != (STATE_SIZE,):
        raise ValueError(f"state must have shape ({STATE_SIZE},), got {state.shape}")
    if control.shape != (CONTROL_SIZE,):
        raise ValueError(
            f"control must have shape ({CONTROL_SIZE},), got {control.shape}"
        )

    params.validate()

    # Unpack the state.
    position_I = state[0:3]
    velocity_I = state[3:6]
    phi, theta, psi = state[6:9]
    omega_B = state[9:12]
    del position_I  # Position does not explicitly appear in this simple model.

    # Unpack the rigid-body control input.
    thrust = control[0]
    tau_B = control[1:4]

    # Translational kinematics.
    position_dot_I = velocity_I

    # Translational dynamics.
    R_BI = body_to_inertial(phi, theta, psi)
    thrust_B = np.array([0.0, 0.0, thrust], dtype=float)
    thrust_I = R_BI @ thrust_B
    gravity_I = np.array([0.0, 0.0, -params.gravity], dtype=float)
    velocity_dot_I = thrust_I / params.mass + gravity_I

    # Rotational kinematics: body rates are not generally Euler-angle rates.
    E = euler_rate_matrix(phi, theta)
    euler_dot = E @ omega_B

    # Rotational dynamics: Euler rigid-body equation.
    inertia = params.inertia
    angular_momentum_B = inertia @ omega_B
    gyroscopic_term = np.cross(omega_B, angular_momentum_B)
    omega_dot_B = np.linalg.solve(inertia, tau_B - gyroscopic_term)

    state_dot = np.concatenate(
        [position_dot_I, velocity_dot_I, euler_dot, omega_dot_B]
    )
    return state_dot

