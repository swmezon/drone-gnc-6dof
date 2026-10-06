from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from src.attitude_controller import CascadedAttitudeController
from src.dynamics import STATE_SIZE, quadrotor_dynamics
from src.integrators import rk4_step
from src.parameters import QuadrotorParams

@dataclass
class ClosedLoopAttitudeResult:
    time: np.ndarray
    state: np.ndarray
    control: np.ndarray
    attitude_command: np.ndarray
    desired_body_rate: np.ndarray
    attitude_error: np.ndarray
    body_rate_error: np.ndarray

def simulate_closed_loop_attitude(
    initial_state,
    attitude_reference,
    controller: CascadedAttitudeController,
    params: QuadrotorParams,
    *,
    t_final: float,
    dt: float,
):
    initial_state = np.asarray(initial_state, dtype=float)
    if initial_state.shape != (STATE_SIZE,):
        raise ValueError(f"initial_state must have shape ({STATE_SIZE},)")

    n_steps = int(round(t_final/dt))
    time = np.linspace(0.0, t_final, n_steps+1)

    state_hist = np.zeros((n_steps+1, STATE_SIZE))
    control_hist = np.zeros((n_steps+1, 4))
    cmd_hist = np.zeros((n_steps+1, 3))
    desired_rate_hist = np.zeros((n_steps+1, 3))
    attitude_error_hist = np.zeros((n_steps+1, 3))
    rate_error_hist = np.zeros((n_steps+1, 3))

    state_hist[0] = initial_state
    hover_thrust = params.mass * params.gravity
    controller.reset()

    for k in range(n_steps):
        t = time[k]
        x = state_hist[k]
        cmd = np.asarray(attitude_reference(t), dtype=float)

        torque, desired_rate, att_err, rate_err = controller.update(
            cmd, x[6:9], x[9:12], dt
        )

        u = np.concatenate(([hover_thrust], torque))

        cmd_hist[k] = cmd
        desired_rate_hist[k] = desired_rate
        attitude_error_hist[k] = att_err
        rate_error_hist[k] = rate_err
        control_hist[k] = u

        def rhs(t_eval, x_eval):
            return quadrotor_dynamics(t_eval, x_eval, u, params)

        state_hist[k+1] = rk4_step(rhs, t, x, dt)

    cmd_hist[-1] = attitude_reference(time[-1])
    control_hist[-1] = control_hist[-2]
    desired_rate_hist[-1] = desired_rate_hist[-2]
    attitude_error_hist[-1] = cmd_hist[-1] - state_hist[-1, 6:9]
    rate_error_hist[-1] = desired_rate_hist[-1] - state_hist[-1, 9:12]

    return ClosedLoopAttitudeResult(
        time, state_hist, control_hist, cmd_hist,
        desired_rate_hist, attitude_error_hist, rate_error_hist
    )
