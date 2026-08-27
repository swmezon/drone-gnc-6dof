from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import numpy as np

from src.dynamics import CONTROL_SIZE, STATE_SIZE, quadrotor_dynamics
from src.integrators import euler_step, rk4_step
from src.parameters import QuadrotorParams

ControlFunction = Callable[[float, np.ndarray], np.ndarray]


@dataclass
class SimulationResult:
    time: np.ndarray
    state: np.ndarray
    control: np.ndarray


def simulate(
    initial_state: np.ndarray,
    control_fn: ControlFunction,
    params: QuadrotorParams,
    *,
    t_final: float,
    dt: float,
    method: str = "rk4",
) -> SimulationResult:
    """Simulate the nonlinear quadrotor model on a uniform time grid."""
    initial_state = np.asarray(initial_state, dtype=float)
    if initial_state.shape != (STATE_SIZE,):
        raise ValueError(
            f"initial_state must have shape ({STATE_SIZE},), got {initial_state.shape}"
        )
    if t_final <= 0.0:
        raise ValueError("t_final must be positive.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    n_float = t_final / dt
    n_steps = int(round(n_float))
    if not np.isclose(n_float, n_steps, rtol=0.0, atol=1e-12):
        raise ValueError("For this project, t_final must be an integer multiple of dt.")

    integrators = {
        "euler": euler_step,
        "rk4": rk4_step,
    }
    try:
        stepper = integrators[method.lower()]
    except KeyError as exc:
        raise ValueError("method must be either 'euler' or 'rk4'.") from exc

    time = np.linspace(0.0, t_final, n_steps + 1)
    state_history = np.zeros((n_steps + 1, STATE_SIZE), dtype=float)
    control_history = np.zeros((n_steps + 1, CONTROL_SIZE), dtype=float)
    state_history[0] = initial_state

    for k in range(n_steps):
        t_k = time[k]
        x_k = state_history[k]

        u_k = np.asarray(control_fn(t_k, x_k.copy()), dtype=float)
        if u_k.shape != (CONTROL_SIZE,):
            raise ValueError(
                f"control_fn must return shape ({CONTROL_SIZE},), got {u_k.shape}"
            )
        control_history[k] = u_k

        def rhs(t_eval: float, x_eval: np.ndarray) -> np.ndarray:
            u_eval = np.asarray(control_fn(t_eval, x_eval.copy()), dtype=float)
            return quadrotor_dynamics(t_eval, x_eval, u_eval, params)

        state_history[k + 1] = stepper(rhs, t_k, x_k, dt)

    control_history[-1] = np.asarray(
        control_fn(time[-1], state_history[-1].copy()), dtype=float
    )

    return SimulationResult(time=time, state=state_history, control=control_history)
