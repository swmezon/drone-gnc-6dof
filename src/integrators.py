from __future__ import annotations

from collections.abc import Callable
import numpy as np

RHS = Callable[[float, np.ndarray], np.ndarray]


def euler_step(rhs: RHS, t: float, state: np.ndarray, dt: float) -> np.ndarray:
    """Advance one explicit-Euler integration step."""
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    return state + dt * rhs(t, state)


def rk4_step(rhs: RHS, t: float, state: np.ndarray, dt: float) -> np.ndarray:
    """Advance one classical fourth-order Runge-Kutta integration step."""
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    k1 = rhs(t, state)
    k2 = rhs(t + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = rhs(t + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = rhs(t + dt, state + dt * k3)

    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
