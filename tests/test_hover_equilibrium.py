import numpy as np

from src.dynamics import quadrotor_dynamics
from src.parameters import nominal_parameters


def test_hover_equilibrium_derivative_is_zero() -> None:
    params = nominal_parameters()
    state = np.zeros(12, dtype=float)
    control = np.array(
        [params.mass * params.gravity, 0.0, 0.0, 0.0], dtype=float
    )

    state_dot = quadrotor_dynamics(0.0, state, control, params)

    np.testing.assert_allclose(state_dot, np.zeros(12), atol=1e-12, rtol=0.0)

