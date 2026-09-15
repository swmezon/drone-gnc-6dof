from __future__ import annotations

import numpy as np

from src.parameters import nominal_parameters
from src.simulation import simulate


def main() -> None:
    params = nominal_parameters()
    x0 = np.zeros(12, dtype=float)
    hover_thrust = params.mass * params.gravity

    def control(t: float, state: np.ndarray) -> np.ndarray:
        del state
        # A short thrust pulse creates nontrivial dynamics for comparison.
        thrust = 1.10 * hover_thrust if t < 1.0 else hover_thrust
        return np.array([thrust, 0.0, 0.0, 0.0], dtype=float)

    dt = 0.02
    euler = simulate(x0, control, params, t_final=3.0, dt=dt, method="euler")
    rk4 = simulate(x0, control, params, t_final=3.0, dt=dt, method="rk4")

    final_difference = np.linalg.norm(euler.state[-1] - rk4.state[-1])
    print("=== Euler vs. RK4 comparison ===")
    print(f"dt                      : {dt:.3f} s")
    print(f"Final-state L2 difference: {final_difference:.6e}")
    print(f"Euler final z           : {euler.state[-1, 2]:.6f} m")
    print(f"RK4 final z             : {rk4.state[-1, 2]:.6f} m")


if __name__ == "__main__":
    main()

