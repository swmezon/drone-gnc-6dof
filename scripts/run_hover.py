from __future__ import annotations

from pathlib import Path
import numpy as np

from src.parameters import nominal_parameters
from src.plotting import save_control_histories, save_state_histories, save_trajectory_3d
from src.simulation import simulate


def main() -> None:
    params = nominal_parameters()
    x0 = np.zeros(12, dtype=float)
    hover_thrust = params.mass * params.gravity

    def hover_control(t: float, state: np.ndarray) -> np.ndarray:
        del t, state
        return np.array([hover_thrust, 0.0, 0.0, 0.0], dtype=float)

    result = simulate(
        x0,
        hover_control,
        params,
        t_final=5.0,
        dt=0.01,
        method="rk4",
    )

    position = result.state[:, 0:3]
    velocity = result.state[:, 3:6]
    attitude = result.state[:, 6:9]

    max_position_drift = np.max(np.linalg.norm(position - position[0], axis=1))
    max_speed = np.max(np.linalg.norm(velocity, axis=1))
    max_attitude_deg = np.rad2deg(np.max(np.linalg.norm(attitude, axis=1)))

    print("=== Hover equilibrium simulation ===")
    print(f"Hover thrust       : {hover_thrust:.6f} N")
    print(f"Max position drift : {max_position_drift:.3e} m")
    print(f"Max speed          : {max_speed:.3e} m/s")
    print(f"Max attitude norm  : {max_attitude_deg:.3e} deg")

    out = Path("results/figures")
    save_state_histories(result, out / "hover_states.png", "Hover State Histories")
    save_control_histories(result, out / "hover_controls.png", "Hover Control Inputs")
    save_trajectory_3d(result, out / "hover_trajectory_3d.png", "Hover 3-D Trajectory")


if __name__ == "__main__":
    main()

