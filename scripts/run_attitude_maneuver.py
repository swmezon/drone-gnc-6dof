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

    # This is an open-loop body-torque maneuver, not closed-loop attitude tracking.
    # A smooth positive/negative roll-torque cycle first builds and then removes
    # body roll rate. Its net torque impulse is zero, while the vehicle acquires a
    # finite roll-angle change during the maneuver.
    maneuver_start = 1.0
    maneuver_duration = 0.6
    target_roll_deg = 5.0
    target_roll_rad = np.deg2rad(target_roll_deg)
    tau_roll_amplitude = (
        target_roll_rad * 2.0 * np.pi * params.Ixx / maneuver_duration**2
    )

    def attitude_control(t: float, state: np.ndarray) -> np.ndarray:
        del state
        tau_x = 0.0
        if maneuver_start <= t <= maneuver_start + maneuver_duration:
            local_time = t - maneuver_start
            phase = 2.0 * np.pi * local_time / maneuver_duration
            tau_x = tau_roll_amplitude * np.sin(phase)
        return np.array([hover_thrust, tau_x, 0.0, 0.0], dtype=float)

    result = simulate(
        x0,
        attitude_control,
        params,
        t_final=4.0,
        dt=0.005,
        method="rk4",
    )

    roll_deg = np.rad2deg(result.state[:, 6])
    p_deg_s = np.rad2deg(result.state[:, 9])

    print("=== Open-loop roll maneuver ===")
    print(f"Target roll angle  : {target_roll_deg:.6f} deg")
    print(f"Peak torque        : {tau_roll_amplitude:.6f} N*m")
    print(f"Final roll angle   : {roll_deg[-1]:.6f} deg")
    print(f"Peak |roll rate|   : {np.max(np.abs(p_deg_s)):.6f} deg/s")
    print(f"Final roll rate    : {p_deg_s[-1]:.6e} deg/s")

    out = Path("results/figures")
    save_state_histories(
        result, out / "attitude_states.png", "Open-Loop Roll Maneuver State Histories"
    )
    save_control_histories(
        result, out / "attitude_controls.png", "Open-Loop Roll Maneuver Inputs"
    )
    save_trajectory_3d(
        result, out / "attitude_trajectory_3d.png", "Open-Loop Roll Maneuver 3-D Trajectory"
    )


if __name__ == "__main__":
    main()
