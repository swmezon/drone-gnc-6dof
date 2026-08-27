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

    # Smooth open-loop climb profile over the first 3 seconds:
    #   0.0-1.5 s: thrust smoothly rises above hover and returns to hover,
    #                producing upward acceleration.
    #   1.5-3.0 s: thrust smoothly falls below hover and returns to hover,
    #                removing the acquired vertical velocity.
    #   3.0-6.0 s: hover thrust.
    #
    # The sinusoidal perturbation has zero net impulse over 3 seconds, so the
    # ideal no-drag model finishes the maneuver at a higher altitude with
    # approximately zero vertical velocity.
    maneuver_duration = 3.0
    thrust_fraction = 0.15

    def climb_control(t: float, state: np.ndarray) -> np.ndarray:
        del state
        if t <= maneuver_duration:
            phase = 2.0 * np.pi * t / maneuver_duration
            thrust = hover_thrust * (1.0 + thrust_fraction * np.sin(phase))
        else:
            thrust = hover_thrust
        return np.array([thrust, 0.0, 0.0, 0.0], dtype=float)

    result = simulate(
        x0,
        climb_control,
        params,
        t_final=6.0,
        dt=0.01,
        method="rk4",
    )

    z = result.state[:, 2]
    vz = result.state[:, 5]

    print("=== Open-loop climb simulation ===")
    print(f"Initial altitude   : {z[0]:.6f} m")
    print(f"Final altitude     : {z[-1]:.6f} m")
    print(f"Altitude gain      : {z[-1] - z[0]:.6f} m")
    print(f"Final vertical vel.: {vz[-1]:.6e} m/s")
    print(f"Peak vertical vel. : {np.max(vz):.6f} m/s")

    out = Path("results/figures")
    save_state_histories(result, out / "climb_states.png", "Open-Loop Climb State Histories")
    save_control_histories(result, out / "climb_controls.png", "Open-Loop Climb Inputs")
    save_trajectory_3d(result, out / "climb_trajectory_3d.png", "Open-Loop Climb 3-D Trajectory")


if __name__ == "__main__":
    main()
