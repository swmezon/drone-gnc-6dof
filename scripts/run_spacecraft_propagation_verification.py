from __future__ import annotations

import numpy as np

from src.integrators import rk4_step
from src.gnc_interfaces import BodyWrenchCommand
from src.quaternion_kinematics import normalize_quaternion
from src.spacecraft_dynamics import SpacecraftParams, spacecraft_dynamics
from src.spacecraft_state import SpacecraftState


def main() -> None:
    params = SpacecraftParams(
        mass_kg=500.0,
        inertia_B_kgm2=np.diag([120.0, 100.0, 80.0]),
    )

    initial_state = SpacecraftState(
        position_I=np.array([0.0, 0.0, 0.0]),
        velocity_I=np.array([1.0, 0.0, 0.0]),
        q_BI=np.array([1.0, 0.0, 0.0, 0.0]),
        omega_B=np.zeros(3),
    )

    command = BodyWrenchCommand(
        force_B=np.zeros(3),
        torque_B=np.zeros(3),
    )

    state = initial_state.to_vector()
    dt = 0.1
    t_final = 10.0
    n_steps = int(round(t_final / dt))

    def rhs(t: float, state_vector: np.ndarray) -> np.ndarray:
        return spacecraft_dynamics(
            t,
            state_vector,
            command.to_vector(),
            params,
        )

    for k in range(n_steps):
        t = k * dt
        state = rk4_step(rhs, t, state, dt)
        state[6:10] = normalize_quaternion(state[6:10])

    final_state = SpacecraftState.from_vector(state)
    expected_position_I = np.array([10.0, 0.0, 0.0])

    position_pass = np.allclose(
        final_state.position_I, expected_position_I, atol=1e-10
    )
    velocity_pass = np.allclose(
        final_state.velocity_I, initial_state.velocity_I, atol=1e-10
    )
    quaternion_pass = np.isclose(
        np.linalg.norm(final_state.q_BI), 1.0, atol=1e-12
    )
    omega_pass = np.allclose(final_state.omega_B, np.zeros(3), atol=1e-12)

    print("\nSPACECRAFT PROPAGATION VERIFICATION")
    print("---------------------------------")
    print(f"Final position I [m]:    {final_state.position_I}")
    print(f"Final velocity I [m/s]:  {final_state.velocity_I}")
    print(f"Quaternion norm:         {np.linalg.norm(final_state.q_BI):.12f}")
    print(f"Body rate [rad/s]:       {final_state.omega_B}")

    print("\nChecks")
    print(f"[{'PASS' if position_pass else 'FAIL'}] Translation")
    print(f"[{'PASS' if velocity_pass else 'FAIL'}] Constant velocity")
    print(f"[{'PASS' if quaternion_pass else 'FAIL'}] Quaternion norm")
    print(f"[{'PASS' if omega_pass else 'FAIL'}] Angular velocity")

    all_pass = position_pass and velocity_pass and quaternion_pass and omega_pass
    if not all_pass:
        raise RuntimeError("Spacecraft propagation verification failed.")

    print("\nSPACECRAFT PROPAGATION VERIFICATION PASSED")


if __name__ == "__main__":
    main()


