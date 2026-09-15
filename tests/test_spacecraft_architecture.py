import numpy as np

from src.integrators import rk4_step
from src.mission_manager import MissionManager, MissionPhase, MissionStatus
from src.quaternion_kinematics import quaternion_to_rotation_matrix
from src.spacecraft_dynamics import SpacecraftParams, spacecraft_dynamics
from src.spacecraft_state import SpacecraftState


def test_identity_quaternion_gives_identity_rotation() -> None:
    q_identity = np.array([1.0, 0.0, 0.0, 0.0])
    R_BI = quaternion_to_rotation_matrix(q_identity)
    np.testing.assert_allclose(R_BI, np.eye(3), atol=1e-12, rtol=0.0)


def test_spacecraft_state_vector_round_trip() -> None:
    state = SpacecraftState(
        position_I=np.array([1.0, 2.0, 3.0]),
        velocity_I=np.array([4.0, 5.0, 6.0]),
        q_BI=np.array([1.0, 0.0, 0.0, 0.0]),
        omega_B=np.array([0.1, 0.2, 0.3]),
    )
    reconstructed = SpacecraftState.from_vector(state.to_vector())
    np.testing.assert_allclose(reconstructed.to_vector(), state.to_vector())


def test_free_space_constant_velocity_propagation() -> None:
    params = SpacecraftParams(
        mass_kg=500.0,
        inertia_B_kgm2=np.diag([120.0, 100.0, 80.0]),
    )
    state = SpacecraftState(
        position_I=np.zeros(3),
        velocity_I=np.array([1.0, 0.0, 0.0]),
        q_BI=np.array([1.0, 0.0, 0.0, 0.0]),
        omega_B=np.zeros(3),
    ).to_vector()
    zero_wrench = np.zeros(6)

    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        return spacecraft_dynamics(t, x, zero_wrench, params)

    final_state = rk4_step(rhs, 0.0, state, 1.0)
    np.testing.assert_allclose(
        final_state[0:3], np.array([1.0, 0.0, 0.0]), atol=1e-12, rtol=0.0
    )


def test_nominal_mission_phase_sequence() -> None:
    manager = MissionManager()
    assert manager.start() == MissionPhase.RENDEZVOUS

    phase = manager.update(MissionStatus(range_m=5.0, relative_speed_mps=0.05))
    assert phase == MissionPhase.HOVER

    phase = manager.update(
        MissionStatus(
            range_m=5.0,
            relative_speed_mps=0.0,
            hover_stable=True,
            tag_authorized=True,
        )
    )
    assert phase == MissionPhase.TAG_DESCENT

    phase = manager.update(
        MissionStatus(
            range_m=0.0,
            relative_speed_mps=0.0,
            contact_detected=True,
        )
    )
    assert phase == MissionPhase.TAG_CONTACT

    phase = manager.update(
        MissionStatus(range_m=0.0, relative_speed_mps=0.0, tag_complete=True)
    )
    assert phase == MissionPhase.DEPARTURE

    phase = manager.update(
        MissionStatus(
            range_m=20.0,
            relative_speed_mps=1.0,
            departure_complete=True,
        )
    )
    assert phase == MissionPhase.COMPLETE

