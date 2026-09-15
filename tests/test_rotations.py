import numpy as np

from src.rotations import body_to_inertial


def test_zero_euler_angles_give_identity_rotation() -> None:
    R = body_to_inertial(0.0, 0.0, 0.0)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12, rtol=0.0)


def test_rotation_matrix_is_orthonormal_and_proper() -> None:
    R = body_to_inertial(phi=0.3, theta=-0.2, psi=0.5)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12, rtol=0.0)

