from __future__ import annotations

import numpy as np


QUATERNION_SIZE = 4


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Return a unit quaternion using scalar-first ordering [w, x, y, z]."""
    q = np.asarray(q, dtype=float)

    if q.shape != (QUATERNION_SIZE,):
        raise ValueError(
            f"Quaternion must have shape ({QUATERNION_SIZE},), got {q.shape}."
        )

    norm_q = np.linalg.norm(q)

    if norm_q < 1e-12:
        raise ValueError("Quaternion norm is too small to normalize.")

    return q / norm_q


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 tensor q2 using scalar-first quaternions."""
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)

    if q1.shape != (4,) or q2.shape != (4,):
        raise ValueError("Both quaternions must have shape (4,).")

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quaternion_to_rotation_matrix(q_BI: np.ndarray) -> np.ndarray:
    """Return R_BI, which maps body-frame vectors into inertial coordinates."""
    w, x, y, z = normalize_quaternion(q_BI)

    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=float,
    )


def quaternion_derivative(q_BI: np.ndarray, omega_B: np.ndarray) -> np.ndarray:
    """Return q_dot for body angular velocity expressed in the body frame."""
    q_BI = normalize_quaternion(q_BI)
    omega_B = np.asarray(omega_B, dtype=float)

    if omega_B.shape != (3,):
        raise ValueError(f"omega_B must have shape (3,), got {omega_B.shape}.")

    omega_quaternion = np.array(
        [0.0, omega_B[0], omega_B[1], omega_B[2]],
        dtype=float,
    )

    return 0.5 * quaternion_multiply(q_BI, omega_quaternion)

