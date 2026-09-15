from __future__ import annotations

import numpy as np


def rotation_x(phi: float) -> np.ndarray:
    """Rotation about the x-axis by roll angle phi [rad]."""
    c = np.cos(phi)
    s = np.sin(phi)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )


def rotation_y(theta: float) -> np.ndarray:
    """Rotation about the y-axis by pitch angle theta [rad]."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def rotation_z(psi: float) -> np.ndarray:
    """Rotation about the z-axis by yaw angle psi [rad]."""
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def body_to_inertial(phi: float, theta: float, psi: float) -> np.ndarray:
    """Return R_B^I for a Z-Y-X yaw-pitch-roll convention.

    The matrix maps a vector expressed in the body frame into the inertial
    frame:
        v_I = R_BI @ v_B
    """
    return rotation_z(psi) @ rotation_y(theta) @ rotation_x(phi)


def euler_rate_matrix(phi: float, theta: float) -> np.ndarray:
    """Map body angular rates [p, q, r] to Euler-angle rates.

    This Z-Y-X Euler-angle representation is singular at pitch = +/- 90 deg.
    The simulation intentionally uses Euler angles because the project goal is
    to learn the classical 12-state model; production flight software often
    uses quaternions to avoid this singularity.
    """
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    c_theta = np.cos(theta)

    if abs(c_theta) < 1e-8:
        raise ValueError(
            "Euler-angle kinematics are singular near pitch = +/- 90 degrees."
        )

    t_theta = np.tan(theta)

    return np.array(
        [
            [1.0, s_phi * t_theta, c_phi * t_theta],
            [0.0, c_phi, -s_phi],
            [0.0, s_phi / c_theta, c_phi / c_theta],
        ],
        dtype=float,
    )

