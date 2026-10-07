import numpy as np
from src.rotations import R_body_to_ned, euler_rate_matrix

# state = [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]
# control = [da, de, dr, throttle]


def airdata(x, wind_ned=None):
    """Return airspeed, angle of attack, and sideslip.

    The rigid-body state stores ground-relative body velocity. Aerodynamic force,
    however, depends on velocity relative to the surrounding air mass. Therefore
    an inertial/NED wind vector is rotated into the body frame and subtracted.
    """
    if wind_ned is None:
        wind_ned = np.zeros(3)
    phi, theta, psi = x[6:9]
    Rbn = R_body_to_ned(phi, theta, psi)
    wind_body = Rbn.T @ np.asarray(wind_ned, float)
    vel_air_b = np.asarray(x[3:6], float) - wind_body
    ua, va, wa = vel_air_b
    Va = max(1e-3, float(np.linalg.norm(vel_air_b)))
    alpha = float(np.arctan2(wa, ua))
    beta = float(np.arcsin(np.clip(va / Va, -0.999, 0.999)))
    return Va, alpha, beta


def forces_moments(x, u_ctrl, p, wind_ned=None):
    da, de, dr, throttle = np.asarray(u_ctrl, float)
    throttle = float(np.clip(throttle, 0.0, 1.0))
    Va, alpha, beta = airdata(x, wind_ned)
    pb, qb, rb = x[9:12]
    qbar = 0.5 * p.rho * Va * Va

    CL = p.CL0 + p.CL_alpha * alpha + p.CL_q * (p.c / (2 * Va)) * qb + p.CL_de * de
    CD = p.CD0 + p.CD_alpha2 * alpha * alpha
    Cm = p.Cm0 + p.Cm_alpha * alpha + p.Cm_q * (p.c / (2 * Va)) * qb + p.Cm_de * de

    CY = (
        p.CY_beta * beta
        + p.CY_p * (p.b / (2 * Va)) * pb
        + p.CY_r * (p.b / (2 * Va)) * rb
        + p.CY_da * da
        + p.CY_dr * dr
    )
    Cl = (
        p.Cl_beta * beta
        + p.Cl_p * (p.b / (2 * Va)) * pb
        + p.Cl_r * (p.b / (2 * Va)) * rb
        + p.Cl_da * da
        + p.Cl_dr * dr
    )
    Cn = (
        p.Cn_beta * beta
        + p.Cn_p * (p.b / (2 * Va)) * pb
        + p.Cn_r * (p.b / (2 * Va)) * rb
        + p.Cn_da * da
        + p.Cn_dr * dr
    )

    lift = qbar * p.S * CL
    drag = qbar * p.S * CD
    side = qbar * p.S * CY
    ca, sa = np.cos(alpha), np.sin(alpha)
    Xa = -drag * ca + lift * sa
    Za = -drag * sa - lift * ca
    thrust = p.thrust_max * throttle * throttle
    F_aero = np.array([Xa, side, Za])
    F_prop = np.array([thrust, 0.0, 0.0])

    phi, theta, psi = x[6:9]
    Rbn = R_body_to_ned(phi, theta, psi)
    Fg_n = np.array([0.0, 0.0, p.mass * p.g])
    Fg_b = Rbn.T @ Fg_n
    force_b = F_aero + F_prop + Fg_b

    Mx = qbar * p.S * p.b * Cl
    My = qbar * p.S * p.c * Cm
    Mz = qbar * p.S * p.b * Cn
    return force_b, np.array([Mx, My, Mz])


def derivative(t, x, u_ctrl, p, wind_ned=None):
    x = np.asarray(x, float)
    vel_b = x[3:6]
    phi, theta, psi = x[6:9]
    omega = x[9:12]
    force_b, moment_b = forces_moments(x, u_ctrl, p, wind_ned)

    pos_dot = R_body_to_ned(phi, theta, psi) @ vel_b
    vel_dot = force_b / p.mass - np.cross(omega, vel_b)
    euler_dot = euler_rate_matrix(phi, theta) @ omega

    inertia = np.array(
        [[p.Jx, 0.0, -p.Jxz], [0.0, p.Jy, 0.0], [-p.Jxz, 0.0, p.Jz]]
    )
    omega_dot = np.linalg.solve(
        inertia, moment_b - np.cross(omega, inertia @ omega)
    )
    return np.concatenate([pos_dot, vel_dot, euler_dot, omega_dot])


def rk4_step(x, u_ctrl, dt, p, wind_ned=None):
    f = lambda xx: derivative(0.0, xx, u_ctrl, p, wind_ned)
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return np.asarray(x) + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
