from dataclasses import replace
import numpy as np
from src.aircraft import nominal_aircraft
from src.trim import trim_state_control
from src.dynamics import rk4_step, airdata, derivative
from src.autopilot import AutopilotState, guidance_command, update_autopilot
from src.sensors import SensorConfig, draw_biases, imu_measurement, gps_measurement
from src.ekf import NavigationEKF


def dispersed_aircraft(rng, sigma_fraction=0.05, clip_fraction=0.15):
    """Create one uncertain aircraft realization for Monte Carlo verification.

    Mass, inertias, and selected aerodynamic derivatives are perturbed by a
    zero-mean Gaussian scale factor. Draws are clipped so a rare random sample
    cannot create an obviously nonphysical aircraft.
    """
    p = nominal_aircraft()

    def scale(name):
        factor = float(np.clip(rng.normal(1.0, sigma_fraction), 1.0 - clip_fraction, 1.0 + clip_fraction))
        return getattr(p, name) * factor

    return replace(
        p,
        mass=scale("mass"),
        Jx=scale("Jx"),
        Jy=scale("Jy"),
        Jz=scale("Jz"),
        CL_alpha=scale("CL_alpha"),
        Cm_alpha=scale("Cm_alpha"),
        Cm_q=scale("Cm_q"),
        Cl_beta=scale("Cl_beta"),
        Cl_p=scale("Cl_p"),
        Cn_beta=scale("Cn_beta"),
        Cn_r=scale("Cn_r"),
    )


def run_mission(seed=1, t_final=70.0, dt=0.02, wind_ned=None, aircraft=None):
    rng = np.random.default_rng(seed)
    p = nominal_aircraft() if aircraft is None else aircraft
    if wind_ned is None:
        wind_ned = np.zeros(3)
    wind_ned = np.asarray(wind_ned, float)

    xtrim, utrim, _ = trim_state_control(p)
    x = xtrim.copy()
    x[:2] = np.array([0.0, 0.0])
    x[6:9] += rng.normal(0, np.deg2rad(0.5), 3)

    waypoints = np.array(
        [[0, 0, -100], [350, 0, -100], [350, 300, -120], [0, 300, -100], [0, 0, -100]],
        float,
    )
    wp_i = 1
    ap = AutopilotState(prev_surfaces=utrim[:3].copy())

    cfg = SensorConfig()
    bg, ba = draw_biases(rng, cfg)
    ekf0 = np.zeros(15)
    ekf0[0:3] = x[0:3] + np.array([8, -6, 4])
    ekf0[3:6] = np.array([25.0, 0.0, 0.0])
    ekf0[6:9] = x[6:9] + np.deg2rad([2, -1.5, 4])
    ekf = NavigationEKF(ekf0)

    n = int(t_final / dt) + 1
    T = np.arange(n) * dt
    X = np.zeros((n, 12))
    XE = np.zeros((n, 15))
    U = np.zeros((n, 4))
    GPS = np.full((n, 6), np.nan)

    for k, t in enumerate(T):
        X[k] = x
        XE[k] = ekf.x
        pos_est = ekf.x[:3]
        vel_est = ekf.x[3:6]

        if np.linalg.norm(waypoints[wp_i, :2] - pos_est[:2]) < 35 and wp_i < len(waypoints) - 1:
            wp_i += 1

        if np.linalg.norm(vel_est[:2]) > 0.5:
            chi = float(np.arctan2(vel_est[1], vel_est[0]))
        else:
            chi = x[8]

        phi_c, theta_c, _ = guidance_command(pos_est, waypoints[wp_i], chi)
        Va, _, _ = airdata(x, wind_ned)
        ucmd = update_autopilot(x, utrim, Va, 25.0, phi_c, theta_c, ap, dt, p)
        U[k] = ucmd

        dx = derivative(0.0, x, ucmd, p, wind_ned)
        gyro, accel = imu_measurement(x, dx, bg, ba, rng, cfg, p.g)
        ekf.predict(gyro, accel, dt)

        gps_stride = max(1, int(round(0.2 / dt)))
        if k % gps_stride == 0:  # GPS updates at approximately 5 Hz.
            z = gps_measurement(x, rng, cfg)
            GPS[k] = z
            ekf.update_gps(z)

        x = rk4_step(x, ucmd, dt, p, wind_ned)

    pos_err = np.linalg.norm(X[:, :3] - XE[:, :3], axis=1)
    att_err = np.linalg.norm(np.rad2deg(X[:, 6:9] - XE[:, 6:9]), axis=1)
    pos_rmse = float(np.sqrt(np.mean(pos_err**2)))
    att_rmse = float(np.sqrt(np.mean(att_err**2)))

    # Convergence requirement: remain below 1 m position error for 3 continuous seconds.
    conv = float("nan")
    hold = int(3 / dt)
    ok = pos_err < 1.0
    for i in range(len(ok) - hold):
        if np.all(ok[i : i + hold]):
            conv = float(T[i])
            break

    wp_final_dist = float(np.linalg.norm(X[-1, :2] - waypoints[-1, :2]))
    at_surface_limit = (
        (np.abs(U[:, 0]) >= p.aileron_limit_rad - 1e-4)
        | (np.abs(U[:, 1]) >= p.elevator_limit_rad - 1e-4)
        | (np.abs(U[:, 2]) >= p.rudder_limit_rad - 1e-4)
    )
    sat_fraction = float(np.mean(at_surface_limit))

    return dict(
        time=T,
        truth=X,
        estimate=XE,
        control=U,
        gps=GPS,
        pos_error=pos_err,
        att_error=att_err,
        pos_rmse=pos_rmse,
        att_rmse_deg=att_rmse,
        convergence_s=conv,
        final_waypoint_distance=wp_final_dist,
        saturation_fraction=sat_fraction,
        waypoints=waypoints,
        trim_state=xtrim,
        trim_control=utrim,
        wind_ned=wind_ned,
        aircraft=p,
    )
