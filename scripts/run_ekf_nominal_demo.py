import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import (
    FuncAnimation,
    PillowWriter,
)


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from drone_gnc.vehicles import QuadrotorModel

from drone_gnc.control.cascaded import (
    CascadedController,
)

from drone_gnc.guidance.waypoints import (
    WaypointGuidance,
)

from drone_gnc.estimation.sensors import (
    ImuGpsSensorSuite,
    SensorConfig,
)

from drone_gnc.estimation.error_state_ekf import (
    ErrorStateEKF15,
)

from drone_gnc.simulation.mission import (
    run_waypoint_mission,
)

from drone_gnc.evaluation.metrics import (
    trajectory_metrics,
    estimation_metrics,
    attitude_tracking_metrics,
)


# ============================================================
# SETTINGS
# ============================================================

DT = 0.01

GPS_HZ = 10.0

DURATION = 40.0

SENSOR_SEED = 24

POSITION_NOISE_M = 0.02


# ============================================================
# WAYPOINTS
# ============================================================

WAYPOINTS = np.array(
    [
        [0.0, 0.0, 1.5],
        [2.0, 0.0, 1.5],
        [2.0, 2.0, 1.5],
        [0.0, 2.0, 1.5],
        [0.0, 0.0, 1.5],
    ],
    dtype=float,
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

for folder in [
    "results/animations",
    "results/data",
    "results/guidance",
    "results/estimation",
]:

    (
        ROOT / folder
    ).mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# LOAD OPTIMIZED FULL-GNC GAINS
# ============================================================

GAIN_FILE = (
    ROOT
    / "config"
    / "optimized_2cm_full_gnc.json"
)


if not GAIN_FILE.exists():

    raise FileNotFoundError(
        "Could not find "
        "config/optimized_2cm_full_gnc.json. "
        "Run optimize_2cm_full_gnc.py first."
    )


gain_data = json.loads(
    GAIN_FILE.read_text()
)


KP_ATT = np.asarray(
    gain_data["kp_att"],
    dtype=float,
)


KP_RATE = np.asarray(
    gain_data["kp_rate"],
    dtype=float,
)


KI_RATE = np.asarray(
    gain_data["ki_rate"],
    dtype=float,
)


KD_RATE = np.asarray(
    gain_data["kd_rate"],
    dtype=float,
)


KP_POS = np.asarray(
    gain_data["kp_pos"],
    dtype=float,
)


KD_POS = np.asarray(
    gain_data["kd_pos"],
    dtype=float,
)


KI_POS = np.asarray(
    gain_data["ki_pos"],
    dtype=float,
)


# ============================================================
# VEHICLE
# ============================================================

vehicle = QuadrotorModel()


# ============================================================
# CONTROLLER
# ============================================================

controller = CascadedController(
    vehicle.params.mass,
    vehicle.params.gravity,
)


controller.kp_att = KP_ATT.copy()

controller.kp_rate = KP_RATE.copy()

controller.ki_rate = KI_RATE.copy()

controller.kd_rate = KD_RATE.copy()

controller.kp_pos = KP_POS.copy()

controller.kd_pos = KD_POS.copy()

controller.ki_pos = KI_POS.copy()


# ============================================================
# GUIDANCE
# ============================================================

guidance = WaypointGuidance(
    WAYPOINTS,
    acceptance_radius=0.25,
    cruise_speed=0.60,
)


# ============================================================
# SENSOR MODEL
# ============================================================

sensor_cfg = SensorConfig(

    position_noise_std_x=
        POSITION_NOISE_M,

    position_noise_std_y=
        POSITION_NOISE_M,

    position_noise_std_z=
        POSITION_NOISE_M,
)


sensors = ImuGpsSensorSuite(
    config=sensor_cfg,
    seed=SENSOR_SEED,
)


# ============================================================
# EKF
# ============================================================

ekf = ErrorStateEKF15(

    dt=DT,

    gps_pos_std=
        sensor_cfg.gps_pos_std,

    gps_vel_std=
        sensor_cfg.gps_vel_std,

    accel_noise_std=
        sensor_cfg.accel_noise_std,

    gyro_noise_std=
        sensor_cfg.gyro_noise_std,

    accel_bias_rw_std=
        sensor_cfg.accel_bias_rw_std,

    gyro_bias_rw_std=
        sensor_cfg.gyro_bias_rw_std,
)


# ============================================================
# CONFIGURATION REPORT
# ============================================================

print()

print("=" * 84)

print(
    "OPTIMIZED 2 CM SENSOR FULL-GNC DEMO"
)

print("=" * 84)

print()

print("INNER LOOP")
print("-" * 84)

print(
    f"Attitude Kp = {controller.kp_att}"
)

print(
    f"Rate Kp     = {controller.kp_rate}"
)

print(
    f"Rate Ki     = {controller.ki_rate}"
)

print(
    f"Rate Kd     = {controller.kd_rate}"
)


print()

print("OUTER LOOP")
print("-" * 84)

print(
    f"Position Kp = {controller.kp_pos}"
)

print(
    f"Velocity Kd = {controller.kd_pos}"
)

print(
    f"Position Ki = {controller.ki_pos}"
)


print()

print("SENSOR")
print("-" * 84)

print(
    f"Position sigma = "
    f"{POSITION_NOISE_M:.3f} m"
)

print(
    f"Measurement rate = "
    f"{GPS_HZ:.1f} Hz"
)

print(
    f"IMU/control rate = "
    f"{1.0 / DT:.1f} Hz"
)

print("=" * 84)


# ============================================================
# RUN CLOSED-LOOP MISSION
# ============================================================

logs = run_waypoint_mission(

    vehicle,

    controller,

    guidance,

    sensor_suite=sensors,

    ekf=ekf,

    duration=DURATION,

    dt=DT,

    gps_hz=GPS_HZ,

    wind=None,
)


# ============================================================
# POINT-TO-SEGMENT DISTANCE
# ============================================================

def point_segment_distance(
    p,
    a,
    b,
):

    ab = b - a

    denominator = float(
        ab @ ab
    )


    if denominator < 1e-12:

        return float(
            np.linalg.norm(
                p - a
            )
        )


    q = np.clip(
        ((p - a) @ ab)
        / denominator,
        0.0,
        1.0,
    )


    projection = (
        a + q * ab
    )


    return float(
        np.linalg.norm(
            p - projection
        )
    )


# ============================================================
# GEOMETRIC PATH
# ============================================================

path_points = np.vstack(
    [
        logs["state"][0, 0:3],
        WAYPOINTS,
    ]
)


# ============================================================
# CROSS-TRACK ERROR
# ============================================================

cross_track_error = []


for position in logs[
    "state"
][:, 0:3]:

    error = min(

        point_segment_distance(
            position,
            path_points[j],
            path_points[j + 1],
        )

        for j in range(
            len(path_points) - 1
        )
    )


    cross_track_error.append(
        error
    )


cross_track_error = np.asarray(
    cross_track_error,
    dtype=float,
)


# ============================================================
# TRAJECTORY METRICS
# ============================================================

metrics = trajectory_metrics(

    logs["state"][:, 0:3],

    logs["ref"],

    waypoint_indices=
        logs["wp_index"],

    final_waypoint=
        WAYPOINTS[-1],

    waypoints_reached=
        int(
            logs[
                "waypoints_reached"
            ][-1]
        ),
)


# ============================================================
# ESTIMATION METRICS
# ============================================================

metrics.update(

    estimation_metrics(
        logs["state"],
        logs["estimate"],
    )
)


# ============================================================
# ATTITUDE METRICS
# ============================================================

metrics.update(

    attitude_tracking_metrics(
        logs["state"][:, 6:9],
        logs["att_ref"],
    )
)


metrics[
    "cross_track_rmse_m"
] = float(

    np.sqrt(
        np.mean(
            cross_track_error**2
        )
    )
)


metrics[
    "cross_track_max_m"
] = float(

    np.max(
        cross_track_error
    )
)


# ============================================================
# SATURATION
# ============================================================

thrust = logs[
    "control"
][:, 0]


torque = logs[
    "control"
][:, 1:4]


metrics[
    "thrust_saturation_pct"
] = float(

    100.0

    * np.mean(

        np.isclose(
            thrust,
            controller.limits.max_thrust,
            atol=1e-9,
        )
    )
)


metrics[
    "torque_saturation_pct"
] = float(

    100.0

    * np.mean(

        np.any(

            np.isclose(
                np.abs(torque),
                controller.limits.max_torque,
                atol=1e-9,
            ),

            axis=1,
        )
    )
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics["position_noise_std_m"] = (
    POSITION_NOISE_M
)


metrics_file = (

    ROOT
    / "results"
    / "data"
    / "optimized_2cm_full_gnc_metrics.json"
)


metrics_file.write_text(

    json.dumps(
        metrics,
        indent=2,
    )
)


# ============================================================
# ANIMATION ARRAYS
# ============================================================

stride = 10


time_anim = logs[
    "t"
][::stride]


true_anim = logs[
    "state"
][::stride, 0:3]


estimate_anim = logs[
    "estimate"
][::stride, 0:3]


desired_anim = logs[
    "ref"
][::stride, 0:3]


cross_anim = (
    cross_track_error[
        ::stride
    ]
)


# ============================================================
# 2D PLOT LIMITS
# ============================================================

all_x = np.concatenate(
    [
        logs["state"][:, 0],
        logs["estimate"][:, 0],
        logs["ref"][:, 0],
    ]
)


all_y = np.concatenate(
    [
        logs["state"][:, 1],
        logs["estimate"][:, 1],
        logs["ref"][:, 1],
    ]
)


margin = 0.18


xmin = float(
    np.min(all_x) - margin
)


xmax = float(
    np.max(all_x) + margin
)


ymin = float(
    np.min(all_y) - margin
)


ymax = float(
    np.max(all_y) + margin
)


# ============================================================
# 2D GIF
# ============================================================

fig_xy, ax_xy = plt.subplots(
    figsize=(7, 7)
)


ax_xy.plot(
    path_points[:, 0],
    path_points[:, 1],
    "o--",
    linewidth=1.2,
    label="Waypoint path",
)


ax_xy.plot(
    logs["ref"][:, 0],
    logs["ref"][:, 1],
    "--",
    linewidth=1.5,
    label="Desired trajectory",
)


true_path_xy, = ax_xy.plot(
    [],
    [],
    linewidth=2.2,
    label="True vehicle path",
)


estimate_path_xy, = ax_xy.plot(
    [],
    [],
    ":",
    linewidth=1.6,
    label="EKF estimated path",
)


desired_marker_xy, = ax_xy.plot(
    [],
    [],
    "x",
    markersize=9,
    markeredgewidth=2,
    label="Desired position",
)


true_marker_xy, = ax_xy.plot(
    [],
    [],
    "o",
    markersize=8,
    label="True vehicle",
)


estimate_marker_xy, = ax_xy.plot(
    [],
    [],
    "s",
    markersize=6,
    label="EKF estimate",
)


ax_xy.set_xlabel(
    "X [m]"
)


ax_xy.set_ylabel(
    "Y [m]"
)


ax_xy.set_title(
    "Optimized IMU/EKF Quadrotor Tracking — 2D"
)


ax_xy.set_aspect(
    "equal",
    adjustable="box",
)


ax_xy.set_xlim(
    xmin,
    xmax,
)


ax_xy.set_ylim(
    ymin,
    ymax,
)


ax_xy.grid(
    True,
    alpha=0.3,
)


ax_xy.legend(
    loc="upper right"
)


info_xy = ax_xy.text(

    0.02,
    0.98,
    "",

    transform=
        ax_xy.transAxes,

    va="top",

    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.87,
    ),
)


def update_xy(
    frame,
):

    true_history = (
        true_anim[
            :frame + 1
        ]
    )


    estimate_history = (
        estimate_anim[
            :frame + 1
        ]
    )


    true_position = (
        true_anim[
            frame
        ]
    )


    estimate_position = (
        estimate_anim[
            frame
        ]
    )


    desired_position = (
        desired_anim[
            frame
        ]
    )


    true_path_xy.set_data(
        true_history[:, 0],
        true_history[:, 1],
    )


    estimate_path_xy.set_data(
        estimate_history[:, 0],
        estimate_history[:, 1],
    )


    true_marker_xy.set_data(
        [true_position[0]],
        [true_position[1]],
    )


    estimate_marker_xy.set_data(
        [estimate_position[0]],
        [estimate_position[1]],
    )


    desired_marker_xy.set_data(
        [desired_position[0]],
        [desired_position[1]],
    )


    tracking_error = float(
        np.linalg.norm(
            true_position
            - desired_position
        )
    )


    ekf_error = float(
        np.linalg.norm(
            true_position
            - estimate_position
        )
    )


    info_xy.set_text(

        f"t = "
        f"{time_anim[frame]:.1f} s\n"

        f"Tracking error = "
        f"{tracking_error:.3f} m\n"

        f"Cross-track error = "
        f"{cross_anim[frame]:.3f} m\n"

        f"EKF position error = "
        f"{ekf_error:.3f} m"
    )


    return (
        true_path_xy,
        estimate_path_xy,
        desired_marker_xy,
        true_marker_xy,
        estimate_marker_xy,
        info_xy,
    )


animation_xy = FuncAnimation(

    fig_xy,

    update_xy,

    frames=len(
        true_anim
    ),

    interval=50,

    blit=False,
)


gif_xy = (

    ROOT
    / "results"
    / "animations"
    / "optimized_2cm_full_gnc_xy.gif"
)


animation_xy.save(

    gif_xy,

    writer=PillowWriter(
        fps=20
    ),
)


plt.close(
    fig_xy
)


# ============================================================
# 3D GIF
# ============================================================

fig_3d = plt.figure(
    figsize=(8, 6)
)


ax_3d = fig_3d.add_subplot(
    111,
    projection="3d",
)


ax_3d.plot(
    path_points[:, 0],
    path_points[:, 1],
    path_points[:, 2],
    "o--",
    linewidth=1.2,
    label="Waypoint path",
)


ax_3d.plot(
    logs["ref"][:, 0],
    logs["ref"][:, 1],
    logs["ref"][:, 2],
    "--",
    linewidth=1.5,
    label="Desired trajectory",
)


true_path_3d, = ax_3d.plot(
    [],
    [],
    [],
    linewidth=2.2,
    label="True vehicle path",
)


estimate_path_3d, = ax_3d.plot(
    [],
    [],
    [],
    ":",
    linewidth=1.6,
    label="EKF estimated path",
)


desired_marker_3d, = ax_3d.plot(
    [],
    [],
    [],
    "x",
    markersize=9,
    markeredgewidth=2,
    label="Desired position",
)


true_marker_3d, = ax_3d.plot(
    [],
    [],
    [],
    "o",
    markersize=8,
    label="True vehicle",
)


estimate_marker_3d, = ax_3d.plot(
    [],
    [],
    [],
    "s",
    markersize=6,
    label="EKF estimate",
)


ax_3d.set_xlabel(
    "X [m]"
)


ax_3d.set_ylabel(
    "Y [m]"
)


ax_3d.set_zlabel(
    "Z [m]"
)


ax_3d.set_title(
    "Optimized IMU/EKF Quadrotor Tracking — 3D"
)


ax_3d.set_xlim(
    xmin,
    xmax,
)


ax_3d.set_ylim(
    ymin,
    ymax,
)


zmin = min(
    np.min(
        logs["state"][:, 2]
    ),
    np.min(
        logs["estimate"][:, 2]
    ),
)


zmax = max(
    np.max(
        logs["state"][:, 2]
    ),
    np.max(
        logs["estimate"][:, 2]
    ),
)


ax_3d.set_zlim(
    max(
        0.0,
        zmin - 0.15,
    ),
    zmax + 0.15,
)


try:

    ax_3d.set_box_aspect(
        (
            1.0,
            1.0,
            0.75,
        )
    )

except Exception:

    pass


ax_3d.legend(
    loc="upper right"
)


def update_3d(
    frame,
):

    true_history = (
        true_anim[
            :frame + 1
        ]
    )


    estimate_history = (
        estimate_anim[
            :frame + 1
        ]
    )


    true_position = (
        true_anim[
            frame
        ]
    )


    estimate_position = (
        estimate_anim[
            frame
        ]
    )


    desired_position = (
        desired_anim[
            frame
        ]
    )


    true_path_3d.set_data(
        true_history[:, 0],
        true_history[:, 1],
    )


    true_path_3d.set_3d_properties(
        true_history[:, 2]
    )


    estimate_path_3d.set_data(
        estimate_history[:, 0],
        estimate_history[:, 1],
    )


    estimate_path_3d.set_3d_properties(
        estimate_history[:, 2]
    )


    true_marker_3d.set_data(
        [true_position[0]],
        [true_position[1]],
    )


    true_marker_3d.set_3d_properties(
        [true_position[2]]
    )


    estimate_marker_3d.set_data(
        [estimate_position[0]],
        [estimate_position[1]],
    )


    estimate_marker_3d.set_3d_properties(
        [estimate_position[2]]
    )


    desired_marker_3d.set_data(
        [desired_position[0]],
        [desired_position[1]],
    )


    desired_marker_3d.set_3d_properties(
        [desired_position[2]]
    )


    return (
        true_path_3d,
        estimate_path_3d,
        desired_marker_3d,
        true_marker_3d,
        estimate_marker_3d,
    )


animation_3d = FuncAnimation(

    fig_3d,

    update_3d,

    frames=len(
        true_anim
    ),

    interval=50,

    blit=False,
)


gif_3d = (

    ROOT
    / "results"
    / "animations"
    / "optimized_2cm_full_gnc_3d.gif"
)


animation_3d.save(

    gif_3d,

    writer=PillowWriter(
        fps=20
    ),
)


plt.close(
    fig_3d
)


# ============================================================
# FINAL REPORT
# ============================================================

print()

print("=" * 84)

print(
    "OPTIMIZED 2 CM FULL-GNC RESULT"
)

print("=" * 84)


print()

print("MISSION")
print("-" * 84)

print(
    f"Waypoints reached:          "
    f"{metrics['waypoints_reached']} / 5"
)


print()

print("TRAJECTORY")
print("-" * 84)

print(
    f"Trajectory RMSE:            "
    f"{metrics['trajectory_position_rmse_m']:.6f} m"
)

print(
    f"Cross-track RMSE:           "
    f"{metrics['cross_track_rmse_m']:.6f} m"
)

print(
    f"Maximum cross-track error:  "
    f"{metrics['cross_track_max_m']:.6f} m"
)

print(
    f"Final waypoint error:       "
    f"{metrics['final_error_m']:.6f} m"
)


print()

print("ESTIMATION")
print("-" * 84)

print(
    f"EKF position RMSE:          "
    f"{metrics['ekf_position_rmse_m']:.6f} m"
)

print(
    f"EKF velocity RMSE:          "
    f"{metrics['ekf_velocity_rmse_mps']:.6f} m/s"
)

print(
    f"EKF roll RMSE:              "
    f"{metrics['ekf_roll_rmse_deg']:.4f} deg"
)

print(
    f"EKF pitch RMSE:             "
    f"{metrics['ekf_pitch_rmse_deg']:.4f} deg"
)

print(
    f"EKF yaw RMSE:               "
    f"{metrics['ekf_yaw_rmse_deg']:.4f} deg"
)


print()

print("CONTROL")
print("-" * 84)

print(
    f"Roll tracking RMSE:         "
    f"{metrics['roll_tracking_rmse_deg']:.4f} deg"
)

print(
    f"Pitch tracking RMSE:        "
    f"{metrics['pitch_tracking_rmse_deg']:.4f} deg"
)

print(
    f"Yaw tracking RMSE:          "
    f"{metrics['yaw_tracking_rmse_deg']:.4f} deg"
)

print(
    f"Thrust saturation:          "
    f"{metrics['thrust_saturation_pct']:.4f} %"
)

print(
    f"Torque saturation:          "
    f"{metrics['torque_saturation_pct']:.4f} %"
)


print()

print("Generated:")
print(gif_xy)
print(gif_3d)
print(metrics_file)

print("=" * 84)