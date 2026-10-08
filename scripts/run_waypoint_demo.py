import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter


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
from drone_gnc.control.cascaded import CascadedController
from drone_gnc.guidance.waypoints import WaypointGuidance
from drone_gnc.estimation.sensors import ImuGpsSensorSuite
from drone_gnc.estimation.error_state_ekf import ErrorStateEKF15
from drone_gnc.simulation.mission import run_waypoint_mission
from drone_gnc.evaluation.metrics import (
    trajectory_metrics,
    estimation_metrics,
    attitude_tracking_metrics,
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

for directory in [
    "results/animations",
    "results/guidance",
    "results/estimation",
    "results/data",
]:
    (ROOT / directory).mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# SIMULATION SETTINGS
# ============================================================

DT = 0.01
GPS_HZ = 10.0
DURATION = 40.0


# ============================================================
# WAYPOINT MISSION
# ============================================================

waypoints = np.array(
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
# VEHICLE
# ============================================================

vehicle = QuadrotorModel()


# ============================================================
# CONTROLLER
# ============================================================

ctrl = CascadedController(
    vehicle.params.mass,
    vehicle.params.gravity,
)


# ============================================================
# LOAD FINAL ROBUST GAINS
# ============================================================

gain_path = ROOT / "config" / "optimized_outer_loop_gains.json"

if not gain_path.exists():
    raise FileNotFoundError(
        "Could not find config/optimized_outer_loop_gains.json"
    )

gain_data = json.loads(
    gain_path.read_text()
)

ctrl.kp_pos = np.asarray(
    gain_data["kp_pos"],
    dtype=float,
)

ctrl.kd_pos = np.asarray(
    gain_data["kd_pos"],
    dtype=float,
)

ctrl.ki_pos = np.asarray(
    gain_data.get(
        "ki_pos",
        [0.0, 0.0, 0.0],
    ),
    dtype=float,
)

print()
print("=" * 72)
print("FINAL ROBUST CONTROLLER")
print("=" * 72)
print(f"Kp = {ctrl.kp_pos}")
print(f"Kv = {ctrl.kd_pos}")
print(f"Ki = {ctrl.ki_pos}")
print("=" * 72)


# ============================================================
# GUIDANCE
# ============================================================

guidance = WaypointGuidance(
    waypoints,
    acceptance_radius=0.25,
    cruise_speed=0.60,
)


# ============================================================
# SENSOR MODEL
# ============================================================

sensors = ImuGpsSensorSuite(
    seed=24
)


# ============================================================
# EKF
# ============================================================

ekf = ErrorStateEKF15(
    dt=DT
)


# ============================================================
# RUN CLOSED-LOOP MISSION
# ============================================================

logs = run_waypoint_mission(
    vehicle,
    ctrl,
    guidance,
    sensors,
    ekf,
    duration=DURATION,
    dt=DT,
    gps_hz=GPS_HZ,
)


# ============================================================
# HELPER: POINT-TO-SEGMENT DISTANCE
# ============================================================

def point_segment_distance(
    p,
    a,
    b,
):
    ab = b - a
    den = float(ab @ ab)

    if den < 1e-12:
        return float(
            np.linalg.norm(p - a)
        )

    q = np.clip(
        ((p - a) @ ab) / den,
        0.0,
        1.0,
    )

    projection = a + q * ab

    return float(
        np.linalg.norm(p - projection)
    )


# ============================================================
# COMPLETE WAYPOINT PATH
# ============================================================

path_points = np.vstack(
    [
        logs["state"][0, 0:3],
        waypoints,
    ]
)


# ============================================================
# CROSS-TRACK METRICS
# ============================================================

path_error = []

for position in logs["state"][:, 0:3]:
    path_error.append(
        min(
            point_segment_distance(
                position,
                path_points[j],
                path_points[j + 1],
            )
            for j in range(len(path_points) - 1)
        )
    )

path_error = np.asarray(
    path_error,
    dtype=float,
)


# ============================================================
# TRAJECTORY METRICS
# ============================================================

metrics = trajectory_metrics(
    logs["state"][:, 0:3],
    logs["ref"],
    waypoint_indices=logs["wp_index"],
    final_waypoint=waypoints[-1],
    waypoints_reached=int(
        logs["waypoints_reached"][-1]
    ),
)

metrics.update(
    estimation_metrics(
        logs["state"],
        logs["estimate"],
    )
)

metrics.update(
    attitude_tracking_metrics(
        logs["state"][:, 6:9],
        logs["att_ref"],
    )
)

metrics["cross_track_rmse_m"] = float(
    np.sqrt(
        np.mean(path_error**2)
    )
)

metrics["cross_track_max_m"] = float(
    np.max(path_error)
)

thrust = logs["control"][:, 0]
torque = logs["control"][:, 1:4]

metrics["max_thrust_N"] = float(
    np.max(thrust)
)

metrics["max_torque_Nm"] = float(
    np.max(np.abs(torque))
)

metrics["thrust_saturation_pct"] = float(
    100.0
    * np.mean(
        np.isclose(
            thrust,
            ctrl.limits.max_thrust,
            atol=1e-9,
        )
    )
)

metrics["torque_saturation_pct"] = float(
    100.0
    * np.mean(
        np.any(
            np.isclose(
                np.abs(torque),
                ctrl.limits.max_torque,
                atol=1e-9,
            ),
            axis=1,
        )
    )
)

metrics["duration_s"] = float(logs["t"][-1])
metrics["sample_rate_hz"] = float(1.0 / DT)
metrics["gps_rate_hz"] = float(GPS_HZ)
metrics["mission_complete"] = bool(
    logs["mission_complete"][-1]
)
metrics["kp_pos"] = ctrl.kp_pos.tolist()
metrics["kd_pos"] = ctrl.kd_pos.tolist()
metrics["ki_pos"] = ctrl.ki_pos.tolist()


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = ROOT / "results" / "data" / "waypoint_metrics.json"

metrics_path.write_text(
    json.dumps(
        metrics,
        indent=2,
    )
)


# ============================================================
# SAVE FULL DATA LOG
# ============================================================

csv_data = np.c_[
    logs["t"],
    logs["state"],
    logs["estimate"],
    logs["ref"],
    logs["vel_ref"],
    logs["acc_ref"],
    logs["att_ref"],
    logs["rate_ref"],
    logs["control"],
    logs["wp_index"],
    logs["waypoints_reached"],
]

csv_header = (
    "t,"
    "x,y,z,u,v,w,phi,theta,psi,p,q,r,"
    "est_x,est_y,est_z,"
    "est_u,est_v,est_w,"
    "est_phi,est_theta,est_psi,"
    "est_p,est_q,est_r,"
    "ref_x,ref_y,ref_z,"
    "ref_u,ref_v,ref_w,"
    "ref_ax,ref_ay,ref_az,"
    "ref_phi,ref_theta,ref_psi,"
    "ref_p,ref_q,ref_r,"
    "T,tau_x,tau_y,tau_z,"
    "wp_index,"
    "waypoints_reached"
)

np.savetxt(
    ROOT / "results" / "data" / "waypoint_log.csv",
    csv_data,
    delimiter=",",
    header=csv_header,
    comments="",
)


# ============================================================
# STATIC 3D TRAJECTORY FIGURE
# ============================================================

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.plot(
    logs["ref"][:, 0],
    logs["ref"][:, 1],
    logs["ref"][:, 2],
    "--",
    linewidth=1.5,
    label="Desired trajectory",
)

ax.plot(
    logs["state"][:, 0],
    logs["state"][:, 1],
    logs["state"][:, 2],
    linewidth=2.0,
    label="Vehicle trajectory",
)

ax.scatter(
    waypoints[:, 0],
    waypoints[:, 1],
    waypoints[:, 2],
    s=35,
    label="Waypoints",
)

ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title("Closed-Loop Waypoint Tracking (3D View)")
ax.legend()

try:
    ax.set_box_aspect((1, 1, 0.8))
except Exception:
    pass

fig.tight_layout()
fig.savefig(
    ROOT / "results" / "guidance" / "waypoint_tracking_3d.png",
    dpi=180,
)
plt.close(fig)


# ============================================================
# STATIC XY FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(7, 7))

ax.plot(
    path_points[:, 0],
    path_points[:, 1],
    "o--",
    linewidth=1.2,
    label="Waypoint path",
)

ax.plot(
    logs["ref"][:, 0],
    logs["ref"][:, 1],
    "--",
    linewidth=1.2,
    label="Desired trajectory",
)

ax.plot(
    logs["state"][:, 0],
    logs["state"][:, 1],
    linewidth=2.0,
    label="Vehicle trajectory",
)

ax.scatter(
    waypoints[:, 0],
    waypoints[:, 1],
    s=45,
    label="Waypoints",
)

ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_title("Closed-Loop Waypoint Tracking (Top View)")
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.3)
ax.legend()

xy_margin = 0.35

xmin = min(
    np.min(logs["state"][:, 0]),
    np.min(logs["ref"][:, 0]),
    np.min(waypoints[:, 0]),
)
xmax = max(
    np.max(logs["state"][:, 0]),
    np.max(logs["ref"][:, 0]),
    np.max(waypoints[:, 0]),
)
ymin = min(
    np.min(logs["state"][:, 1]),
    np.min(logs["ref"][:, 1]),
    np.min(waypoints[:, 1]),
)
ymax = max(
    np.max(logs["state"][:, 1]),
    np.max(logs["ref"][:, 1]),
    np.max(waypoints[:, 1]),
)

ax.set_xlim(xmin - xy_margin, xmax + xy_margin)
ax.set_ylim(ymin - xy_margin, ymax + xy_margin)

fig.tight_layout()
fig.savefig(
    ROOT / "results" / "guidance" / "waypoint_tracking_xy.png",
    dpi=180,
)
plt.close(fig)


# ============================================================
# EKF POSITION FIGURE
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(9, 7),
    sharex=True,
)

axis_names = ["X", "Y", "Z"]

for i, axis_name in enumerate(axis_names):
    axes[i].plot(
        logs["t"],
        logs["state"][:, i],
        label="True",
    )
    axes[i].plot(
        logs["t"],
        logs["estimate"][:, i],
        label="EKF",
    )
    axes[i].set_ylabel(f"{axis_name} [m]")
    axes[i].grid(True, alpha=0.3)

axes[0].legend()
axes[-1].set_xlabel("Time [s]")
fig.suptitle("GPS/IMU State Estimation")
fig.tight_layout()

fig.savefig(
    ROOT / "results" / "estimation" / "ekf_position_tracking.png",
    dpi=180,
)
plt.close(fig)


# ============================================================
# TOP-DOWN GIF DATA
# ============================================================

stride = 10

t_anim = logs["t"][::stride]
actual_anim = logs["state"][::stride, 0:3]
desired_anim = logs["ref"][::stride, 0:3]
cross_anim = path_error[::stride]


# ============================================================
# TOP-DOWN GIF FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(7, 7))

# Waypoint path
ax.plot(
    path_points[:, 0],
    path_points[:, 1],
    "o--",
    linewidth=1.3,
    label="Waypoint path",
)

# Desired full trajectory
ax.plot(
    logs["ref"][:, 0],
    logs["ref"][:, 1],
    "--",
    linewidth=1.0,
    alpha=0.75,
    label="Desired trajectory",
)

# Vehicle trail
vehicle_trail, = ax.plot(
    [],
    [],
    linewidth=2.2,
    label="Vehicle path",
)

# Desired marker
desired_marker, = ax.plot(
    [],
    [],
    "x",
    markersize=9,
    markeredgewidth=2.0,
    label="Desired position",
)

# Vehicle marker
vehicle_marker, = ax.plot(
    [],
    [],
    "o",
    markersize=8,
    label="Quadrotor",
)

# Instantaneous error line
error_line, = ax.plot(
    [],
    [],
    ":",
    linewidth=1.5,
    alpha=0.9,
    label="Instantaneous position error",
)

ax.scatter(
    waypoints[:, 0],
    waypoints[:, 1],
    s=50,
)

ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_title("Closed-Loop Waypoint Tracking (Top View)")
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.3)

ax.set_xlim(xmin - xy_margin, xmax + xy_margin)
ax.set_ylim(ymin - xy_margin, ymax + xy_margin)

ax.legend(loc="upper right")

info_text = ax.text(
    0.02,
    0.98,
    "",
    transform=ax.transAxes,
    va="top",
    ha="left",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85,
    ),
)


# ============================================================
# ANIMATION UPDATE
# ============================================================

def update(frame_index):
    vehicle_history = actual_anim[: frame_index + 1]
    current_vehicle = actual_anim[frame_index]
    current_desired = desired_anim[frame_index]

    vehicle_trail.set_data(
        vehicle_history[:, 0],
        vehicle_history[:, 1],
    )

    vehicle_marker.set_data(
        [current_vehicle[0]],
        [current_vehicle[1]],
    )

    desired_marker.set_data(
        [current_desired[0]],
        [current_desired[1]],
    )

    error_line.set_data(
        [current_desired[0], current_vehicle[0]],
        [current_desired[1], current_vehicle[1]],
    )

    instantaneous_position_error = float(
        np.linalg.norm(
            current_vehicle - current_desired
        )
    )

    info_text.set_text(
        f"t = {t_anim[frame_index]:.1f} s\n"
        f"XY position error = {np.linalg.norm(current_vehicle[0:2] - current_desired[0:2]):.3f} m\n"
        f"3D position error = {instantaneous_position_error:.3f} m\n"
        f"Cross-track error = {cross_anim[frame_index]:.3f} m\n"
        f"z actual = {current_vehicle[2]:.3f} m\n"
        f"z desired = {current_desired[2]:.3f} m"
    )

    return (
        vehicle_trail,
        vehicle_marker,
        desired_marker,
        error_line,
        info_text,
    )


# ============================================================
# GENERATE GIF
# ============================================================

animation = FuncAnimation(
    fig,
    update,
    frames=len(actual_anim),
    interval=50,
    blit=False,
)

gif_path = ROOT / "results" / "animations" / "waypoint_tracking.gif"

animation.save(
    gif_path,
    writer=PillowWriter(fps=20),
)

plt.close(fig)


# ============================================================
# FINAL CONSOLE REPORT
# ============================================================

print()
print("=" * 72)
print("FINAL ROBUST-CONTROLLER NOMINAL MISSION")
print("=" * 72)

print()
print("CONTROLLER GAINS")
print("-" * 72)
print(f"Kp:                              {ctrl.kp_pos}")
print(f"Kv:                              {ctrl.kd_pos}")
print(f"Ki:                              {ctrl.ki_pos}")

print()
print("MISSION")
print("-" * 72)
print(
    f"Waypoints completed:              "
    f"{metrics['waypoints_reached']} / {len(waypoints)}"
)
print(
    f"Mission complete:                 "
    f"{metrics['mission_complete']}"
)

print()
print("TRAJECTORY TRACKING")
print("-" * 72)
print(
    f"Trajectory position RMSE:         "
    f"{metrics['trajectory_position_rmse_m']:.3f} m"
)
print(
    f"X-axis trajectory RMSE:           "
    f"{metrics['trajectory_x_rmse_m']:.3f} m"
)
print(
    f"Y-axis trajectory RMSE:           "
    f"{metrics['trajectory_y_rmse_m']:.3f} m"
)
print(
    f"Z-axis trajectory RMSE:           "
    f"{metrics['trajectory_z_rmse_m']:.3f} m"
)
print(
    f"Cross-track RMSE:                 "
    f"{metrics['cross_track_rmse_m']:.3f} m"
)
print(
    f"Maximum cross-track error:        "
    f"{metrics['cross_track_max_m']:.3f} m"
)
print(
    f"Final waypoint error:             "
    f"{metrics['final_error_m']:.3f} m"
)

print()
print("NAVIGATION / ESTIMATION")
print("-" * 72)
print(
    f"EKF position RMSE:                "
    f"{metrics['ekf_position_rmse_m']:.3f} m"
)
print(
    f"EKF velocity RMSE:                "
    f"{metrics['ekf_velocity_rmse_mps']:.3f} m/s"
)

print()
print("CONTROL")
print("-" * 72)
print(
    f"Roll tracking RMSE:               "
    f"{metrics['roll_tracking_rmse_deg']:.3f} deg"
)
print(
    f"Pitch tracking RMSE:              "
    f"{metrics['pitch_tracking_rmse_deg']:.3f} deg"
)
print(
    f"Yaw tracking RMSE:                "
    f"{metrics['yaw_tracking_rmse_deg']:.3f} deg"
)

print()
print("Generated files:")
print(ROOT / "results" / "animations" / "waypoint_tracking.gif")
print(ROOT / "results" / "guidance" / "waypoint_tracking_3d.png")
print(ROOT / "results" / "guidance" / "waypoint_tracking_xy.png")
print("=" * 72)