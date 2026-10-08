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
from drone_gnc.simulation.mission import run_waypoint_mission
from drone_gnc.evaluation.metrics import (
    trajectory_metrics,
    attitude_tracking_metrics,
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

for directory in [
    "results/animations",
    "results/guidance",
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
DURATION = 40.0


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
# LOAD NOMINAL PID GAINS
# ============================================================

GAIN_FILE = ROOT / "config" / "nominal_pid_gains.json"

if not GAIN_FILE.exists():
    raise FileNotFoundError(
        "Could not find config/nominal_pid_gains.json. "
        "Run optimize_nominal_pid.py first."
    )

gain_data = json.loads(
    GAIN_FILE.read_text()
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

controller.kp_pos = KP_POS.copy()
controller.kd_pos = KD_POS.copy()
controller.ki_pos = KI_POS.copy()


print()
print("=" * 72)
print("NOMINAL PERFECT-STATE PID CONTROLLER")
print("=" * 72)
print(f"Kp = {controller.kp_pos}")
print(f"Kv = {controller.kd_pos}")
print(f"Ki = {controller.ki_pos}")
print("=" * 72)


# ============================================================
# GUIDANCE
# ============================================================

guidance = WaypointGuidance(
    WAYPOINTS,
    acceptance_radius=0.25,
    cruise_speed=0.60,
)


# ============================================================
# RUN NOMINAL MISSION
# ============================================================

# No sensors
# No EKF
# No wind
# No Monte Carlo
# Perfect true-state feedback

logs = run_waypoint_mission(
    vehicle,
    controller,
    guidance,
    sensor_suite=None,
    ekf=None,
    duration=DURATION,
    dt=DT,
    gps_hz=10.0,
    wind=None,
)


# ============================================================
# HELPER: POINT-TO-SEGMENT DISTANCE
# ============================================================

def point_segment_distance(p, a, b):
    ab = b - a
    denominator = float(ab @ ab)

    if denominator < 1e-12:
        return float(np.linalg.norm(p - a))

    q = np.clip(
        ((p - a) @ ab) / denominator,
        0.0,
        1.0,
    )

    projection = a + q * ab

    return float(np.linalg.norm(p - projection))


# ============================================================
# PATH GEOMETRY
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

for position in logs["state"][:, 0:3]:
    distance = min(
        point_segment_distance(
            position,
            path_points[j],
            path_points[j + 1],
        )
        for j in range(len(path_points) - 1)
    )
    cross_track_error.append(distance)

cross_track_error = np.asarray(
    cross_track_error,
    dtype=float,
)


# ============================================================
# METRICS
# ============================================================

metrics = trajectory_metrics(
    logs["state"][:, 0:3],
    logs["ref"],
    waypoint_indices=logs["wp_index"],
    final_waypoint=WAYPOINTS[-1],
    waypoints_reached=int(logs["waypoints_reached"][-1]),
)

attitude_metrics = attitude_tracking_metrics(
    logs["state"][:, 6:9],
    logs["att_ref"],
)

metrics["cross_track_rmse_m"] = float(
    np.sqrt(np.mean(cross_track_error**2))
)

metrics["cross_track_max_m"] = float(
    np.max(cross_track_error)
)

metrics.update(attitude_metrics)

metrics["kp_pos"] = KP_POS.tolist()
metrics["kd_pos"] = KD_POS.tolist()
metrics["ki_pos"] = KI_POS.tolist()
metrics["feedback"] = "perfect_true_state"
metrics["ekf_enabled"] = False
metrics["sensor_noise_enabled"] = False
metrics["monte_carlo_enabled"] = False


# ============================================================
# SAVE METRICS
# ============================================================

metrics_file = ROOT / "results" / "data" / "nominal_pid_metrics.json"

metrics_file.write_text(
    json.dumps(
        metrics,
        indent=2,
    )
)


# ============================================================
# STATIC 2D PLOT
# ============================================================

fig2d, ax2d = plt.subplots(figsize=(7, 7))

ax2d.plot(
    path_points[:, 0],
    path_points[:, 1],
    "o--",
    linewidth=1.2,
    label="Waypoint path",
)

ax2d.plot(
    logs["ref"][:, 0],
    logs["ref"][:, 1],
    "--",
    linewidth=1.5,
    label="Desired trajectory",
)

ax2d.plot(
    logs["state"][:, 0],
    logs["state"][:, 1],
    linewidth=2.2,
    label="Vehicle path",
)

ax2d.scatter(
    WAYPOINTS[:, 0],
    WAYPOINTS[:, 1],
    s=45,
    label="Waypoints",
)

ax2d.set_xlabel("X [m]")
ax2d.set_ylabel("Y [m]")
ax2d.set_title("Nominal Closed-Loop Trajectory Tracking (2D Top View)")
ax2d.set_aspect("equal", adjustable="box")
ax2d.grid(True, alpha=0.3)
ax2d.legend(loc="upper right")

margin_xy = 0.30

xmin = min(np.min(logs["state"][:, 0]), np.min(logs["ref"][:, 0]))
xmax = max(np.max(logs["state"][:, 0]), np.max(logs["ref"][:, 0]))
ymin = min(np.min(logs["state"][:, 1]), np.min(logs["ref"][:, 1]))
ymax = max(np.max(logs["state"][:, 1]), np.max(logs["ref"][:, 1]))

ax2d.set_xlim(xmin - margin_xy, xmax + margin_xy)
ax2d.set_ylim(ymin - margin_xy, ymax + margin_xy)

fig2d.tight_layout()

static_2d_path = ROOT / "results" / "guidance" / "nominal_pid_tracking_xy.png"
fig2d.savefig(static_2d_path, dpi=180)
plt.close(fig2d)


# ============================================================
# STATIC 3D PLOT
# ============================================================

fig3d = plt.figure(figsize=(8, 6))
ax3d = fig3d.add_subplot(111, projection="3d")

ax3d.plot(
    path_points[:, 0],
    path_points[:, 1],
    path_points[:, 2],
    "o--",
    linewidth=1.2,
    label="Waypoint path",
)

ax3d.plot(
    logs["ref"][:, 0],
    logs["ref"][:, 1],
    logs["ref"][:, 2],
    "--",
    linewidth=1.5,
    label="Desired trajectory",
)

ax3d.plot(
    logs["state"][:, 0],
    logs["state"][:, 1],
    logs["state"][:, 2],
    linewidth=2.2,
    label="Vehicle path",
)

ax3d.scatter(
    WAYPOINTS[:, 0],
    WAYPOINTS[:, 1],
    WAYPOINTS[:, 2],
    s=45,
    label="Waypoints",
)

ax3d.set_xlabel("X [m]")
ax3d.set_ylabel("Y [m]")
ax3d.set_zlabel("Z [m]")
ax3d.set_title("Nominal Closed-Loop Trajectory Tracking (3D View)")
ax3d.legend(loc="upper right")

zmin = min(np.min(logs["state"][:, 2]), np.min(logs["ref"][:, 2]))
zmax = max(np.max(logs["state"][:, 2]), np.max(logs["ref"][:, 2]))

ax3d.set_xlim(xmin - margin_xy, xmax + margin_xy)
ax3d.set_ylim(ymin - margin_xy, ymax + margin_xy)
ax3d.set_zlim(max(0.0, zmin - 0.2), zmax + 0.2)

try:
    ax3d.set_box_aspect((1.0, 1.0, 0.8))
except Exception:
    pass

fig3d.tight_layout()

static_3d_path = ROOT / "results" / "guidance" / "nominal_pid_tracking_3d.png"
fig3d.savefig(static_3d_path, dpi=180)
plt.close(fig3d)


# ============================================================
# ANIMATION DATA
# ============================================================

stride = 10

time_anim = logs["t"][::stride]
actual_anim = logs["state"][::stride, 0:3]
desired_anim = logs["ref"][::stride, 0:3]
cross_anim = cross_track_error[::stride]


# ============================================================
# 2D GIF
# ============================================================

fig_xy, ax_xy = plt.subplots(figsize=(7, 7))

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

vehicle_trail_xy, = ax_xy.plot(
    [],
    [],
    linewidth=2.3,
    label="Vehicle path",
)

desired_marker_xy, = ax_xy.plot(
    [],
    [],
    "x",
    markersize=9,
    markeredgewidth=2,
    label="Desired position",
)

vehicle_marker_xy, = ax_xy.plot(
    [],
    [],
    "o",
    markersize=8,
    label="Quadrotor",
)

error_line_xy, = ax_xy.plot(
    [],
    [],
    ":",
    linewidth=1.4,
    label="Position error",
)

ax_xy.set_xlabel("X [m]")
ax_xy.set_ylabel("Y [m]")
ax_xy.set_title("Nominal PID Trajectory Tracking (2D Top View)")
ax_xy.set_aspect("equal", adjustable="box")
ax_xy.grid(True, alpha=0.3)
ax_xy.set_xlim(xmin - margin_xy, xmax + margin_xy)
ax_xy.set_ylim(ymin - margin_xy, ymax + margin_xy)
ax_xy.legend(loc="upper right")

info_text_xy = ax_xy.text(
    0.02,
    0.98,
    "",
    transform=ax_xy.transAxes,
    va="top",
    ha="left",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85,
    ),
)


def update_xy(frame_index):
    history = actual_anim[: frame_index + 1]
    vehicle_position = actual_anim[frame_index]
    desired_position = desired_anim[frame_index]

    vehicle_trail_xy.set_data(
        history[:, 0],
        history[:, 1],
    )

    vehicle_marker_xy.set_data(
        [vehicle_position[0]],
        [vehicle_position[1]],
    )

    desired_marker_xy.set_data(
        [desired_position[0]],
        [desired_position[1]],
    )

    error_line_xy.set_data(
        [desired_position[0], vehicle_position[0]],
        [desired_position[1], vehicle_position[1]],
    )

    xy_error = float(
        np.linalg.norm(vehicle_position[0:2] - desired_position[0:2])
    )

    error_3d = float(
        np.linalg.norm(vehicle_position - desired_position)
    )

    info_text_xy.set_text(
        f"t = {time_anim[frame_index]:.1f} s\n"
        f"XY error = {xy_error:.3f} m\n"
        f"3D trajectory error = {error_3d:.3f} m\n"
        f"Cross-track error = {cross_anim[frame_index]:.3f} m\n"
        f"z actual = {vehicle_position[2]:.3f} m\n"
        f"z desired = {desired_position[2]:.3f} m"
    )

    return (
        vehicle_trail_xy,
        vehicle_marker_xy,
        desired_marker_xy,
        error_line_xy,
        info_text_xy,
    )


anim_xy = FuncAnimation(
    fig_xy,
    update_xy,
    frames=len(actual_anim),
    interval=50,
    blit=False,
)

gif_2d_path = ROOT / "results" / "animations" / "nominal_pid_tracking_xy.gif"

anim_xy.save(
    gif_2d_path,
    writer=PillowWriter(fps=20),
)

plt.close(fig_xy)


# ============================================================
# 3D GIF
# ============================================================

fig_3d_anim = plt.figure(figsize=(8, 6))
ax_3d_anim = fig_3d_anim.add_subplot(111, projection="3d")

ax_3d_anim.plot(
    path_points[:, 0],
    path_points[:, 1],
    path_points[:, 2],
    "o--",
    linewidth=1.2,
    label="Waypoint path",
)

ax_3d_anim.plot(
    logs["ref"][:, 0],
    logs["ref"][:, 1],
    logs["ref"][:, 2],
    "--",
    linewidth=1.5,
    label="Desired trajectory",
)

vehicle_trail_3d, = ax_3d_anim.plot(
    [],
    [],
    [],
    linewidth=2.3,
    label="Vehicle path",
)

desired_marker_3d, = ax_3d_anim.plot(
    [],
    [],
    [],
    "x",
    markersize=9,
    markeredgewidth=2,
    label="Desired position",
)

vehicle_marker_3d, = ax_3d_anim.plot(
    [],
    [],
    [],
    "o",
    markersize=8,
    label="Quadrotor",
)

ax_3d_anim.set_xlabel("X [m]")
ax_3d_anim.set_ylabel("Y [m]")
ax_3d_anim.set_zlabel("Z [m]")
ax_3d_anim.set_title("Nominal PID Trajectory Tracking (3D View)")
ax_3d_anim.set_xlim(xmin - margin_xy, xmax + margin_xy)
ax_3d_anim.set_ylim(ymin - margin_xy, ymax + margin_xy)
ax_3d_anim.set_zlim(max(0.0, zmin - 0.2), zmax + 0.2)

try:
    ax_3d_anim.set_box_aspect((1.0, 1.0, 0.8))
except Exception:
    pass

ax_3d_anim.legend(loc="upper right")


def update_3d(frame_index):
    history = actual_anim[: frame_index + 1]
    vehicle_position = actual_anim[frame_index]
    desired_position = desired_anim[frame_index]

    vehicle_trail_3d.set_data(
        history[:, 0],
        history[:, 1],
    )
    vehicle_trail_3d.set_3d_properties(history[:, 2])

    vehicle_marker_3d.set_data(
        [vehicle_position[0]],
        [vehicle_position[1]],
    )
    vehicle_marker_3d.set_3d_properties([vehicle_position[2]])

    desired_marker_3d.set_data(
        [desired_position[0]],
        [desired_position[1]],
    )
    desired_marker_3d.set_3d_properties([desired_position[2]])

    return (
        vehicle_trail_3d,
        vehicle_marker_3d,
        desired_marker_3d,
    )


anim_3d = FuncAnimation(
    fig_3d_anim,
    update_3d,
    frames=len(actual_anim),
    interval=50,
    blit=False,
)

gif_3d_path = ROOT / "results" / "animations" / "nominal_pid_tracking_3d.gif"

anim_3d.save(
    gif_3d_path,
    writer=PillowWriter(fps=20),
)

plt.close(fig_3d_anim)


# ============================================================
# TERMINAL REPORT
# ============================================================

print()
print("=" * 72)
print("NOMINAL PID TRAJECTORY-TRACKING RESULT")
print("=" * 72)

print()
print("CONTROLLER")
print("-" * 72)
print(f"Kp:                              {KP_POS}")
print(f"Kv:                              {KD_POS}")
print(f"Ki:                              {KI_POS}")

print()
print("MISSION")
print("-" * 72)
print(
    f"Waypoints reached:               "
    f"{metrics['waypoints_reached']} / {len(WAYPOINTS)}"
)

print()
print("TRAJECTORY TRACKING")
print("-" * 72)
print(f"Trajectory RMSE:                 {metrics['trajectory_position_rmse_m']:.4f} m")
print(f"Cross-track RMSE:                {metrics['cross_track_rmse_m']:.4f} m")
print(f"Maximum cross-track error:       {metrics['cross_track_max_m']:.4f} m")
print(f"Final waypoint error:            {metrics['final_error_m']:.4f} m")

print()
print("ATTITUDE TRACKING")
print("-" * 72)
print(f"Roll RMSE:                       {metrics['roll_tracking_rmse_deg']:.4f} deg")
print(f"Pitch RMSE:                      {metrics['pitch_tracking_rmse_deg']:.4f} deg")
print(f"Yaw RMSE:                        {metrics['yaw_tracking_rmse_deg']:.4f} deg")

print()
print("Generated files:")
print(gif_2d_path)
print(gif_3d_path)
print(static_2d_path)
print(static_3d_path)
print("=" * 72)