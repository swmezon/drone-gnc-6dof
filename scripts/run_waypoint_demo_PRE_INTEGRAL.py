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
    str(
        ROOT / "src"
    ),
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from drone_gnc.vehicles import (
    QuadrotorModel,
)

from drone_gnc.control.cascaded import (
    CascadedController,
)

from drone_gnc.guidance.waypoints import (
    WaypointGuidance,
)

from drone_gnc.estimation.sensors import (
    ImuGpsSensorSuite,
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
# OUTPUT DIRECTORIES
# ============================================================

for directory in [
    "results/animations",
    "results/guidance",
    "results/estimation",
    "results/data",
]:

    (
        ROOT / directory
    ).mkdir(
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
# VEHICLE MODEL
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
# LOAD OPTIMIZED OUTER-LOOP GAINS
# ============================================================

optimized_gain_path = (
    ROOT
    / "config"
    / "optimized_outer_loop_gains.json"
)


if optimized_gain_path.exists():

    optimized_gain_data = json.loads(
        optimized_gain_path.read_text()
    )

    ctrl.kp_pos = np.array(
        optimized_gain_data["kp_pos"],
        dtype=float,
    )

    ctrl.kd_pos = np.array(
        optimized_gain_data["kd_pos"],
        dtype=float,
    )

    print()

    print(
        "=" * 64
    )

    print(
        "USING OPTIMIZED OUTER-LOOP GAINS"
    )

    print(
        "=" * 64
    )

    print(
        f"kp_pos = {ctrl.kp_pos}"
    )

    print(
        f"kd_pos = {ctrl.kd_pos}"
    )

    print(
        "=" * 64
    )

else:

    print()

    print(
        "WARNING:"
    )

    print(
        "Optimized gain file not found."
    )

    print(
        "Using controller default gains."
    )

    print()


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
# TRAJECTORY TRACKING METRICS
# ============================================================

metrics = trajectory_metrics(

    logs["state"][:, 0:3],

    logs["ref"],

    waypoint_indices=
        logs["wp_index"],

    final_waypoint=
        waypoints[-1],

    waypoints_reached=
        int(
            logs[
                "waypoints_reached"
            ][-1]
        ),
)


# ============================================================
# NAVIGATION / EKF METRICS
# ============================================================

metrics.update(

    estimation_metrics(

        logs["state"],

        logs["estimate"],
    )
)


# ============================================================
# ATTITUDE CONTROL METRICS
# ============================================================

metrics.update(

    attitude_tracking_metrics(

        logs["state"][:, 6:9],

        logs["att_ref"],
    )
)


# ============================================================
# CROSS-TRACK ERROR FUNCTION
# ============================================================

def point_segment_distance(
    p,
    a,
    b,
):

    ab = (
        b - a
    )

    den = float(
        ab @ ab
    )

    if den < 1e-12:

        return float(
            np.linalg.norm(
                p - a
            )
        )

    q = np.clip(

        (
            (p - a) @ ab
        )
        / den,

        0.0,
        1.0,
    )

    projection = (
        a
        + q * ab
    )

    return float(
        np.linalg.norm(
            p - projection
        )
    )


# ============================================================
# COMPLETE PATH GEOMETRY
# ============================================================

# Include the takeoff segment from the
# initial vehicle position to waypoint 1.

path_points = np.vstack(
    [
        logs[
            "state"
        ][0, 0:3],

        waypoints,
    ]
)


# ============================================================
# CROSS-TRACK METRICS
# ============================================================

path_err = []


for pnt in logs[
    "state"
][:, 0:3]:

    path_err.append(

        min(

            point_segment_distance(

                pnt,

                path_points[j],

                path_points[
                    j + 1
                ],
            )

            for j in range(
                len(
                    path_points
                )
                - 1
            )
        )
    )


path_err = np.asarray(
    path_err
)


metrics[
    "cross_track_rmse_m"
] = float(

    np.sqrt(

        np.mean(

            path_err**2
        )
    )
)


metrics[
    "cross_track_max_m"
] = float(

    np.max(
        path_err
    )
)


# ============================================================
# ACTUATOR METRICS
# ============================================================

thrust = (
    logs[
        "control"
    ][:, 0]
)


torque = (
    logs[
        "control"
    ][:, 1:4]
)


metrics[
    "max_thrust_N"
] = float(

    np.max(
        thrust
    )
)


metrics[
    "max_torque_Nm"
] = float(

    np.max(

        np.abs(
            torque
        )
    )
)


metrics[
    "thrust_saturation_pct"
] = float(

    100.0

    * np.mean(

        np.isclose(

            thrust,

            ctrl.limits.max_thrust,

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

                np.abs(
                    torque
                ),

                ctrl.limits.max_torque,

                atol=1e-9,
            ),

            axis=1,
        )
    )
)


# ============================================================
# SIMULATION INFORMATION
# ============================================================

metrics[
    "duration_s"
] = float(
    logs["t"][-1]
)


metrics[
    "sample_rate_hz"
] = float(
    1.0 / DT
)


metrics[
    "gps_rate_hz"
] = float(
    GPS_HZ
)


metrics[
    "mission_complete"
] = bool(

    logs[
        "mission_complete"
    ][-1]
)


# ============================================================
# SAVE OPTIMIZED GAINS USED IN THIS RUN
# ============================================================

metrics[
    "kp_pos"
] = ctrl.kp_pos.tolist()


metrics[
    "kd_pos"
] = ctrl.kd_pos.tolist()


# ============================================================
# SAVE METRICS JSON
# ============================================================

(
    ROOT
    / "results/data/waypoint_metrics.json"
).write_text(

    json.dumps(
        metrics,
        indent=2,
    )
)


# ============================================================
# SAVE FULL VALIDATION CSV
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

    ROOT
    / "results/data/waypoint_log.csv",

    csv_data,

    delimiter=",",

    header=csv_header,

    comments="",
)


# ============================================================
# TRAJECTORY PLOT
# ============================================================

fig = plt.figure(
    figsize=(8, 6)
)


ax = fig.add_subplot(
    111,
    projection="3d",
)


ax.plot(

    logs["state"][:, 0],

    logs["state"][:, 1],

    logs["state"][:, 2],

    label="Actual trajectory",
)


ax.plot(

    logs["ref"][:, 0],

    logs["ref"][:, 1],

    logs["ref"][:, 2],

    "--",

    label="Desired trajectory",
)


ax.plot(

    waypoints[:, 0],

    waypoints[:, 1],

    waypoints[:, 2],

    "o",

    label="Waypoints",
)


ax.set_xlabel(
    "X [m]"
)

ax.set_ylabel(
    "Y [m]"
)

ax.set_zlabel(
    "Z [m]"
)

ax.set_title(
    "Closed-Loop Waypoint Tracking"
)

ax.legend()


fig.tight_layout()


fig.savefig(

    ROOT
    / "results/guidance/"
      "waypoint_tracking_3d.png",

    dpi=180,
)


plt.close(
    fig
)


# ============================================================
# EKF POSITION PLOT
# ============================================================

fig, axs = plt.subplots(

    3,
    1,

    figsize=(9, 7),

    sharex=True,
)


for i, label in enumerate(
    [
        "X",
        "Y",
        "Z",
    ]
):

    axs[i].plot(

        logs["t"],

        logs["state"][:, i],

        label="True",
    )


    axs[i].plot(

        logs["t"],

        logs["estimate"][:, i],

        label="EKF",

        alpha=0.8,
    )


    axs[i].set_ylabel(
        f"{label} [m]"
    )


    axs[i].grid(
        True,
        alpha=0.3,
    )


axs[0].legend()


axs[-1].set_xlabel(
    "Time [s]"
)


fig.suptitle(
    "GPS/IMU State Estimation"
)


fig.tight_layout()


fig.savefig(

    ROOT
    / "results/estimation/"
      "ekf_position_tracking.png",

    dpi=180,
)


plt.close(
    fig
)


# ============================================================
# ANIMATION SETTINGS
# ============================================================

stride = 12


pts = (
    logs[
        "state"
    ][::stride, 0:3]
)


refs = (
    logs[
        "ref"
    ][::stride, 0:3]
)


fig = plt.figure(
    figsize=(7, 5)
)


ax = fig.add_subplot(
    111,
    projection="3d",
)


# ============================================================
# WAYPOINT PATH
# ============================================================

ax.plot(

    path_points[:, 0],

    path_points[:, 1],

    path_points[:, 2],

    "o--",

    label="Waypoint path",
)


# ============================================================
# VEHICLE TRAIL
# ============================================================

trail, = ax.plot(

    [],
    [],
    [],

    lw=2,

    label="Vehicle path",
)


# ============================================================
# DESIRED POSITION MARKER
# ============================================================

desired_marker, = ax.plot(

    [],
    [],
    [],

    "x",

    markersize=8,

    label="Desired position",
)


# ============================================================
# QUADROTOR MARKER
# ============================================================

marker, = ax.plot(

    [],
    [],
    [],

    "o",

    markersize=8,

    label="Quadrotor",
)


# ============================================================
# AXIS LIMITS
# ============================================================

ax.set_xlim(
    -0.5,
    2.5,
)

ax.set_ylim(
    -0.5,
    2.5,
)

ax.set_zlim(
    0.0,
    2.2,
)


ax.set_xlabel(
    "X [m]"
)

ax.set_ylabel(
    "Y [m]"
)

ax.set_zlabel(
    "Z [m]"
)


ax.set_title(
    "Closed-Loop Waypoint Tracking"
)


ax.legend(
    loc="upper right"
)


# ============================================================
# ANIMATION UPDATE
# ============================================================

def update(i):

    q = pts[
        : i + 1
    ]


    trail.set_data(
        q[:, 0],
        q[:, 1],
    )


    trail.set_3d_properties(
        q[:, 2]
    )


    marker.set_data(
        [
            q[-1, 0]
        ],
        [
            q[-1, 1]
        ],
    )


    marker.set_3d_properties(
        [
            q[-1, 2]
        ]
    )


    desired_marker.set_data(
        [
            refs[i, 0]
        ],
        [
            refs[i, 1]
        ],
    )


    desired_marker.set_3d_properties(
        [
            refs[i, 2]
        ]
    )


    return (
        trail,
        marker,
        desired_marker,
    )


# ============================================================
# BUILD GIF
# ============================================================

ani = FuncAnimation(

    fig,

    update,

    frames=len(
        pts
    ),

    interval=45,

    blit=False,
)


ani.save(

    ROOT
    / "results/animations/"
      "waypoint_tracking.gif",

    writer=PillowWriter(
        fps=20
    ),
)


plt.close(
    fig
)


# ============================================================
# TERMINAL VALIDATION REPORT
# ============================================================

print()

print(
    "=" * 64
)

print(
    "QUADROTOR GNC VALIDATION"
)

print(
    "=" * 64
)


print(
    "\nOPTIMIZED OUTER-LOOP GAINS"
)

print(
    "-" * 64
)


print(
    f"kp_pos:                          "
    f"{ctrl.kp_pos}"
)


print(
    f"kd_pos:                          "
    f"{ctrl.kd_pos}"
)


print(
    "\nMISSION"
)

print(
    "-" * 64
)


print(
    f"Waypoints completed:              "
    f"{metrics['waypoints_reached']} "
    f"/ {len(waypoints)}"
)


print(
    f"Mission complete:                 "
    f"{metrics['mission_complete']}"
)


print(
    f"Mission duration:                 "
    f"{metrics['duration_s']:.2f} s"
)


print(
    "\nTRAJECTORY TRACKING"
)

print(
    "-" * 64
)


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


print(
    "\nNAVIGATION / ESTIMATION"
)

print(
    "-" * 64
)


print(
    f"EKF position RMSE:                "
    f"{metrics['ekf_position_rmse_m']:.3f} m"
)


print(
    f"EKF velocity RMSE:                "
    f"{metrics['ekf_velocity_rmse_mps']:.3f} m/s"
)


print(
    f"EKF roll RMSE:                    "
    f"{metrics['ekf_roll_rmse_deg']:.3f} deg"
)


print(
    f"EKF pitch RMSE:                   "
    f"{metrics['ekf_pitch_rmse_deg']:.3f} deg"
)


print(
    f"EKF yaw RMSE:                     "
    f"{metrics['ekf_yaw_rmse_deg']:.3f} deg"
)


print(
    "\nCONTROL"
)

print(
    "-" * 64
)


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


print(
    f"Maximum thrust:                   "
    f"{metrics['max_thrust_N']:.3f} N"
)


print(
    f"Maximum torque:                   "
    f"{metrics['max_torque_Nm']:.3f} N m"
)


print(
    f"Thrust saturation:                "
    f"{metrics['thrust_saturation_pct']:.3f} %"
)


print(
    f"Torque saturation:                "
    f"{metrics['torque_saturation_pct']:.3f} %"
)


print(
    "\nRATES"
)

print(
    "-" * 64
)


print(
    f"Controller update rate:           "
    f"{metrics['sample_rate_hz']:.1f} Hz"
)


print(
    f"GPS update rate:                  "
    f"{metrics['gps_rate_hz']:.1f} Hz"
)


print()

print(
    "Metrics JSON:"
)


print(
    ROOT
    / "results/data/"
      "waypoint_metrics.json"
)


print()

print(
    "Generated GIF:"
)


print(
    ROOT
    / "results/animations/"
      "waypoint_tracking.gif"
)


print(
    "=" * 64
)