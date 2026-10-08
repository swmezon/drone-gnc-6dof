import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


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
# MONTE CARLO SETTINGS
# ============================================================

NUM_RUNS = 100

BASE_SEED = 1000


# ============================================================
# SIMULATION SETTINGS
# ============================================================

DT = 0.01

GPS_HZ = 10.0

DURATION = 40.0


# ============================================================
# FIXED SENSOR MODEL
# ============================================================
#
# IMPORTANT:
#
# This remains fixed for every Monte Carlo run.
#
# Only the random realization changes.
# ============================================================

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
# PERFORMANCE TARGETS
# ============================================================
#
# These are NOT used to tune the controller.
#
# They are only used after each run to classify whether the
# trial remained within a tight-performance envelope.
#
# Mission completion and performance qualification are kept
# separate.
# ============================================================

TARGET_TRAJECTORY_RMSE_M = 0.05

TARGET_CROSS_TRACK_RMSE_M = 0.04

TARGET_FINAL_ERROR_M = 0.05


# ============================================================
# FILES
# ============================================================

GAIN_FILE = (
    ROOT
    / "config"
    / "optimized_2cm_full_gnc.json"
)


RESULTS_DIR = (
    ROOT
    / "results"
    / "data"
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CSV_FILE = (
    RESULTS_DIR
    / "monte_carlo_2cm_runs.csv"
)


SUMMARY_FILE = (
    RESULTS_DIR
    / "monte_carlo_2cm_summary.json"
)


# ============================================================
# LOAD FIXED OPTIMIZED CONTROLLER
# ============================================================

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
# POINT-TO-SEGMENT DISTANCE
# ============================================================

def point_segment_distance(
    p,
    a,
    b,
):

    ab = (
        b - a
    )


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
        a
        + q
        * ab
    )


    return float(
        np.linalg.norm(
            p - projection
        )
    )


# ============================================================
# BUILD ONE COMPLETE GNC SYSTEM
# ============================================================

def build_system(
    sensor_seed,
):

    # --------------------------------------------------------
    # VEHICLE
    # --------------------------------------------------------

    vehicle = QuadrotorModel()


    # --------------------------------------------------------
    # CONTROLLER
    # --------------------------------------------------------

    controller = CascadedController(
        vehicle.params.mass,
        vehicle.params.gravity,
    )


    controller.kp_att = (
        KP_ATT.copy()
    )


    controller.kp_rate = (
        KP_RATE.copy()
    )


    controller.ki_rate = (
        KI_RATE.copy()
    )


    controller.kd_rate = (
        KD_RATE.copy()
    )


    controller.kp_pos = (
        KP_POS.copy()
    )


    controller.kd_pos = (
        KD_POS.copy()
    )


    controller.ki_pos = (
        KI_POS.copy()
    )


    # --------------------------------------------------------
    # GUIDANCE
    # --------------------------------------------------------

    guidance = WaypointGuidance(
        WAYPOINTS,
        acceptance_radius=0.25,
        cruise_speed=0.60,
    )


    # --------------------------------------------------------
    # SENSOR MODEL
    # --------------------------------------------------------

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
        seed=sensor_seed,
    )


    # --------------------------------------------------------
    # EKF
    # --------------------------------------------------------

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


    return (
        vehicle,
        controller,
        guidance,
        sensors,
        ekf,
    )


# ============================================================
# CALCULATE CROSS-TRACK ERROR
# ============================================================

def calculate_cross_track(
    positions,
):

    path_points = np.vstack(
        [
            positions[0],
            WAYPOINTS,
        ]
    )


    errors = []


    for position in positions:

        error = min(

            point_segment_distance(
                position,
                path_points[j],
                path_points[j + 1],
            )

            for j in range(
                len(
                    path_points
                )
                - 1
            )
        )


        errors.append(
            error
        )


    errors = np.asarray(
        errors,
        dtype=float,
    )


    return (
        float(
            np.sqrt(
                np.mean(
                    errors**2
                )
            )
        ),

        float(
            np.max(
                errors
            )
        ),
    )


# ============================================================
# RUN ONE MONTE CARLO CASE
# ============================================================

def run_case(
    case_number,
    seed,
):

    (
        vehicle,
        controller,
        guidance,
        sensors,
        ekf,
    ) = build_system(
        seed
    )


    try:

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


    except Exception as exc:

        return {

            "case":
                case_number,

            "seed":
                seed,

            "mission_success":
                False,

            "performance_pass":
                False,

            "waypoints_reached":
                0,

            "trajectory_rmse_m":
                np.nan,

            "cross_track_rmse_m":
                np.nan,

            "cross_track_max_m":
                np.nan,

            "final_error_m":
                np.nan,

            "ekf_position_rmse_m":
                np.nan,

            "ekf_velocity_rmse_mps":
                np.nan,

            "roll_tracking_rmse_deg":
                np.nan,

            "pitch_tracking_rmse_deg":
                np.nan,

            "yaw_tracking_rmse_deg":
                np.nan,

            "thrust_saturation_pct":
                np.nan,

            "torque_saturation_pct":
                np.nan,

            "failure_reason":
                f"exception: {exc}",
        }


    # ========================================================
    # NUMERICAL VALIDITY
    # ========================================================

    if not np.all(
        np.isfinite(
            logs["state"]
        )
    ):

        return {

            "case":
                case_number,

            "seed":
                seed,

            "mission_success":
                False,

            "performance_pass":
                False,

            "waypoints_reached":
                0,

            "trajectory_rmse_m":
                np.nan,

            "cross_track_rmse_m":
                np.nan,

            "cross_track_max_m":
                np.nan,

            "final_error_m":
                np.nan,

            "ekf_position_rmse_m":
                np.nan,

            "ekf_velocity_rmse_mps":
                np.nan,

            "roll_tracking_rmse_deg":
                np.nan,

            "pitch_tracking_rmse_deg":
                np.nan,

            "yaw_tracking_rmse_deg":
                np.nan,

            "thrust_saturation_pct":
                np.nan,

            "torque_saturation_pct":
                np.nan,

            "failure_reason":
                "nonfinite_state",
        }


    # ========================================================
    # TRAJECTORY METRICS
    # ========================================================

    trajectory = trajectory_metrics(

        logs[
            "state"
        ][:, 0:3],

        logs[
            "ref"
        ],

        waypoint_indices=
            logs[
                "wp_index"
            ],

        final_waypoint=
            WAYPOINTS[-1],

        waypoints_reached=
            int(
                logs[
                    "waypoints_reached"
                ][-1]
            ),
    )


    # ========================================================
    # ESTIMATION METRICS
    # ========================================================

    estimation = estimation_metrics(

        logs[
            "state"
        ],

        logs[
            "estimate"
        ],
    )


    # ========================================================
    # ATTITUDE TRACKING
    # ========================================================

    attitude = attitude_tracking_metrics(

        logs[
            "state"
        ][:, 6:9],

        logs[
            "att_ref"
        ],
    )


    # ========================================================
    # CROSS-TRACK
    # ========================================================

    (
        cross_rmse,
        cross_max,
    ) = calculate_cross_track(

        logs[
            "state"
        ][:, 0:3]
    )


    # ========================================================
    # ACTUATOR SATURATION
    # ========================================================

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


    thrust_sat = float(

        100.0

        * np.mean(

            np.isclose(
                thrust,
                controller.limits.max_thrust,
                atol=1e-9,
            )
        )
    )


    torque_sat = float(

        100.0

        * np.mean(

            np.any(

                np.isclose(
                    np.abs(
                        torque
                    ),
                    controller.limits.max_torque,
                    atol=1e-9,
                ),

                axis=1,
            )
        )
    )


    # ========================================================
    # MISSION COMPLETION
    # ========================================================

    waypoints_reached = int(
        logs[
            "waypoints_reached"
        ][-1]
    )


    mission_complete = bool(
        logs[
            "mission_complete"
        ][-1]
    )


    mission_success = bool(

        mission_complete

        and

        waypoints_reached
        == len(
            WAYPOINTS
        )
    )


    trajectory_rmse = float(
        trajectory[
            "trajectory_position_rmse_m"
        ]
    )


    final_error = float(
        trajectory[
            "final_error_m"
        ]
    )


    # ========================================================
    # PERFORMANCE QUALIFICATION
    # ========================================================
    #
    # This is intentionally separate from mission completion.
    #
    # A run may finish all five waypoints but still have larger
    # tracking error than desired.
    # ========================================================

    performance_pass = bool(

        mission_success

        and

        trajectory_rmse
        <= TARGET_TRAJECTORY_RMSE_M

        and

        cross_rmse
        <= TARGET_CROSS_TRACK_RMSE_M

        and

        final_error
        <= TARGET_FINAL_ERROR_M
    )


    # ========================================================
    # FAILURE / STATUS REASON
    # ========================================================

    if not mission_success:

        failure_reason = (
            "mission_not_complete"
        )


    elif trajectory_rmse > TARGET_TRAJECTORY_RMSE_M:

        failure_reason = (
            "trajectory_rmse_limit"
        )


    elif cross_rmse > TARGET_CROSS_TRACK_RMSE_M:

        failure_reason = (
            "cross_track_limit"
        )


    elif final_error > TARGET_FINAL_ERROR_M:

        failure_reason = (
            "final_error_limit"
        )


    else:

        failure_reason = (
            "none"
        )


    # ========================================================
    # RETURN CASE
    # ========================================================

    return {

        "case":
            case_number,

        "seed":
            seed,

        "mission_success":
            mission_success,

        "performance_pass":
            performance_pass,

        "waypoints_reached":
            waypoints_reached,

        "trajectory_rmse_m":
            trajectory_rmse,

        "cross_track_rmse_m":
            cross_rmse,

        "cross_track_max_m":
            cross_max,

        "final_error_m":
            final_error,

        "ekf_position_rmse_m":
            float(
                estimation[
                    "ekf_position_rmse_m"
                ]
            ),

        "ekf_velocity_rmse_mps":
            float(
                estimation[
                    "ekf_velocity_rmse_mps"
                ]
            ),

        "roll_tracking_rmse_deg":
            float(
                attitude[
                    "roll_tracking_rmse_deg"
                ]
            ),

        "pitch_tracking_rmse_deg":
            float(
                attitude[
                    "pitch_tracking_rmse_deg"
                ]
            ),

        "yaw_tracking_rmse_deg":
            float(
                attitude[
                    "yaw_tracking_rmse_deg"
                ]
            ),

        "thrust_saturation_pct":
            thrust_sat,

        "torque_saturation_pct":
            torque_sat,

        "failure_reason":
            failure_reason,
    }


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def summarize_metric(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )


    values = values[
        np.isfinite(
            values
        )
    ]


    if len(
        values
    ) == 0:

        return {
            "mean":
                None,

            "median":
                None,

            "std":
                None,

            "p95":
                None,

            "worst":
                None,

            "best":
                None,
        }


    return {

        "mean":
            float(
                np.mean(
                    values
                )
            ),

        "median":
            float(
                np.median(
                    values
                )
            ),

        "std":
            float(
                np.std(
                    values
                )
            ),

        "p95":
            float(
                np.percentile(
                    values,
                    95.0,
                )
            ),

        "worst":
            float(
                np.max(
                    values
                )
            ),

        "best":
            float(
                np.min(
                    values
                )
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "=" * 90
    )


    print(
        "2 CM SENSOR MONTE CARLO VALIDATION"
    )


    print(
        "=" * 90
    )


    print(
        f"Number of runs:               "
        f"{NUM_RUNS}"
    )


    print(
        f"Position noise sigma:         "
        f"{POSITION_NOISE_M:.3f} m"
    )


    print(
        f"Control / IMU rate:           "
        f"{1.0 / DT:.1f} Hz"
    )


    print(
        f"Absolute measurement rate:    "
        f"{GPS_HZ:.1f} Hz"
    )


    print()

    print(
        "Fixed controller:"
    )


    print(
        f"Kp attitude = "
        f"{KP_ATT}"
    )


    print(
        f"Kp rate     = "
        f"{KP_RATE}"
    )


    print(
        f"Ki rate     = "
        f"{KI_RATE}"
    )


    print(
        f"Kd rate     = "
        f"{KD_RATE}"
    )


    print(
        f"Kp position = "
        f"{KP_POS}"
    )


    print(
        f"Kd position = "
        f"{KD_POS}"
    )


    print(
        f"Ki position = "
        f"{KI_POS}"
    )


    print()

    print(
        "Performance envelope:"
    )


    print(
        f"Trajectory RMSE <= "
        f"{TARGET_TRAJECTORY_RMSE_M:.3f} m"
    )


    print(
        f"Cross-track RMSE <= "
        f"{TARGET_CROSS_TRACK_RMSE_M:.3f} m"
    )


    print(
        f"Final error <= "
        f"{TARGET_FINAL_ERROR_M:.3f} m"
    )


    print(
        "=" * 90
    )


    # ========================================================
    # RUN CASES
    # ========================================================

    results = []


    for index in range(
        NUM_RUNS
    ):

        case_number = (
            index + 1
        )


        seed = (
            BASE_SEED
            + index
        )


        result = run_case(
            case_number,
            seed,
        )


        results.append(
            result
        )


        mission_label = (
            "PASS"
            if result[
                "mission_success"
            ]
            else "FAIL"
        )


        performance_label = (
            "QUAL"
            if result[
                "performance_pass"
            ]
            else "----"
        )


        print(

            f"Case "
            f"{case_number:03d}/{NUM_RUNS}"

            f" | {mission_label}"

            f" | {performance_label}"

            f" | WP "
            f"{result['waypoints_reached']}/5"

            f" | traj "
            f"{result['trajectory_rmse_m']:.4f}"

            f" | cross "
            f"{result['cross_track_rmse_m']:.4f}"

            f" | final "
            f"{result['final_error_m']:.4f}"

            f" | EKF "
            f"{result['ekf_position_rmse_m']:.4f}"

            f" | "
            f"{result['failure_reason']}"
        )


    # ========================================================
    # SAVE CSV
    # ========================================================

    fieldnames = list(
        results[
            0
        ].keys()
    )


    with CSV_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )


        writer.writeheader()


        writer.writerows(
            results
        )


    # ========================================================
    # COUNTS
    # ========================================================

    mission_success_count = sum(

        int(
            result[
                "mission_success"
            ]
        )

        for result in results
    )


    performance_pass_count = sum(

        int(
            result[
                "performance_pass"
            ]
        )

        for result in results
    )


    failure_reasons = Counter(

        result[
            "failure_reason"
        ]

        for result in results

        if result[
            "failure_reason"
        ]
        != "none"
    )


    # ========================================================
    # STATISTICS
    # ========================================================

    trajectory_stats = summarize_metric(
        [
            r[
                "trajectory_rmse_m"
            ]
            for r in results
        ]
    )


    cross_stats = summarize_metric(
        [
            r[
                "cross_track_rmse_m"
            ]
            for r in results
        ]
    )


    cross_max_stats = summarize_metric(
        [
            r[
                "cross_track_max_m"
            ]
            for r in results
        ]
    )


    final_stats = summarize_metric(
        [
            r[
                "final_error_m"
            ]
            for r in results
        ]
    )


    ekf_position_stats = summarize_metric(
        [
            r[
                "ekf_position_rmse_m"
            ]
            for r in results
        ]
    )


    ekf_velocity_stats = summarize_metric(
        [
            r[
                "ekf_velocity_rmse_mps"
            ]
            for r in results
        ]
    )


    roll_stats = summarize_metric(
        [
            r[
                "roll_tracking_rmse_deg"
            ]
            for r in results
        ]
    )


    pitch_stats = summarize_metric(
        [
            r[
                "pitch_tracking_rmse_deg"
            ]
            for r in results
        ]
    )


    # ========================================================
    # SUMMARY JSON
    # ========================================================

    summary = {

        "num_runs":
            NUM_RUNS,

        "position_noise_std_m":
            POSITION_NOISE_M,

        "mission_success_count":
            mission_success_count,

        "mission_success_rate_pct":
            float(
                100.0
                * mission_success_count
                / NUM_RUNS
            ),

        "performance_pass_count":
            performance_pass_count,

        "performance_pass_rate_pct":
            float(
                100.0
                * performance_pass_count
                / NUM_RUNS
            ),

        "performance_limits":
            {

                "trajectory_rmse_m":
                    TARGET_TRAJECTORY_RMSE_M,

                "cross_track_rmse_m":
                    TARGET_CROSS_TRACK_RMSE_M,

                "final_error_m":
                    TARGET_FINAL_ERROR_M,
            },

        "failure_reasons":
            dict(
                failure_reasons
            ),

        "trajectory_rmse_m":
            trajectory_stats,

        "cross_track_rmse_m":
            cross_stats,

        "cross_track_max_m":
            cross_max_stats,

        "final_error_m":
            final_stats,

        "ekf_position_rmse_m":
            ekf_position_stats,

        "ekf_velocity_rmse_mps":
            ekf_velocity_stats,

        "roll_tracking_rmse_deg":
            roll_stats,

        "pitch_tracking_rmse_deg":
            pitch_stats,
    }


    SUMMARY_FILE.write_text(

        json.dumps(
            summary,
            indent=2,
        )
    )


    # ========================================================
    # PRINT FINAL SUMMARY
    # ========================================================

    print()

    print(
        "=" * 90
    )


    print(
        "MONTE CARLO SUMMARY"
    )


    print(
        "=" * 90
    )


    print()

    print(
        "MISSION COMPLETION"
    )


    print(
        "-" * 90
    )


    print(
        f"Successful runs:          "
        f"{mission_success_count} / {NUM_RUNS}"
    )


    print(
        f"Mission success rate:     "
        f"{100.0 * mission_success_count / NUM_RUNS:.1f} %"
    )


    print()

    print(
        "TIGHT-PERFORMANCE QUALIFICATION"
    )


    print(
        "-" * 90
    )


    print(
        f"Qualified runs:           "
        f"{performance_pass_count} / {NUM_RUNS}"
    )


    print(
        f"Qualification rate:       "
        f"{100.0 * performance_pass_count / NUM_RUNS:.1f} %"
    )


    print()

    print(
        "TRAJECTORY RMSE"
    )


    print(
        "-" * 90
    )


    print(
        f"Mean:                     "
        f"{trajectory_stats['mean']:.4f} m"
    )


    print(
        f"Median:                   "
        f"{trajectory_stats['median']:.4f} m"
    )


    print(
        f"95th percentile:          "
        f"{trajectory_stats['p95']:.4f} m"
    )


    print(
        f"Worst case:               "
        f"{trajectory_stats['worst']:.4f} m"
    )


    print()

    print(
        "CROSS-TRACK RMSE"
    )


    print(
        "-" * 90
    )


    print(
        f"Mean:                     "
        f"{cross_stats['mean']:.4f} m"
    )


    print(
        f"Median:                   "
        f"{cross_stats['median']:.4f} m"
    )


    print(
        f"95th percentile:          "
        f"{cross_stats['p95']:.4f} m"
    )


    print(
        f"Worst case:               "
        f"{cross_stats['worst']:.4f} m"
    )


    print()

    print(
        "MAXIMUM CROSS-TRACK ERROR"
    )


    print(
        "-" * 90
    )


    print(
        f"Mean:                     "
        f"{cross_max_stats['mean']:.4f} m"
    )


    print(
        f"95th percentile:          "
        f"{cross_max_stats['p95']:.4f} m"
    )


    print(
        f"Worst case:               "
        f"{cross_max_stats['worst']:.4f} m"
    )


    print()

    print(
        "FINAL WAYPOINT ERROR"
    )


    print(
        "-" * 90
    )


    print(
        f"Mean:                     "
        f"{final_stats['mean']:.4f} m"
    )


    print(
        f"Median:                   "
        f"{final_stats['median']:.4f} m"
    )


    print(
        f"95th percentile:          "
        f"{final_stats['p95']:.4f} m"
    )


    print(
        f"Worst case:               "
        f"{final_stats['worst']:.4f} m"
    )


    print()

    print(
        "EKF POSITION RMSE"
    )


    print(
        "-" * 90
    )


    print(
        f"Mean:                     "
        f"{ekf_position_stats['mean']:.4f} m"
    )


    print(
        f"95th percentile:          "
        f"{ekf_position_stats['p95']:.4f} m"
    )


    print(
        f"Worst case:               "
        f"{ekf_position_stats['worst']:.4f} m"
    )


    print()

    print(
        "ATTITUDE TRACKING"
    )


    print(
        "-" * 90
    )


    print(
        f"Roll RMSE mean:           "
        f"{roll_stats['mean']:.3f} deg"
    )


    print(
        f"Roll RMSE p95:            "
        f"{roll_stats['p95']:.3f} deg"
    )


    print(
        f"Pitch RMSE mean:          "
        f"{pitch_stats['mean']:.3f} deg"
    )


    print(
        f"Pitch RMSE p95:           "
        f"{pitch_stats['p95']:.3f} deg"
    )


    print()

    print(
        "FAILURE REASONS"
    )


    print(
        "-" * 90
    )


    print(
        dict(
            failure_reasons
        )
    )


    print()

    print(
        "Saved CSV:"
    )


    print(
        CSV_FILE
    )


    print()

    print(
        "Saved summary:"
    )


    print(
        SUMMARY_FILE
    )


    print(
        "=" * 90
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()