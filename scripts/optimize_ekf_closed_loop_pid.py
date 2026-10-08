import json
import sys
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
# SIMULATION SETTINGS
# ============================================================

DT = 0.01

GPS_HZ = 10.0

DURATION = 40.0

SENSOR_SEED = 24


# ============================================================
# WAYPOINT MISSION
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
# FILES
# ============================================================

NOMINAL_PID_FILE = (
    ROOT
    / "config"
    / "nominal_pid_gains.json"
)


EKF_CONFIG_FILE = (
    ROOT
    / "config"
    / "optimized_ekf_covariance.json"
)


OUTPUT_GAIN_FILE = (
    ROOT
    / "config"
    / "ekf_closed_loop_pid_gains.json"
)


HISTORY_FILE = (
    ROOT
    / "results"
    / "data"
    / "ekf_closed_loop_pid_optimization_history.json"
)


# ============================================================
# VERIFY REQUIRED FILES
# ============================================================

if not NOMINAL_PID_FILE.exists():

    raise FileNotFoundError(
        "Could not find "
        "config/nominal_pid_gains.json"
    )


if not EKF_CONFIG_FILE.exists():

    raise FileNotFoundError(
        "Could not find "
        "config/optimized_ekf_covariance.json"
    )


# ============================================================
# LOAD NOMINAL PID STARTING POINT
# ============================================================

nominal_pid = json.loads(
    NOMINAL_PID_FILE.read_text()
)


INITIAL_GAINS = np.r_[
    np.asarray(
        nominal_pid["kp_pos"],
        dtype=float,
    ),

    np.asarray(
        nominal_pid["kd_pos"],
        dtype=float,
    ),

    np.asarray(
        nominal_pid["ki_pos"],
        dtype=float,
    ),
]


# ============================================================
# LOAD OPTIMIZED EKF CONFIGURATION
# ============================================================

ekf_cfg = json.loads(
    EKF_CONFIG_FILE.read_text()
)


P0_SCALE = float(
    ekf_cfg["p0_scale"]
)


Q_ACCEL_SCALE = float(
    ekf_cfg["q_accel_scale"]
)


Q_GYRO_SCALE = float(
    ekf_cfg["q_gyro_scale"]
)


Q_ACCEL_BIAS_SCALE = float(
    ekf_cfg["q_accel_bias_scale"]
)


Q_GYRO_BIAS_SCALE = float(
    ekf_cfg["q_gyro_bias_scale"]
)


R_GPS_POS_SCALE = float(
    ekf_cfg["r_gps_pos_scale"]
)


R_GPS_VEL_SCALE = float(
    ekf_cfg["r_gps_vel_scale"]
)


# ============================================================
# PARAMETER VECTOR
# ============================================================

# [
#   Kpx, Kpy, Kpz,
#   Kvx, Kvy, Kvz,
#   Kix, Kiy, Kiz
# ]

PARAMETER_NAMES = [
    "Kpx",
    "Kpy",
    "Kpz",

    "Kvx",
    "Kvy",
    "Kvz",

    "Kix",
    "Kiy",
    "Kiz",
]


# ============================================================
# SEARCH BOUNDS
# ============================================================

LOWER_BOUNDS = np.array(
    [
        0.20,
        0.20,
        0.50,

        0.50,
        0.50,
        0.80,

        0.00,
        0.00,
        0.00,
    ],
    dtype=float,
)


UPPER_BOUNDS = np.array(
    [
        8.00,
        8.00,
        8.00,

        10.00,
        10.00,
        10.00,

        1.50,
        1.50,
        1.50,
    ],
    dtype=float,
)


# ============================================================
# INITIAL SEARCH STEP
# ============================================================

INITIAL_STEP = np.array(
    [
        0.30,
        0.30,
        0.30,

        0.30,
        0.30,
        0.30,

        0.05,
        0.05,
        0.05,
    ],
    dtype=float,
)


MIN_STEP = np.array(
    [
        0.025,
        0.025,
        0.025,

        0.025,
        0.025,
        0.025,

        0.0125,
        0.0125,
        0.0125,
    ],
    dtype=float,
)


MAX_SWEEPS = 10


# ============================================================
# OBJECTIVE WEIGHTS
# ============================================================

# Main goal:
#
# 1. complete all 5 waypoints
# 2. stay close to path
# 3. suppress serpentine motion
# 4. maintain reasonable attitude commands
# 5. avoid saturation

W_TRAJECTORY_RMSE = 2.0

W_CROSS_TRACK_RMSE = 7.0

W_CROSS_TRACK_MAX = 3.0

W_FINAL_ERROR = 2.0

# Strongly penalizes sideways motion relative to path direction.
W_LATERAL_VELOCITY = 4.0

# Penalizes extra distance traveled due to snaking.
W_PATH_LENGTH_EXCESS = 3.0

# Prevent excessive roll/pitch oscillation.
W_ROLL_PITCH_TRACKING = 0.10

W_TORQUE_SAT = 0.20

W_THRUST_SAT = 0.20


# ============================================================
# HELPER: POINT-TO-SEGMENT DISTANCE
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
        + q * ab
    )


    return float(
        np.linalg.norm(
            p - projection
        )
    )


# ============================================================
# BUILD EKF USING OPTIMIZED COVARIANCE
# ============================================================

def make_ekf(
    sensor_cfg,
):

    ekf = ErrorStateEKF15(

        dt=DT,

        gps_pos_std=(
            sensor_cfg.gps_pos_std
            * np.sqrt(
                R_GPS_POS_SCALE
            )
        ),

        gps_vel_std=(
            sensor_cfg.gps_vel_std
            * np.sqrt(
                R_GPS_VEL_SCALE
            )
        ),

        accel_noise_std=(
            sensor_cfg.accel_noise_std
            * np.sqrt(
                Q_ACCEL_SCALE
            )
        ),

        gyro_noise_std=(
            sensor_cfg.gyro_noise_std
            * np.sqrt(
                Q_GYRO_SCALE
            )
        ),

        accel_bias_rw_std=(
            sensor_cfg.accel_bias_rw_std
            * np.sqrt(
                Q_ACCEL_BIAS_SCALE
            )
        ),

        gyro_bias_rw_std=(
            sensor_cfg.gyro_bias_rw_std
            * np.sqrt(
                Q_GYRO_BIAS_SCALE
            )
        ),
    )


    ekf.P *= (
        P0_SCALE
    )


    return ekf


# ============================================================
# COMPUTE CROSS-TRACK METRICS
# ============================================================

def calculate_cross_track_metrics(
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

        errors.append(

            min(

                point_segment_distance(
                    position,
                    path_points[j],
                    path_points[j + 1],
                )

                for j in range(
                    len(path_points) - 1
                )
            )
        )


    errors = np.asarray(
        errors,
        dtype=float,
    )


    rmse = float(
        np.sqrt(
            np.mean(
                errors**2
            )
        )
    )


    maximum = float(
        np.max(
            errors
        )
    )


    return (
        rmse,
        maximum,
        errors,
    )


# ============================================================
# LATERAL VELOCITY METRIC
# ============================================================

def calculate_lateral_velocity_rmse(
    velocity,
    velocity_reference,
):

    lateral_squared = []


    for v, vref in zip(
        velocity,
        velocity_reference,
    ):

        tangent_xy = np.asarray(
            vref[0:2],
            dtype=float,
        )


        speed_ref = float(
            np.linalg.norm(
                tangent_xy
            )
        )


        if speed_ref < 1e-6:

            continue


        tangent_xy = (
            tangent_xy
            / speed_ref
        )


        actual_xy = np.asarray(
            v[0:2],
            dtype=float,
        )


        along_track_velocity = (
            np.dot(
                actual_xy,
                tangent_xy,
            )
            * tangent_xy
        )


        lateral_velocity = (
            actual_xy
            - along_track_velocity
        )


        lateral_squared.append(
            float(
                lateral_velocity
                @ lateral_velocity
            )
        )


    if len(
        lateral_squared
    ) == 0:

        return 0.0


    return float(
        np.sqrt(
            np.mean(
                lateral_squared
            )
        )
    )


# ============================================================
# PATH-LENGTH EXCESS
# ============================================================

def calculate_path_length_excess(
    positions,
):

    delta = np.diff(
        positions,
        axis=0,
    )


    actual_length = float(
        np.sum(
            np.linalg.norm(
                delta,
                axis=1,
            )
        )
    )


    # Desired geometric path:
    #
    # vertical takeoff 1.5 m
    # +
    # square perimeter 8 m

    reference_points = np.vstack(
        [
            positions[0],
            WAYPOINTS,
        ]
    )


    reference_delta = np.diff(
        reference_points,
        axis=0,
    )


    reference_length = float(
        np.sum(
            np.linalg.norm(
                reference_delta,
                axis=1,
            )
        )
    )


    excess = max(
        0.0,
        actual_length
        - reference_length,
    )


    return (
        excess,
        actual_length,
        reference_length,
    )


# ============================================================
# RUN ONE PID CANDIDATE
# ============================================================

def evaluate_candidate(
    gains,
):

    gains = np.asarray(
        gains,
        dtype=float,
    )


    # --------------------------------------------------------
    # VEHICLE
    # --------------------------------------------------------

    vehicle = QuadrotorModel()


    # --------------------------------------------------------
    # PID CONTROLLER
    # --------------------------------------------------------

    controller = CascadedController(
        vehicle.params.mass,
        vehicle.params.gravity,
    )


    controller.kp_pos = np.asarray(
        gains[0:3],
        dtype=float,
    )


    controller.kd_pos = np.asarray(
        gains[3:6],
        dtype=float,
    )


    controller.ki_pos = np.asarray(
        gains[6:9],
        dtype=float,
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

    sensor_cfg = SensorConfig()


    sensors = ImuGpsSensorSuite(
        config=sensor_cfg,
        seed=SENSOR_SEED,
    )


    # --------------------------------------------------------
    # OPTIMIZED EKF
    # --------------------------------------------------------

    ekf = make_ekf(
        sensor_cfg
    )


    # --------------------------------------------------------
    # RUN CLOSED LOOP
    # --------------------------------------------------------

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

            "success":
                False,

            "cost":
                1e6,

            "failure":
                str(
                    exc
                ),

            "waypoints_reached":
                0,

            "trajectory_rmse_m":
                100.0,

            "cross_track_rmse_m":
                100.0,

            "cross_track_max_m":
                100.0,

            "final_error_m":
                100.0,

            "lateral_velocity_rmse_mps":
                100.0,

            "path_length_excess_m":
                100.0,

            "roll_rmse_deg":
                100.0,

            "pitch_rmse_deg":
                100.0,

            "torque_saturation_pct":
                100.0,

            "thrust_saturation_pct":
                100.0,
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

            "success":
                False,

            "cost":
                1e6,

            "failure":
                "nonfinite_state",

            "waypoints_reached":
                0,

            "trajectory_rmse_m":
                100.0,

            "cross_track_rmse_m":
                100.0,

            "cross_track_max_m":
                100.0,

            "final_error_m":
                100.0,

            "lateral_velocity_rmse_mps":
                100.0,

            "path_length_excess_m":
                100.0,

            "roll_rmse_deg":
                100.0,

            "pitch_rmse_deg":
                100.0,

            "torque_saturation_pct":
                100.0,

            "thrust_saturation_pct":
                100.0,
        }


    # ========================================================
    # TRAJECTORY METRICS
    # ========================================================

    trajectory = trajectory_metrics(

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


    # ========================================================
    # ATTITUDE TRACKING
    # ========================================================

    attitude = attitude_tracking_metrics(

        logs["state"][:, 6:9],

        logs["att_ref"],
    )


    # ========================================================
    # CROSS TRACK
    # ========================================================

    (
        cross_track_rmse,
        cross_track_max,
        cross_track_errors,
    ) = calculate_cross_track_metrics(

        logs["state"][:, 0:3]
    )


    # ========================================================
    # LATERAL MOTION / SERPENTINE METRIC
    # ========================================================

    lateral_velocity_rmse = (
        calculate_lateral_velocity_rmse(

            logs["state"][:, 3:6],

            logs["vel_ref"],
        )
    )


    # ========================================================
    # PATH-LENGTH EXCESS
    # ========================================================

    (
        path_length_excess,
        actual_path_length,
        reference_path_length,
    ) = calculate_path_length_excess(

        logs["state"][:, 0:3]
    )


    # ========================================================
    # ACTUATOR SATURATION
    # ========================================================

    thrust = (
        logs["control"][:, 0]
    )


    torque = (
        logs["control"][:, 1:4]
    )


    thrust_saturation_pct = float(

        100.0

        * np.mean(

            np.isclose(
                thrust,
                controller.limits.max_thrust,
                atol=1e-9,
            )
        )
    )


    torque_saturation_pct = float(

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
    # SUCCESS
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


    success = bool(

        mission_complete

        and waypoints_reached
        == len(
            WAYPOINTS
        )
    )


    # ========================================================
    # ROLL / PITCH PENALTY
    # ========================================================

    roll_rmse = float(
        attitude[
            "roll_tracking_rmse_deg"
        ]
    )


    pitch_rmse = float(
        attitude[
            "pitch_tracking_rmse_deg"
        ]
    )


    roll_pitch_rmse = float(

        np.sqrt(

            (
                roll_rmse**2
                + pitch_rmse**2
            )

            / 2.0
        )
    )


    # ========================================================
    # OBJECTIVE FUNCTION
    # ========================================================

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


    cost = (

        W_TRAJECTORY_RMSE
        * trajectory_rmse

        + W_CROSS_TRACK_RMSE
        * cross_track_rmse

        + W_CROSS_TRACK_MAX
        * cross_track_max

        + W_FINAL_ERROR
        * final_error

        + W_LATERAL_VELOCITY
        * lateral_velocity_rmse

        + W_PATH_LENGTH_EXCESS
        * path_length_excess

        + W_ROLL_PITCH_TRACKING
        * roll_pitch_rmse

        + W_TORQUE_SAT
        * torque_saturation_pct

        + W_THRUST_SAT
        * thrust_saturation_pct
    )


    # Mission completion has absolute priority.

    if not success:

        missing_waypoints = (
            len(
                WAYPOINTS
            )
            - waypoints_reached
        )


        cost += (
            100.0

            + 25.0
            * missing_waypoints
        )


    return {

        "success":
            success,

        "cost":
            float(
                cost
            ),

        "waypoints_reached":
            waypoints_reached,

        "trajectory_rmse_m":
            trajectory_rmse,

        "cross_track_rmse_m":
            cross_track_rmse,

        "cross_track_max_m":
            cross_track_max,

        "final_error_m":
            final_error,

        "lateral_velocity_rmse_mps":
            lateral_velocity_rmse,

        "path_length_excess_m":
            path_length_excess,

        "actual_path_length_m":
            actual_path_length,

        "reference_path_length_m":
            reference_path_length,

        "roll_rmse_deg":
            roll_rmse,

        "pitch_rmse_deg":
            pitch_rmse,

        "torque_saturation_pct":
            torque_saturation_pct,

        "thrust_saturation_pct":
            thrust_saturation_pct,
    }


# ============================================================
# CANDIDATE COMPARISON
# ============================================================

def candidate_is_better(
    candidate,
    current,
):

    # Mission completion first.

    if (
        candidate["success"]
        and not current["success"]
    ):

        return True


    if (
        current["success"]
        and not candidate["success"]
    ):

        return False


    # Then objective.

    return (
        candidate["cost"]
        <
        current["cost"]
    )


# ============================================================
# RESULT PRINTING
# ============================================================

def print_result(
    gains,
    result,
):

    print(
        f"Kp = "
        f"{np.round(gains[0:3], 4)}"
    )


    print(
        f"Kv = "
        f"{np.round(gains[3:6], 4)}"
    )


    print(
        f"Ki = "
        f"{np.round(gains[6:9], 4)}"
    )


    print()


    print(
        f"Waypoints:               "
        f"{result['waypoints_reached']} / 5"
    )


    print(
        f"Trajectory RMSE:         "
        f"{result['trajectory_rmse_m']:.4f} m"
    )


    print(
        f"Cross-track RMSE:        "
        f"{result['cross_track_rmse_m']:.4f} m"
    )


    print(
        f"Cross-track max:         "
        f"{result['cross_track_max_m']:.4f} m"
    )


    print(
        f"Final error:             "
        f"{result['final_error_m']:.4f} m"
    )


    print(
        f"Lateral velocity RMSE:   "
        f"{result['lateral_velocity_rmse_mps']:.4f} m/s"
    )


    print(
        f"Path-length excess:      "
        f"{result['path_length_excess_m']:.4f} m"
    )


    print(
        f"Roll RMSE:               "
        f"{result['roll_rmse_deg']:.4f} deg"
    )


    print(
        f"Pitch RMSE:              "
        f"{result['pitch_rmse_deg']:.4f} deg"
    )


    print(
        f"Torque saturation:       "
        f"{result['torque_saturation_pct']:.4f} %"
    )


    print(
        f"Objective:               "
        f"{result['cost']:.6f}"
    )


# ============================================================
# OPTIMIZER
# ============================================================

def optimize():

    current_gains = (
        INITIAL_GAINS.copy()
    )


    step = (
        INITIAL_STEP.copy()
    )


    # ========================================================
    # BASELINE
    # ========================================================

    current_result = evaluate_candidate(
        current_gains
    )


    print()

    print(
        "=" * 88
    )


    print(
        "EKF-IN-THE-LOOP PID TRAJECTORY OPTIMIZER"
    )


    print(
        "=" * 88
    )


    print(
        "EKF covariance is frozen."
    )


    print(
        "Sensor noise is frozen."
    )


    print(
        "No Monte Carlo."
    )


    print(
        "PID is optimized using estimated-state feedback."
    )


    print()

    print(
        "INITIAL EKF-CLOSED-LOOP RESULT"
    )


    print(
        "-" * 88
    )


    print_result(
        current_gains,
        current_result,
    )


    history = []


    history.append(
        {
            "iteration":
                0,

            "gains":
                current_gains.tolist(),

            **current_result,
        }
    )


    # ========================================================
    # COORDINATE SEARCH
    # ========================================================

    for sweep in range(
        1,
        MAX_SWEEPS + 1,
    ):

        print()

        print(
            "=" * 88
        )


        print(
            f"SWEEP "
            f"{sweep}/{MAX_SWEEPS}"
        )


        print(
            "=" * 88
        )


        improved = False


        for i, name in enumerate(
            PARAMETER_NAMES
        ):

            base = float(
                current_gains[i]
            )


            best_gains = (
                current_gains.copy()
            )


            best_result = (
                current_result.copy()
            )


            print()

            print(
                f"Testing "
                f"{name} around "
                f"{base:.4f}"
            )


            for direction in (
                -1.0,
                +1.0,
            ):

                candidate = (
                    current_gains.copy()
                )


                candidate[i] = np.clip(

                    base
                    + direction
                    * step[i],

                    LOWER_BOUNDS[i],

                    UPPER_BOUNDS[i],
                )


                if np.isclose(
                    candidate[i],
                    base,
                ):

                    continue


                result = evaluate_candidate(
                    candidate
                )


                status = (
                    "PASS"
                    if result["success"]
                    else "FAIL"
                )


                print(

                    f"  "
                    f"{name}"
                    f" = "
                    f"{candidate[i]:.4f}"

                    f" | {status}"

                    f" | traj "
                    f"{result['trajectory_rmse_m']:.4f}"

                    f" | cross "
                    f"{result['cross_track_rmse_m']:.4f}"

                    f" | max "
                    f"{result['cross_track_max_m']:.4f}"

                    f" | lat-v "
                    f"{result['lateral_velocity_rmse_mps']:.4f}"

                    f" | excess "
                    f"{result['path_length_excess_m']:.4f}"

                    f" | J "
                    f"{result['cost']:.5f}"
                )


                if candidate_is_better(
                    result,
                    best_result,
                ):

                    best_gains = (
                        candidate.copy()
                    )


                    best_result = (
                        result.copy()
                    )


            # =================================================
            # ACCEPT / REJECT
            # =================================================

            if candidate_is_better(
                best_result,
                current_result,
            ):

                current_gains = (
                    best_gains
                )


                current_result = (
                    best_result
                )


                improved = True


                print(

                    f"  ACCEPTED "
                    f"{name}"
                    f" = "
                    f"{current_gains[i]:.4f}"
                )


                history.append(
                    {
                        "iteration":
                            len(
                                history
                            ),

                        "sweep":
                            sweep,

                        "parameter":
                            name,

                        "gains":
                            current_gains.tolist(),

                        **current_result,
                    }
                )


            else:

                print(
                    f"  REJECTED changes "
                    f"to {name}"
                )


        # ====================================================
        # REDUCE STEP IF STALLED
        # ====================================================

        if not improved:

            step *= 0.5


            print()

            print(
                "No improvement in full sweep."
            )


            print(
                "Reduced search steps:"
            )


            print(
                step
            )


        # Stop when every parameter has reached
        # its desired search resolution.

        if np.all(
            step <= MIN_STEP
        ):

            print()

            print(
                "Minimum PID search resolution reached."
            )

            break


    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    final_result = evaluate_candidate(
        current_gains
    )


    print()

    print(
        "=" * 88
    )


    print(
        "EKF-CLOSED-LOOP PID OPTIMIZATION COMPLETE"
    )


    print(
        "=" * 88
    )


    print_result(
        current_gains,
        final_result,
    )


    # ========================================================
    # SAVE GAINS
    # ========================================================

    output = {

        "tuning_method":
            "ekf_in_the_loop_pid_coordinate_search",

        "sensor_seed":
            SENSOR_SEED,

        "ekf_covariance_file":
            "optimized_ekf_covariance.json",

        "kp_pos":
            current_gains[
                0:3
            ].tolist(),

        "kd_pos":
            current_gains[
                3:6
            ].tolist(),

        "ki_pos":
            current_gains[
                6:9
            ].tolist(),

        "result":
            final_result,
    }


    OUTPUT_GAIN_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    OUTPUT_GAIN_FILE.write_text(

        json.dumps(
            output,
            indent=2,
        )
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    HISTORY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    HISTORY_FILE.write_text(

        json.dumps(
            history,
            indent=2,
        )
    )


    print()

    print(
        "Saved EKF-closed-loop PID gains:"
    )


    print(
        OUTPUT_GAIN_FILE
    )


    print()

    print(
        "Saved optimization history:"
    )


    print(
        HISTORY_FILE
    )


    print(
        "=" * 88
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    optimize()