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
)


# ============================================================
# SIMULATION SETTINGS
# ============================================================

DT = 0.01

GPS_HZ = 10.0

DURATION = 40.0

SENSOR_SEED = 24


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
# LOAD FIXED PID GAINS
# ============================================================

PID_FILE = (
    ROOT
    / "config"
    / "nominal_pid_gains.json"
)


if not PID_FILE.exists():

    raise FileNotFoundError(
        "Could not find "
        "config/nominal_pid_gains.json"
    )


pid_data = json.loads(
    PID_FILE.read_text()
)


KP_POS = np.asarray(
    pid_data["kp_pos"],
    dtype=float,
)


KD_POS = np.asarray(
    pid_data["kd_pos"],
    dtype=float,
)


KI_POS = np.asarray(
    pid_data["ki_pos"],
    dtype=float,
)


# ============================================================
# ACTUAL SENSOR MODEL
# ============================================================

sensor_cfg = SensorConfig()


# ============================================================
# EKF TUNING PARAMETERS
# ============================================================

# These parameters do NOT change the actual simulated sensors.
#
# They only change what the EKF assumes about:
#
# P0 = initial covariance
# Q  = process covariance
# R  = GPS measurement covariance
#
#
# parameter vector:
#
# [
#   P0 scale,
#   accelerometer Q scale,
#   gyro Q scale,
#   accel-bias Q scale,
#   gyro-bias Q scale,
#   GPS-position R scale,
#   GPS-velocity R scale
# ]

PARAMETER_NAMES = [
    "P0",
    "Q_accel",
    "Q_gyro",
    "Q_accel_bias",
    "Q_gyro_bias",
    "R_gps_pos",
    "R_gps_vel",
]


# ============================================================
# INITIAL VALUES
# ============================================================

INITIAL_PARAMETERS = np.array(
    [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ],
    dtype=float,
)


# ============================================================
# SEARCH BOUNDS
# ============================================================

LOWER_BOUNDS = np.array(
    [
        0.10,
        0.10,
        0.10,
        0.10,
        0.10,
        0.20,
        0.20,
    ],
    dtype=float,
)


UPPER_BOUNDS = np.array(
    [
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        5.0,
        5.0,
    ],
    dtype=float,
)


# ============================================================
# MULTIPLICATIVE SEARCH FACTOR
# ============================================================

INITIAL_FACTOR = 2.0

MINIMUM_FACTOR = 1.05

MAX_SWEEPS = 8


# ============================================================
# OBJECTIVE WEIGHTS
# ============================================================

# Estimator accuracy receives highest priority.
#
# Tracking metrics are included because the EKF is being used
# directly in closed-loop control.

W_EKF_POSITION = 5.0

W_EKF_VELOCITY = 3.0

W_EKF_ATTITUDE = 0.20

W_TRAJECTORY = 2.0

W_CROSS_TRACK = 3.0

W_FINAL_ERROR = 1.0


# ============================================================
# CROSS-TRACK HELPER
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
# BUILD EKF FROM SCALE PARAMETERS
# ============================================================

def make_ekf(
    parameters,
):

    (
        p0_scale,
        q_accel_scale,
        q_gyro_scale,
        q_accel_bias_scale,
        q_gyro_bias_scale,
        r_pos_scale,
        r_vel_scale,
    ) = parameters


    # ========================================================
    # IMPORTANT
    # ========================================================
    #
    # Standard deviations are scaled by sqrt(scale)
    # because covariance is variance:
    #
    #     variance = sigma^2
    #
    # Therefore:
    #
    #     R_new = scale * R
    #
    # requires
    #
    #     sigma_new = sqrt(scale) * sigma
    #


    ekf = ErrorStateEKF15(

        dt=DT,

        gps_pos_std=(
            sensor_cfg.gps_pos_std
            * np.sqrt(
                r_pos_scale
            )
        ),

        gps_vel_std=(
            sensor_cfg.gps_vel_std
            * np.sqrt(
                r_vel_scale
            )
        ),

        accel_noise_std=(
            sensor_cfg.accel_noise_std
            * np.sqrt(
                q_accel_scale
            )
        ),

        gyro_noise_std=(
            sensor_cfg.gyro_noise_std
            * np.sqrt(
                q_gyro_scale
            )
        ),

        accel_bias_rw_std=(
            sensor_cfg.accel_bias_rw_std
            * np.sqrt(
                q_accel_bias_scale
            )
        ),

        gyro_bias_rw_std=(
            sensor_cfg.gyro_bias_rw_std
            * np.sqrt(
                q_gyro_bias_scale
            )
        ),
    )


    # Scale initial covariance P0.

    ekf.P *= (
        p0_scale
    )


    return ekf


# ============================================================
# RUN ONE CANDIDATE
# ============================================================

def evaluate_candidate(
    parameters,
):

    parameters = np.asarray(
        parameters,
        dtype=float,
    )


    # ========================================================
    # VEHICLE
    # ========================================================

    vehicle = QuadrotorModel()


    # ========================================================
    # FIXED PID CONTROLLER
    # ========================================================

    controller = CascadedController(
        vehicle.params.mass,
        vehicle.params.gravity,
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


    # ========================================================
    # GUIDANCE
    # ========================================================

    guidance = WaypointGuidance(
        WAYPOINTS,
        acceptance_radius=0.25,
        cruise_speed=0.60,
    )


    # ========================================================
    # ACTUAL SENSOR MODEL
    # ========================================================

    sensors = ImuGpsSensorSuite(
        config=SensorConfig(),
        seed=SENSOR_SEED,
    )


    # ========================================================
    # CANDIDATE EKF
    # ========================================================

    ekf = make_ekf(
        parameters
    )


    # ========================================================
    # RUN
    # ========================================================

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
            "cost":
                1e6,

            "success":
                False,

            "failure":
                str(
                    exc
                ),

            "trajectory_rmse_m":
                100.0,

            "cross_track_rmse_m":
                100.0,

            "final_error_m":
                100.0,

            "ekf_position_rmse_m":
                100.0,

            "ekf_velocity_rmse_mps":
                100.0,

            "ekf_roll_rmse_deg":
                100.0,

            "ekf_pitch_rmse_deg":
                100.0,

            "ekf_yaw_rmse_deg":
                100.0,
        }


    # ========================================================
    # CHECK NUMERICS
    # ========================================================

    if not np.all(
        np.isfinite(
            logs[
                "estimate"
            ]
        )
    ):

        return {
            "cost":
                1e6,

            "success":
                False,

            "failure":
                "nonfinite_estimate",

            "trajectory_rmse_m":
                100.0,

            "cross_track_rmse_m":
                100.0,

            "final_error_m":
                100.0,

            "ekf_position_rmse_m":
                100.0,

            "ekf_velocity_rmse_mps":
                100.0,

            "ekf_roll_rmse_deg":
                100.0,

            "ekf_pitch_rmse_deg":
                100.0,

            "ekf_yaw_rmse_deg":
                100.0,
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
            WAYPOINTS[
                -1
            ],

        waypoints_reached=
            int(
                logs[
                    "waypoints_reached"
                ][
                    -1
                ]
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
    # CROSS-TRACK ERROR
    # ========================================================

    path_points = np.vstack(
        [
            logs[
                "state"
            ][
                0,
                0:3
            ],

            WAYPOINTS,
        ]
    )


    cross_errors = []


    for position in logs[
        "state"
    ][:, 0:3]:

        cross_errors.append(

            min(

                point_segment_distance(
                    position,
                    path_points[
                        j
                    ],
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


    cross_errors = np.asarray(
        cross_errors,
        dtype=float,
    )


    cross_track_rmse = float(

        np.sqrt(

            np.mean(
                cross_errors**2
            )
        )
    )


    # ========================================================
    # MISSION SUCCESS
    # ========================================================

    waypoints_reached = int(
        logs[
            "waypoints_reached"
        ][
            -1
        ]
    )


    mission_complete = bool(
        logs[
            "mission_complete"
        ][
            -1
        ]
    )


    success = bool(
        mission_complete
        and waypoints_reached
        == len(
            WAYPOINTS
        )
    )


    # ========================================================
    # COMBINED ATTITUDE ESTIMATION RMSE
    # ========================================================

    attitude_rmse = float(

        np.sqrt(

            (
                estimation[
                    "ekf_roll_rmse_deg"
                ] ** 2

                + estimation[
                    "ekf_pitch_rmse_deg"
                ] ** 2

                + estimation[
                    "ekf_yaw_rmse_deg"
                ] ** 2
            )

            / 3.0
        )
    )


    # ========================================================
    # OBJECTIVE
    # ========================================================

    cost = (

        W_EKF_POSITION
        * estimation[
            "ekf_position_rmse_m"
        ]

        + W_EKF_VELOCITY
        * estimation[
            "ekf_velocity_rmse_mps"
        ]

        + W_EKF_ATTITUDE
        * attitude_rmse

        + W_TRAJECTORY
        * trajectory[
            "trajectory_position_rmse_m"
        ]

        + W_CROSS_TRACK
        * cross_track_rmse

        + W_FINAL_ERROR
        * trajectory[
            "final_error_m"
        ]
    )


    # Mission completion remains mandatory.

    if not success:

        cost += 100.0


    return {

        "cost":
            float(
                cost
            ),

        "success":
            success,

        "waypoints_reached":
            waypoints_reached,

        "trajectory_rmse_m":
            float(
                trajectory[
                    "trajectory_position_rmse_m"
                ]
            ),

        "cross_track_rmse_m":
            cross_track_rmse,

        "final_error_m":
            float(
                trajectory[
                    "final_error_m"
                ]
            ),

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

        "ekf_roll_rmse_deg":
            float(
                estimation[
                    "ekf_roll_rmse_deg"
                ]
            ),

        "ekf_pitch_rmse_deg":
            float(
                estimation[
                    "ekf_pitch_rmse_deg"
                ]
            ),

        "ekf_yaw_rmse_deg":
            float(
                estimation[
                    "ekf_yaw_rmse_deg"
                ]
            ),
    }


# ============================================================
# BETTER-CANDIDATE TEST
# ============================================================

def candidate_is_better(
    candidate,
    current,
):

    if (
        candidate[
            "success"
        ]
        and not current[
            "success"
        ]
    ):

        return True


    if (
        current[
            "success"
        ]
        and not candidate[
            "success"
        ]
    ):

        return False


    return (
        candidate[
            "cost"
        ]
        <
        current[
            "cost"
        ]
    )


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    parameters,
    result,
):

    print(
        f"P0 scale:           "
        f"{parameters[0]:.4f}"
    )


    print(
        f"Q accel scale:      "
        f"{parameters[1]:.4f}"
    )


    print(
        f"Q gyro scale:       "
        f"{parameters[2]:.4f}"
    )


    print(
        f"Q accel-bias scale: "
        f"{parameters[3]:.4f}"
    )


    print(
        f"Q gyro-bias scale:  "
        f"{parameters[4]:.4f}"
    )


    print(
        f"R GPS-pos scale:    "
        f"{parameters[5]:.4f}"
    )


    print(
        f"R GPS-vel scale:    "
        f"{parameters[6]:.4f}"
    )


    print()


    print(
        f"Waypoints:          "
        f"{result['waypoints_reached']} / 5"
    )


    print(
        f"Trajectory RMSE:    "
        f"{result['trajectory_rmse_m']:.4f} m"
    )


    print(
        f"Cross-track RMSE:   "
        f"{result['cross_track_rmse_m']:.4f} m"
    )


    print(
        f"Final error:        "
        f"{result['final_error_m']:.4f} m"
    )


    print(
        f"EKF position RMSE:  "
        f"{result['ekf_position_rmse_m']:.4f} m"
    )


    print(
        f"EKF velocity RMSE:  "
        f"{result['ekf_velocity_rmse_mps']:.4f} m/s"
    )


    print(
        f"EKF roll RMSE:      "
        f"{result['ekf_roll_rmse_deg']:.4f} deg"
    )


    print(
        f"EKF pitch RMSE:     "
        f"{result['ekf_pitch_rmse_deg']:.4f} deg"
    )


    print(
        f"EKF yaw RMSE:       "
        f"{result['ekf_yaw_rmse_deg']:.4f} deg"
    )


    print(
        f"Objective cost:     "
        f"{result['cost']:.6f}"
    )


# ============================================================
# OPTIMIZER
# ============================================================

def optimize():

    current = (
        INITIAL_PARAMETERS.copy()
    )


    factor = float(
        INITIAL_FACTOR
    )


    current_result = evaluate_candidate(
        current
    )


    print()

    print(
        "=" * 84
    )


    print(
        "EKF COVARIANCE OPTIMIZER"
    )


    print(
        "=" * 84
    )


    print(
        "PID GAINS ARE FROZEN"
    )


    print(
        f"Kp = {KP_POS}"
    )


    print(
        f"Kv = {KD_POS}"
    )


    print(
        f"Ki = {KI_POS}"
    )


    print()

    print(
        "INITIAL EKF RESULT"
    )


    print(
        "-" * 84
    )


    print_result(
        current,
        current_result,
    )


    history = []


    history.append(
        {
            "iteration":
                0,

            "parameters":
                current.tolist(),

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
            "=" * 84
        )


        print(
            f"SWEEP "
            f"{sweep}/{MAX_SWEEPS}"
            f"   "
            f"factor = {factor:.4f}"
        )


        print(
            "=" * 84
        )


        improved = False


        for i, name in enumerate(
            PARAMETER_NAMES
        ):

            base = (
                current[
                    i
                ]
            )


            best_parameters = (
                current.copy()
            )


            best_result = (
                current_result.copy()
            )


            print()

            print(
                f"Testing {name} "
                f"around {base:.4f}"
            )


            # =================================================
            # TRY LOWER
            # =================================================

            candidate_low = (
                current.copy()
            )


            candidate_low[
                i
            ] = np.clip(

                base
                / factor,

                LOWER_BOUNDS[
                    i
                ],

                UPPER_BOUNDS[
                    i
                ],
            )


            # =================================================
            # TRY HIGHER
            # =================================================

            candidate_high = (
                current.copy()
            )


            candidate_high[
                i
            ] = np.clip(

                base
                * factor,

                LOWER_BOUNDS[
                    i
                ],

                UPPER_BOUNDS[
                    i
                ],
            )


            for candidate in [
                candidate_low,
                candidate_high,
            ]:

                if np.isclose(
                    candidate[
                        i
                    ],
                    base,
                ):

                    continue


                result = evaluate_candidate(
                    candidate
                )


                status = (
                    "PASS"
                    if result[
                        "success"
                    ]
                    else "FAIL"
                )


                print(

                    f"  {name}"
                    f" = "
                    f"{candidate[i]:.4f}"

                    f" | {status}"

                    f" | pos "
                    f"{result['ekf_position_rmse_m']:.4f}"

                    f" | vel "
                    f"{result['ekf_velocity_rmse_mps']:.4f}"

                    f" | traj "
                    f"{result['trajectory_rmse_m']:.4f}"

                    f" | cross "
                    f"{result['cross_track_rmse_m']:.4f}"

                    f" | J "
                    f"{result['cost']:.5f}"
                )


                if candidate_is_better(
                    result,
                    best_result,
                ):

                    best_parameters = (
                        candidate.copy()
                    )


                    best_result = (
                        result.copy()
                    )


            # =================================================
            # ACCEPT OR REJECT
            # =================================================

            if candidate_is_better(
                best_result,
                current_result,
            ):

                current = (
                    best_parameters
                )


                current_result = (
                    best_result
                )


                improved = True


                print(
                    f"  ACCEPTED "
                    f"{name} = "
                    f"{current[i]:.4f}"
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

                        "parameters":
                            current.tolist(),

                        **current_result,
                    }
                )


            else:

                print(
                    f"  REJECTED changes "
                    f"to {name}"
                )


        # ====================================================
        # REFINE SEARCH FACTOR
        # ====================================================

        if not improved:

            factor = (
                1.0
                + (
                    factor
                    - 1.0
                )
                * 0.5
            )


            print()

            print(
                "No improvement in full sweep."
            )


            print(
                f"New search factor = "
                f"{factor:.4f}"
            )


        if factor <= MINIMUM_FACTOR:

            print()

            print(
                "Minimum covariance search "
                "resolution reached."
            )

            break


    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    final_result = evaluate_candidate(
        current
    )


    print()

    print(
        "=" * 84
    )


    print(
        "EKF COVARIANCE OPTIMIZATION COMPLETE"
    )


    print(
        "=" * 84
    )


    print_result(
        current,
        final_result,
    )


    # ========================================================
    # SAVE CONFIGURATION
    # ========================================================

    output = {

        "tuning_method":
            "single_run_nominal_ekf_covariance_coordinate_search",

        "sensor_seed":
            SENSOR_SEED,

        "p0_scale":
            float(
                current[
                    0
                ]
            ),

        "q_accel_scale":
            float(
                current[
                    1
                ]
            ),

        "q_gyro_scale":
            float(
                current[
                    2
                ]
            ),

        "q_accel_bias_scale":
            float(
                current[
                    3
                ]
            ),

        "q_gyro_bias_scale":
            float(
                current[
                    4
                ]
            ),

        "r_gps_pos_scale":
            float(
                current[
                    5
                ]
            ),

        "r_gps_vel_scale":
            float(
                current[
                    6
                ]
            ),

        "result":
            final_result,
    }


    output_file = (
        ROOT
        / "config"
        / "optimized_ekf_covariance.json"
    )


    output_file.write_text(

        json.dumps(
            output,
            indent=2,
        )
    )


    history_file = (
        ROOT
        / "results"
        / "data"
        / "ekf_covariance_optimization_history.json"
    )


    history_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    history_file.write_text(

        json.dumps(
            history,
            indent=2,
        )
    )


    print()

    print(
        "Saved EKF configuration:"
    )


    print(
        output_file
    )


    print()

    print(
        "Saved optimization history:"
    )


    print(
        history_file
    )


    print(
        "=" * 84
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    optimize()