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
# SIMULATION SETTINGS
# ============================================================

DT = 0.01

GPS_HZ = 10.0

DURATION = 40.0

SENSOR_SEED = 24


# ============================================================
# FIXED SENSOR NOISE
# ============================================================

POSITION_NOISE_M = 0.02


# ============================================================
# TRAJECTORY TARGET
# ============================================================
#
# This is an optimization aspiration, NOT a guaranteed result.
#
# With 2 cm position measurement noise, achieving 5 mm
# trajectory RMSE may not be physically supported by the
# estimator/controller architecture.
# ============================================================

TARGET_TRAJECTORY_RMSE_M = 0.005

TARGET_CROSS_TRACK_RMSE_M = 0.005


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
# INPUT / OUTPUT FILES
# ============================================================

NOMINAL_PID_FILE = (
    ROOT
    / "config"
    / "nominal_pid_gains.json"
)


OUTPUT_FILE = (
    ROOT
    / "config"
    / "optimized_2cm_full_gnc.json"
)


HISTORY_FILE = (
    ROOT
    / "results"
    / "data"
    / "optimized_2cm_full_gnc_history.json"
)


# ============================================================
# LOAD OUTER-LOOP STARTING GAINS
# ============================================================

if not NOMINAL_PID_FILE.exists():

    raise FileNotFoundError(
        "Could not find "
        "config/nominal_pid_gains.json"
    )


nominal_data = json.loads(
    NOMINAL_PID_FILE.read_text()
)


START_KP_POS = np.asarray(
    nominal_data["kp_pos"],
    dtype=float,
)


START_KD_POS = np.asarray(
    nominal_data["kd_pos"],
    dtype=float,
)


START_KI_POS = np.asarray(
    nominal_data["ki_pos"],
    dtype=float,
)


# ============================================================
# HELPER
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
        a
        + q * ab
    )


    return float(
        np.linalg.norm(
            p - projection
        )
    )


# ============================================================
# CREATE SENSOR + EKF
# ============================================================

def create_navigation():

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
        sensor_cfg,
        sensors,
        ekf,
    )


# ============================================================
# APPLY GAIN VECTOR
# ============================================================
#
# Gain vector:
#
# 0  = attitude Kp XY
# 1  = rate Kp XY
# 2  = rate Ki XY
# 3  = rate Kd XY
#
# 4  = position Kp XY
# 5  = velocity Kd XY
# 6  = position Ki XY
#
# Z position gains remain independently tunable:
#
# 7  = Kp Z
# 8  = Kd Z
# 9  = Ki Z
#
# Yaw inner-loop gains are intentionally left unchanged.
# ============================================================

def apply_gains(
    controller,
    gains,
):

    gains = np.asarray(
        gains,
        dtype=float,
    )


    # --------------------------------------------------------
    # REQUIRED INNER LOOP ATTRIBUTES
    # --------------------------------------------------------

    required = [
        "kp_att",
        "kp_rate",
        "ki_rate",
        "kd_rate",
        "kp_pos",
        "kd_pos",
    ]


    for name in required:

        if not hasattr(
            controller,
            name,
        ):

            raise AttributeError(
                f"CascadedController is missing "
                f"required attribute '{name}'."
            )


    # --------------------------------------------------------
    # INNER ATTITUDE LOOP
    # --------------------------------------------------------

    controller.kp_att[
        0
    ] = gains[0]

    controller.kp_att[
        1
    ] = gains[0]


    # --------------------------------------------------------
    # INNER RATE PID
    # --------------------------------------------------------

    controller.kp_rate[
        0
    ] = gains[1]

    controller.kp_rate[
        1
    ] = gains[1]


    controller.ki_rate[
        0
    ] = gains[2]

    controller.ki_rate[
        1
    ] = gains[2]


    controller.kd_rate[
        0
    ] = gains[3]

    controller.kd_rate[
        1
    ] = gains[3]


    # --------------------------------------------------------
    # OUTER POSITION LOOP
    # --------------------------------------------------------

    controller.kp_pos = np.array(
        [
            gains[4],
            gains[4],
            gains[7],
        ],
        dtype=float,
    )


    controller.kd_pos = np.array(
        [
            gains[5],
            gains[5],
            gains[8],
        ],
        dtype=float,
    )


    # --------------------------------------------------------
    # POSITION INTEGRAL LOOP
    # --------------------------------------------------------

    if hasattr(
        controller,
        "ki_pos",
    ):

        controller.ki_pos = np.array(
            [
                gains[6],
                gains[6],
                gains[9],
            ],
            dtype=float,
        )


    return controller


# ============================================================
# INITIAL GAIN VECTOR
# ============================================================

probe_vehicle = QuadrotorModel()

probe_controller = CascadedController(
    probe_vehicle.params.mass,
    probe_vehicle.params.gravity,
)


INITIAL_GAINS = np.array(
    [
        # attitude Kp XY
        float(
            np.mean(
                probe_controller.kp_att[
                    0:2
                ]
            )
        ),

        # rate Kp XY
        float(
            np.mean(
                probe_controller.kp_rate[
                    0:2
                ]
            )
        ),

        # rate Ki XY
        float(
            np.mean(
                probe_controller.ki_rate[
                    0:2
                ]
            )
        ),

        # rate Kd XY
        float(
            np.mean(
                probe_controller.kd_rate[
                    0:2
                ]
            )
        ),

        # position Kp XY
        float(
            np.mean(
                START_KP_POS[
                    0:2
                ]
            )
        ),

        # velocity Kd XY
        float(
            np.mean(
                START_KD_POS[
                    0:2
                ]
            )
        ),

        # position Ki XY
        float(
            np.mean(
                START_KI_POS[
                    0:2
                ]
            )
        ),

        # position Kp Z
        float(
            START_KP_POS[
                2
            ]
        ),

        # velocity Kd Z
        float(
            START_KD_POS[
                2
            ]
        ),

        # position Ki Z
        float(
            START_KI_POS[
                2
            ]
        ),
    ],
    dtype=float,
)


# ============================================================
# BOUNDS
# ============================================================

LOWER_BOUNDS = np.array(
    [
        1.0,     # attitude Kp XY
        0.05,    # rate Kp XY
        0.000,   # rate Ki XY
        0.001,   # rate Kd XY

        0.20,    # pos Kp XY
        0.50,    # pos Kd XY
        0.000,   # pos Ki XY

        0.50,    # pos Kp Z
        0.50,    # pos Kd Z
        0.000,   # pos Ki Z
    ],
    dtype=float,
)


UPPER_BOUNDS = np.array(
    [
        10.0,    # attitude Kp XY
        1.50,    # rate Kp XY
        0.150,   # rate Ki XY
        0.100,   # rate Kd XY

        8.00,    # pos Kp XY
        10.0,    # pos Kd XY
        1.00,    # pos Ki XY

        8.00,    # pos Kp Z
        10.0,    # pos Kd Z
        1.00,    # pos Ki Z
    ],
    dtype=float,
)


# ============================================================
# PARAMETER NAMES
# ============================================================

PARAMETER_NAMES = [
    "Katt_xy",
    "Krate_p_xy",
    "Krate_i_xy",
    "Krate_d_xy",
    "Kpos_p_xy",
    "Kpos_d_xy",
    "Kpos_i_xy",
    "Kpos_p_z",
    "Kpos_d_z",
    "Kpos_i_z",
]


# ============================================================
# INITIAL SEARCH STEPS
# ============================================================

INITIAL_STEPS = np.array(
    [
        0.50,
        0.05,
        0.005,
        0.003,

        0.25,
        0.25,
        0.05,

        0.25,
        0.25,
        0.05,
    ],
    dtype=float,
)


MINIMUM_STEPS = np.array(
    [
        0.0625,
        0.00625,
        0.00125,
        0.00075,

        0.03125,
        0.03125,
        0.0125,

        0.03125,
        0.03125,
        0.0125,
    ],
    dtype=float,
)


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    logs,
    controller,
):

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
                ][-1]
            ),
    )


    estimation = estimation_metrics(
        logs[
            "state"
        ],
        logs[
            "estimate"
        ],
    )


    attitude = attitude_tracking_metrics(
        logs[
            "state"
        ][:, 6:9],
        logs[
            "att_ref"
        ],
    )


    # --------------------------------------------------------
    # CROSS-TRACK
    # --------------------------------------------------------

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
        )


    cross_errors = np.asarray(
        cross_errors,
        dtype=float,
    )


    cross_rmse = float(

        np.sqrt(

            np.mean(
                cross_errors**2
            )
        )
    )


    cross_max = float(
        np.max(
            cross_errors
        )
    )


    # --------------------------------------------------------
    # SATURATION
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # WAYPOINT SUCCESS
    # --------------------------------------------------------

    reached = int(
        logs[
            "waypoints_reached"
        ][-1]
    )


    complete = bool(
        logs[
            "mission_complete"
        ][-1]
    )


    success = bool(
        complete
        and reached
        == len(
            WAYPOINTS
        )
    )


    return {

        "success":
            success,

        "waypoints_reached":
            reached,

        "trajectory_rmse_m":
            float(
                trajectory[
                    "trajectory_position_rmse_m"
                ]
            ),

        "cross_track_rmse_m":
            cross_rmse,

        "cross_track_max_m":
            cross_max,

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

        "roll_rmse_deg":
            float(
                attitude[
                    "roll_tracking_rmse_deg"
                ]
            ),

        "pitch_rmse_deg":
            float(
                attitude[
                    "pitch_tracking_rmse_deg"
                ]
            ),

        "yaw_rmse_deg":
            float(
                attitude[
                    "yaw_tracking_rmse_deg"
                ]
            ),

        "thrust_saturation_pct":
            thrust_sat,

        "torque_saturation_pct":
            torque_sat,
    }


# ============================================================
# OBJECTIVE FUNCTIONS
# ============================================================

def inner_loop_cost(
    metrics,
):

    if not metrics[
        "success"
    ]:

        return (
            1000.0
        )


    roll_pitch = np.sqrt(

        (
            metrics[
                "roll_rmse_deg"
            ] ** 2

            + metrics[
                "pitch_rmse_deg"
            ] ** 2
        )

        / 2.0
    )


    return float(

        2.0
        * roll_pitch

        + 2.0
        * metrics[
            "trajectory_rmse_m"
        ]

        + 4.0
        * metrics[
            "cross_track_rmse_m"
        ]

        + 0.25
        * metrics[
            "torque_saturation_pct"
        ]

        + 0.25
        * metrics[
            "thrust_saturation_pct"
        ]
    )


def outer_loop_cost(
    metrics,
):

    if not metrics[
        "success"
    ]:

        return (
            1000.0
        )


    trajectory_penalty = (
        metrics[
            "trajectory_rmse_m"
        ]
    )


    cross_penalty = (
        metrics[
            "cross_track_rmse_m"
        ]
    )


    final_penalty = (
        metrics[
            "final_error_m"
        ]
    )


    max_penalty = (
        metrics[
            "cross_track_max_m"
        ]
    )


    # --------------------------------------------------------
    # EXTRA PENALTY ABOVE 5 mm TARGET
    # --------------------------------------------------------

    target_trajectory_excess = max(
        0.0,
        trajectory_penalty
        - TARGET_TRAJECTORY_RMSE_M,
    )


    target_cross_excess = max(
        0.0,
        cross_penalty
        - TARGET_CROSS_TRACK_RMSE_M,
    )


    return float(

        12.0
        * trajectory_penalty

        + 18.0
        * cross_penalty

        + 5.0
        * max_penalty

        + 5.0
        * final_penalty

        + 12.0
        * target_trajectory_excess

        + 15.0
        * target_cross_excess

        + 0.15
        * metrics[
            "roll_rmse_deg"
        ]

        + 0.15
        * metrics[
            "pitch_rmse_deg"
        ]

        + 0.25
        * metrics[
            "torque_saturation_pct"
        ]

        + 0.25
        * metrics[
            "thrust_saturation_pct"
        ]
    )


# ============================================================
# RUN ONE CANDIDATE
# ============================================================

def evaluate(
    gains,
    stage,
):

    vehicle = QuadrotorModel()


    controller = CascadedController(
        vehicle.params.mass,
        vehicle.params.gravity,
    )


    controller = apply_gains(
        controller,
        gains,
    )


    guidance = WaypointGuidance(
        WAYPOINTS,
        acceptance_radius=0.25,
        cruise_speed=0.60,
    )


    (
        sensor_cfg,
        sensors,
        ekf,
    ) = create_navigation()


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


        metrics = calculate_metrics(
            logs,
            controller,
        )


    except Exception as exc:

        return {

            "success":
                False,

            "cost":
                1e9,

            "failure":
                str(
                    exc
                ),

            "waypoints_reached":
                0,

            "trajectory_rmse_m":
                999.0,

            "cross_track_rmse_m":
                999.0,

            "cross_track_max_m":
                999.0,

            "final_error_m":
                999.0,

            "ekf_position_rmse_m":
                999.0,

            "ekf_velocity_rmse_mps":
                999.0,

            "roll_rmse_deg":
                999.0,

            "pitch_rmse_deg":
                999.0,

            "yaw_rmse_deg":
                999.0,

            "thrust_saturation_pct":
                100.0,

            "torque_saturation_pct":
                100.0,
        }


    if stage == "inner":

        cost = inner_loop_cost(
            metrics
        )


    elif stage == "outer":

        cost = outer_loop_cost(
            metrics
        )


    else:

        raise ValueError(
            "stage must be "
            "'inner' or 'outer'"
        )


    metrics[
        "cost"
    ] = float(
        cost
    )


    return metrics


# ============================================================
# PRINT CANDIDATE
# ============================================================

def print_candidate(
    name,
    value,
    result,
):

    print(

        f"{name:14s}"
        f" = {value:9.5f}"

        f" | traj "
        f"{result['trajectory_rmse_m']:.5f}"

        f" | cross "
        f"{result['cross_track_rmse_m']:.5f}"

        f" | final "
        f"{result['final_error_m']:.5f}"

        f" | roll "
        f"{result['roll_rmse_deg']:.3f}"

        f" | pitch "
        f"{result['pitch_rmse_deg']:.3f}"

        f" | J "
        f"{result['cost']:.6f}"
    )


# ============================================================
# COORDINATE OPTIMIZER
# ============================================================

def coordinate_optimize(
    start_gains,
    indices,
    stage,
    max_sweeps,
    steps,
):

    current = (
        start_gains.copy()
    )


    current_result = evaluate(
        current,
        stage,
    )


    history = []


    print()

    print(
        "=" * 94
    )


    print(
        f"{stage.upper()} LOOP OPTIMIZATION"
    )


    print(
        "=" * 94
    )


    print(
        f"Starting trajectory RMSE: "
        f"{current_result['trajectory_rmse_m']:.6f} m"
    )


    print(
        f"Starting cross-track RMSE: "
        f"{current_result['cross_track_rmse_m']:.6f} m"
    )


    print(
        f"Starting roll RMSE: "
        f"{current_result['roll_rmse_deg']:.4f} deg"
    )


    print(
        f"Starting pitch RMSE: "
        f"{current_result['pitch_rmse_deg']:.4f} deg"
    )


    for sweep in range(
        1,
        max_sweeps + 1,
    ):

        print()

        print(
            "-" * 94
        )


        print(
            f"{stage.upper()} SWEEP "
            f"{sweep}/{max_sweeps}"
        )


        print(
            "-" * 94
        )


        improved = False


        for index in indices:

            name = (
                PARAMETER_NAMES[
                    index
                ]
            )


            base = float(
                current[
                    index
                ]
            )


            best_gains = (
                current.copy()
            )


            best_result = (
                current_result.copy()
            )


            for direction in (
                -1.0,
                +1.0,
            ):

                candidate = (
                    current.copy()
                )


                candidate[
                    index
                ] = np.clip(

                    base

                    + direction
                    * steps[
                        index
                    ],

                    LOWER_BOUNDS[
                        index
                    ],

                    UPPER_BOUNDS[
                        index
                    ],
                )


                if np.isclose(
                    candidate[
                        index
                    ],
                    base,
                ):

                    continue


                result = evaluate(
                    candidate,
                    stage,
                )


                print_candidate(
                    name,
                    candidate[
                        index
                    ],
                    result,
                )


                if (
                    result[
                        "success"
                    ]
                    and result[
                        "cost"
                    ]
                    <
                    best_result[
                        "cost"
                    ]
                ):

                    best_gains = (
                        candidate.copy()
                    )


                    best_result = (
                        result.copy()
                    )


            if (
                best_result[
                    "cost"
                ]
                <
                current_result[
                    "cost"
                ]
            ):

                current = (
                    best_gains
                )


                current_result = (
                    best_result
                )


                improved = True


                print(
                    f"  ACCEPTED "
                    f"{name} = "
                    f"{current[index]:.6f}"
                )


                history.append(
                    {
                        "stage":
                            stage,

                        "sweep":
                            sweep,

                        "parameter":
                            name,

                        "gains":
                            current.tolist(),

                        **current_result,
                    }
                )


            else:

                print(
                    f"  REJECTED "
                    f"{name}"
                )


        # ----------------------------------------------------
        # REDUCE STEP IF NO IMPROVEMENT
        # ----------------------------------------------------

        if not improved:

            for index in indices:

                steps[
                    index
                ] *= (
                    0.5
                )


            print()

            print(
                "No improvement in this sweep."
            )


            print(
                "Reducing search step sizes."
            )


        # ----------------------------------------------------
        # STOP ON RESOLUTION
        # ----------------------------------------------------

        finished = True


        for index in indices:

            if (
                steps[
                    index
                ]
                >
                MINIMUM_STEPS[
                    index
                ]
            ):

                finished = False


        if finished:

            print()

            print(
                "Minimum search resolution reached."
            )

            break


        # ----------------------------------------------------
        # STOP IF TARGET IS ACTUALLY ACHIEVED
        # ----------------------------------------------------

        if (
            current_result[
                "trajectory_rmse_m"
            ]
            <= TARGET_TRAJECTORY_RMSE_M

            and

            current_result[
                "cross_track_rmse_m"
            ]
            <= TARGET_CROSS_TRACK_RMSE_M
        ):

            print()

            print(
                "5 mm optimization target reached."
            )

            break


    return (
        current,
        current_result,
        history,
        steps,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "=" * 94
    )


    print(
        "2 CM SENSOR — FULL CASCADED GNC OPTIMIZER"
    )


    print(
        "=" * 94
    )


    print(
        "Injected position sigma:"
    )


    print(
        f"    X = "
        f"{POSITION_NOISE_M:.3f} m"
    )


    print(
        f"    Y = "
        f"{POSITION_NOISE_M:.3f} m"
    )


    print(
        f"    Z = "
        f"{POSITION_NOISE_M:.3f} m"
    )


    print()


    print(
        "Requested aspiration:"
    )


    print(
        f"    trajectory RMSE <= "
        f"{TARGET_TRAJECTORY_RMSE_M:.3f} m"
    )


    print(
        f"    cross-track RMSE <= "
        f"{TARGET_CROSS_TRACK_RMSE_M:.3f} m"
    )


    print()


    print(
        "NOTE:"
    )


    print(
        "5 mm is a search target, not a guaranteed "
        "or assumed result."
    )


    print(
        "=" * 94
    )


    # ========================================================
    # BASELINE
    # ========================================================

    baseline = evaluate(
        INITIAL_GAINS,
        "outer",
    )


    print()

    print(
        "INITIAL FULL-GNC RESULT"
    )


    print(
        "-" * 94
    )


    print(
        f"Trajectory RMSE:       "
        f"{baseline['trajectory_rmse_m']:.6f} m"
    )


    print(
        f"Cross-track RMSE:      "
        f"{baseline['cross_track_rmse_m']:.6f} m"
    )


    print(
        f"Final error:           "
        f"{baseline['final_error_m']:.6f} m"
    )


    print(
        f"EKF position RMSE:     "
        f"{baseline['ekf_position_rmse_m']:.6f} m"
    )


    print(
        f"Roll tracking RMSE:    "
        f"{baseline['roll_rmse_deg']:.4f} deg"
    )


    print(
        f"Pitch tracking RMSE:   "
        f"{baseline['pitch_rmse_deg']:.4f} deg"
    )


    # ========================================================
    # STAGE 1:
    # INNER ATTITUDE / RATE LOOP
    # ========================================================

    steps = (
        INITIAL_STEPS.copy()
    )


    INNER_INDICES = [
        0,
        1,
        2,
        3,
    ]


    (
        inner_gains,
        inner_result,
        inner_history,
        steps,
    ) = coordinate_optimize(

        INITIAL_GAINS,

        INNER_INDICES,

        "inner",

        max_sweeps=5,

        steps=steps,
    )


    # ========================================================
    # STAGE 2:
    # OUTER POSITION LOOP
    # ========================================================

    OUTER_INDICES = [
        4,
        5,
        6,
        7,
        8,
        9,
    ]


    (
        final_gains,
        final_result,
        outer_history,
        steps,
    ) = coordinate_optimize(

        inner_gains,

        OUTER_INDICES,

        "outer",

        max_sweeps=7,

        steps=steps,
    )


    # ========================================================
    # FINAL VERIFICATION
    # ========================================================

    final_result = evaluate(
        final_gains,
        "outer",
    )


    print()

    print(
        "=" * 94
    )


    print(
        "FULL GNC OPTIMIZATION COMPLETE"
    )


    print(
        "=" * 94
    )


    print()

    print(
        "OPTIMIZED INNER LOOP"
    )


    print(
        "-" * 94
    )


    print(
        f"Attitude Kp XY:       "
        f"{final_gains[0]:.6f}"
    )


    print(
        f"Rate Kp XY:           "
        f"{final_gains[1]:.6f}"
    )


    print(
        f"Rate Ki XY:           "
        f"{final_gains[2]:.6f}"
    )


    print(
        f"Rate Kd XY:           "
        f"{final_gains[3]:.6f}"
    )


    print()

    print(
        "OPTIMIZED OUTER LOOP"
    )


    print(
        "-" * 94
    )


    print(
        f"Kp XY:                 "
        f"{final_gains[4]:.6f}"
    )


    print(
        f"Kd XY:                 "
        f"{final_gains[5]:.6f}"
    )


    print(
        f"Ki XY:                 "
        f"{final_gains[6]:.6f}"
    )


    print(
        f"Kp Z:                  "
        f"{final_gains[7]:.6f}"
    )


    print(
        f"Kd Z:                  "
        f"{final_gains[8]:.6f}"
    )


    print(
        f"Ki Z:                  "
        f"{final_gains[9]:.6f}"
    )


    print()

    print(
        "FINAL PERFORMANCE"
    )


    print(
        "-" * 94
    )


    print(
        f"Waypoints reached:     "
        f"{final_result['waypoints_reached']} / 5"
    )


    print(
        f"Trajectory RMSE:       "
        f"{final_result['trajectory_rmse_m']:.6f} m"
    )


    print(
        f"Cross-track RMSE:      "
        f"{final_result['cross_track_rmse_m']:.6f} m"
    )


    print(
        f"Max cross-track:       "
        f"{final_result['cross_track_max_m']:.6f} m"
    )


    print(
        f"Final waypoint error:  "
        f"{final_result['final_error_m']:.6f} m"
    )


    print(
        f"EKF position RMSE:     "
        f"{final_result['ekf_position_rmse_m']:.6f} m"
    )


    print(
        f"EKF velocity RMSE:     "
        f"{final_result['ekf_velocity_rmse_mps']:.6f} m/s"
    )


    print(
        f"Roll tracking RMSE:    "
        f"{final_result['roll_rmse_deg']:.4f} deg"
    )


    print(
        f"Pitch tracking RMSE:   "
        f"{final_result['pitch_rmse_deg']:.4f} deg"
    )


    print(
        f"Yaw tracking RMSE:     "
        f"{final_result['yaw_rmse_deg']:.4f} deg"
    )


    print(
        f"Thrust saturation:     "
        f"{final_result['thrust_saturation_pct']:.4f} %"
    )


    print(
        f"Torque saturation:     "
        f"{final_result['torque_saturation_pct']:.4f} %"
    )


    # ========================================================
    # SAVE CONFIG
    # ========================================================

    output = {

        "position_noise_std_m":
            POSITION_NOISE_M,

        "sensor_seed":
            SENSOR_SEED,

        "requested_target_trajectory_rmse_m":
            TARGET_TRAJECTORY_RMSE_M,

        "requested_target_cross_track_rmse_m":
            TARGET_CROSS_TRACK_RMSE_M,

        "kp_att":
            [
                float(
                    final_gains[0]
                ),

                float(
                    final_gains[0]
                ),

                float(
                    probe_controller.kp_att[
                        2
                    ]
                ),
            ],

        "kp_rate":
            [
                float(
                    final_gains[1]
                ),

                float(
                    final_gains[1]
                ),

                float(
                    probe_controller.kp_rate[
                        2
                    ]
                ),
            ],

        "ki_rate":
            [
                float(
                    final_gains[2]
                ),

                float(
                    final_gains[2]
                ),

                float(
                    probe_controller.ki_rate[
                        2
                    ]
                ),
            ],

        "kd_rate":
            [
                float(
                    final_gains[3]
                ),

                float(
                    final_gains[3]
                ),

                float(
                    probe_controller.kd_rate[
                        2
                    ]
                ),
            ],

        "kp_pos":
            [
                float(
                    final_gains[4]
                ),

                float(
                    final_gains[4]
                ),

                float(
                    final_gains[7]
                ),
            ],

        "kd_pos":
            [
                float(
                    final_gains[5]
                ),

                float(
                    final_gains[5]
                ),

                float(
                    final_gains[8]
                ),
            ],

        "ki_pos":
            [
                float(
                    final_gains[6]
                ),

                float(
                    final_gains[6]
                ),

                float(
                    final_gains[9]
                ),
            ],

        "final_result":
            final_result,
    }


    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    OUTPUT_FILE.write_text(

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


    history = (
        inner_history
        + outer_history
    )


    HISTORY_FILE.write_text(

        json.dumps(
            history,
            indent=2,
        )
    )


    print()

    print(
        "Saved optimized configuration:"
    )


    print(
        OUTPUT_FILE
    )


    print()

    print(
        "Saved optimization history:"
    )


    print(
        HISTORY_FILE
    )


    print(
        "=" * 94
    )


if __name__ == "__main__":

    main()