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
# ROBUST TRAINING SET
# ============================================================

# These are the fixed dispersed cases used DURING optimization.
#
# We intentionally keep these fixed so that every candidate
# controller sees exactly the same uncertainty realizations.
#
# The final 100-case validation uses a DIFFERENT random seed.

N_TRAINING_CASES = 10

TRAINING_MASTER_SEED = 314159


# ============================================================
# UNCERTAINTY MODEL
# ============================================================

# Physical plant mass variation:
# +/- 10 percent

MASS_SCALE_MIN = 0.90
MASS_SCALE_MAX = 1.10


# Principal inertia variation:
# +/- 10 percent

INERTIA_SCALE_MIN = 0.90
INERTIA_SCALE_MAX = 1.10


# Constant disturbance acceleration.
#
# Horizontal:
# +/- 0.35 m/s^2
#
# Vertical:
# +/- 0.175 m/s^2

WIND_XY_MAX = 0.35
WIND_Z_MAX = 0.175


# ============================================================
# SUCCESS REQUIREMENT
# ============================================================

# A run is successful only if:
#
# 1. The mission completes.
# 2. All 5 waypoints are reached.
# 3. Final position error is less than 0.25 m.

FINAL_ERROR_LIMIT_M = 0.25


# ============================================================
# PARAMETER ORDER
# ============================================================

# Optimization vector:
#
# theta =
#
# [Kpx, Kpy, Kpz, Kvx, Kvy, Kvz]

PARAMETER_NAMES = [
    "Kpx",
    "Kpy",
    "Kpz",
    "Kvx",
    "Kvy",
    "Kvz",
]


# ============================================================
# SEARCH BOUNDS
# ============================================================

LOWER_BOUNDS = np.array(
    [
        0.30,   # Kpx
        0.30,   # Kpy
        0.50,   # Kpz

        0.60,   # Kvx
        0.60,   # Kvy
        0.80,   # Kvz
    ],
    dtype=float,
)


# Increased upper bounds because the previous robust optimizer
# ended with Kvx = 5.0 and Kvz = 5.0 exactly at their bounds.

UPPER_BOUNDS = np.array(
    [
        4.00,   # Kpx
        4.00,   # Kpy
        5.00,   # Kpz

        7.00,   # Kvx
        7.00,   # Kvy
        7.00,   # Kvz
    ],
    dtype=float,
)


# ============================================================
# SEARCH RESOLUTION
# ============================================================

INITIAL_STEP = np.array(
    [
        0.30,
        0.30,
        0.30,

        0.30,
        0.30,
        0.30,
    ],
    dtype=float,
)


# Allow finer final tuning than before.

MINIMUM_STEP = 0.05


# Allow more optimization sweeps than before.

MAX_SWEEPS = 10


# ============================================================
# CURRENT STARTING GAINS
# ============================================================

GAIN_FILE = (
    ROOT
    / "config"
    / "optimized_outer_loop_gains.json"
)


# The script starts from the gains currently stored in the
# optimized gain file.
#
# After your previous run, this should be approximately:
#
# kp_pos = [2.7, 1.0, 1.4]
# kd_pos = [5.0, 2.6, 5.0]

if GAIN_FILE.exists():

    existing_gain_data = json.loads(
        GAIN_FILE.read_text()
    )

    INITIAL_GAINS = np.r_[
        np.asarray(
            existing_gain_data["kp_pos"],
            dtype=float,
        ),
        np.asarray(
            existing_gain_data["kd_pos"],
            dtype=float,
        ),
    ]

else:

    INITIAL_GAINS = np.array(
        [
            0.90,
            0.90,
            2.00,

            1.90,
            1.90,
            1.70,
        ],
        dtype=float,
    )


# ============================================================
# ROBUST OBJECTIVE WEIGHTS
# ============================================================

# We penalize both average performance and poor-tail
# performance.
#
# Failure rate receives the strongest single penalty.

W_TRAJ_MEAN = 1.00
W_TRAJ_P95 = 1.00

W_CROSS_MEAN = 0.50
W_CROSS_P95 = 0.50

W_FINAL_MEAN = 0.75
W_FINAL_P95 = 0.75

W_FAILURE_RATE = 3.00

W_TORQUE_SAT = 0.02


# ============================================================
# WIND MODEL
# ============================================================

class ConstantWindAcceleration:

    def __init__(
        self,
        acceleration_vector,
    ):

        self._acceleration = np.asarray(
            acceleration_vector,
            dtype=float,
        )

    def acceleration(self):

        return self._acceleration.copy()


# ============================================================
# CROSS-TRACK HELPER
# ============================================================

def point_segment_distance(
    p,
    a,
    b,
):

    ab = b - a

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
        ((p - a) @ ab) / den,
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
# GENERATE FIXED TRAINING CASES
# ============================================================

def generate_training_cases():

    rng = np.random.default_rng(
        TRAINING_MASTER_SEED
    )

    cases = []

    for case_index in range(
        N_TRAINING_CASES
    ):

        mass_scale = float(
            rng.uniform(
                MASS_SCALE_MIN,
                MASS_SCALE_MAX,
            )
        )

        inertia_scale = rng.uniform(
            INERTIA_SCALE_MIN,
            INERTIA_SCALE_MAX,
            size=3,
        )

        wind = np.array(
            [
                rng.uniform(
                    -WIND_XY_MAX,
                    WIND_XY_MAX,
                ),

                rng.uniform(
                    -WIND_XY_MAX,
                    WIND_XY_MAX,
                ),

                rng.uniform(
                    -WIND_Z_MAX,
                    WIND_Z_MAX,
                ),
            ],
            dtype=float,
        )

        sensor_seed = int(
            rng.integers(
                1,
                2**31 - 1,
            )
        )

        cases.append(
            {
                "case":
                    case_index + 1,

                "mass_scale":
                    mass_scale,

                "inertia_scale":
                    inertia_scale,

                "wind":
                    wind,

                "sensor_seed":
                    sensor_seed,
            }
        )

    return cases


TRAINING_CASES = (
    generate_training_cases()
)


# ============================================================
# APPLY INERTIA DISPERSION
# ============================================================

def apply_inertia_scale(
    vehicle,
    scale,
):

    scale = np.asarray(
        scale,
        dtype=float,
    )

    if not hasattr(
        vehicle.params,
        "inertia",
    ):

        return

    inertia = np.asarray(
        vehicle.params.inertia,
        dtype=float,
    )

    if inertia.shape == (3,):

        vehicle.params.inertia = (
            inertia
            * scale
        )

    elif inertia.shape == (3, 3):

        modified = inertia.copy()

        modified[0, 0] *= (
            scale[0]
        )

        modified[1, 1] *= (
            scale[1]
        )

        modified[2, 2] *= (
            scale[2]
        )

        vehicle.params.inertia = (
            modified
        )


# ============================================================
# RUN ONE DISPERSED CASE
# ============================================================

def run_one_case(
    gains,
    case,
):

    gains = np.asarray(
        gains,
        dtype=float,
    )


    # --------------------------------------------------------
    # NOMINAL MODEL VALUES USED BY CONTROLLER
    # --------------------------------------------------------

    nominal_vehicle = QuadrotorModel()

    nominal_mass = float(
        nominal_vehicle.params.mass
    )

    nominal_gravity = float(
        nominal_vehicle.params.gravity
    )


    # --------------------------------------------------------
    # PHYSICAL PLANT
    # --------------------------------------------------------

    vehicle = QuadrotorModel()

    vehicle.params.mass = (
        nominal_mass
        * case["mass_scale"]
    )

    apply_inertia_scale(
        vehicle,
        case["inertia_scale"],
    )


    # --------------------------------------------------------
    # CONTROLLER
    # --------------------------------------------------------

    # IMPORTANT:
    #
    # The controller receives NOMINAL vehicle mass.
    #
    # Therefore the physical plant can be heavier or lighter
    # without the controller knowing the exact mass.

    controller = CascadedController(
        nominal_mass,
        nominal_gravity,
    )

    controller.kp_pos = np.asarray(
        gains[0:3],
        dtype=float,
    )

    controller.kd_pos = np.asarray(
        gains[3:6],
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

    sensors = ImuGpsSensorSuite(
        seed=case["sensor_seed"],
    )


    # --------------------------------------------------------
    # EKF
    # --------------------------------------------------------

    ekf = ErrorStateEKF15(
        dt=DT,
    )


    # --------------------------------------------------------
    # WIND
    # --------------------------------------------------------

    wind = ConstantWindAcceleration(
        case["wind"]
    )


    # --------------------------------------------------------
    # RUN CLOSED-LOOP MISSION
    # --------------------------------------------------------

    try:

        logs = run_waypoint_mission(
            vehicle,
            controller,
            guidance,
            sensors,
            ekf,
            duration=DURATION,
            dt=DT,
            gps_hz=GPS_HZ,
            wind=wind,
        )

    except Exception as exc:

        return {
            "valid":
                False,

            "success":
                False,

            "failure_reason":
                f"simulation_exception: {exc}",

            "trajectory_rmse_m":
                10.0,

            "cross_track_rmse_m":
                10.0,

            "final_error_m":
                10.0,

            "ekf_position_rmse_m":
                10.0,

            "torque_saturation_pct":
                100.0,
        }


    # --------------------------------------------------------
    # NUMERICAL CHECK
    # --------------------------------------------------------

    if not np.all(
        np.isfinite(
            logs["state"]
        )
    ):

        return {
            "valid":
                False,

            "success":
                False,

            "failure_reason":
                "nonfinite_state",

            "trajectory_rmse_m":
                10.0,

            "cross_track_rmse_m":
                10.0,

            "final_error_m":
                10.0,

            "ekf_position_rmse_m":
                10.0,

            "torque_saturation_pct":
                100.0,
        }


    # ========================================================
    # TRAJECTORY METRICS
    # ========================================================

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


    # ========================================================
    # EKF METRICS
    # ========================================================

    estimator_metrics = (
        estimation_metrics(
            logs["state"],
            logs["estimate"],
        )
    )


    # ========================================================
    # CROSS-TRACK ERROR
    # ========================================================

    path_points = np.vstack(
        [
            logs[
                "state"
            ][0, 0:3],

            WAYPOINTS,
        ]
    )

    path_error = []

    for position in logs[
        "state"
    ][:, 0:3]:

        path_error.append(

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

    path_error = np.asarray(
        path_error,
        dtype=float,
    )

    cross_track_rmse = float(
        np.sqrt(
            np.mean(
                path_error**2
            )
        )
    )


    # ========================================================
    # ACTUATOR SATURATION
    # ========================================================

    torque = (
        logs[
            "control"
        ][:, 1:4]
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
    # SUCCESS / FAILURE
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

    final_error = float(
        metrics[
            "final_error_m"
        ]
    )

    success = bool(

        mission_complete

        and waypoints_reached
        == len(WAYPOINTS)

        and final_error
        < FINAL_ERROR_LIMIT_M
    )


    if not mission_complete:

        failure_reason = (
            "mission_not_complete"
        )

    elif (
        waypoints_reached
        != len(WAYPOINTS)
    ):

        failure_reason = (
            "waypoints_not_completed"
        )

    elif (
        final_error
        >= FINAL_ERROR_LIMIT_M
    ):

        failure_reason = (
            "final_error_limit"
        )

    else:

        failure_reason = (
            "none"
        )


    return {

        "valid":
            True,

        "success":
            success,

        "failure_reason":
            failure_reason,

        "trajectory_rmse_m":
            float(
                metrics[
                    "trajectory_position_rmse_m"
                ]
            ),

        "cross_track_rmse_m":
            cross_track_rmse,

        "final_error_m":
            final_error,

        "ekf_position_rmse_m":
            float(
                estimator_metrics[
                    "ekf_position_rmse_m"
                ]
            ),

        "torque_saturation_pct":
            torque_saturation_pct,
    }


# ============================================================
# SUMMARIZE ONE CANDIDATE
# ============================================================

def summarize_candidate(
    case_results,
):

    trajectory = np.asarray(
        [
            r["trajectory_rmse_m"]
            for r in case_results
        ],
        dtype=float,
    )

    cross_track = np.asarray(
        [
            r["cross_track_rmse_m"]
            for r in case_results
        ],
        dtype=float,
    )

    final_error = np.asarray(
        [
            r["final_error_m"]
            for r in case_results
        ],
        dtype=float,
    )

    torque_sat = np.asarray(
        [
            r["torque_saturation_pct"]
            for r in case_results
        ],
        dtype=float,
    )

    successes = np.asarray(
        [
            r["success"]
            for r in case_results
        ],
        dtype=float,
    )

    success_rate = float(
        np.mean(
            successes
        )
    )

    failure_rate = float(
        1.0
        - success_rate
    )

    summary = {

        "success_rate":
            success_rate,

        "failure_rate":
            failure_rate,

        "trajectory_mean_m":
            float(
                np.mean(
                    trajectory
                )
            ),

        "trajectory_p95_m":
            float(
                np.percentile(
                    trajectory,
                    95,
                )
            ),

        "cross_track_mean_m":
            float(
                np.mean(
                    cross_track
                )
            ),

        "cross_track_p95_m":
            float(
                np.percentile(
                    cross_track,
                    95,
                )
            ),

        "final_error_mean_m":
            float(
                np.mean(
                    final_error
                )
            ),

        "final_error_p95_m":
            float(
                np.percentile(
                    final_error,
                    95,
                )
            ),

        "torque_saturation_mean_pct":
            float(
                np.mean(
                    torque_sat
                )
            ),
    }


    # ========================================================
    # ROBUST OBJECTIVE FUNCTION
    # ========================================================

    cost = (

        W_TRAJ_MEAN
        * summary[
            "trajectory_mean_m"
        ]

        + W_TRAJ_P95
        * summary[
            "trajectory_p95_m"
        ]

        + W_CROSS_MEAN
        * summary[
            "cross_track_mean_m"
        ]

        + W_CROSS_P95
        * summary[
            "cross_track_p95_m"
        ]

        + W_FINAL_MEAN
        * summary[
            "final_error_mean_m"
        ]

        + W_FINAL_P95
        * summary[
            "final_error_p95_m"
        ]

        + W_FAILURE_RATE
        * summary[
            "failure_rate"
        ]

        + W_TORQUE_SAT
        * summary[
            "torque_saturation_mean_pct"
        ]
    )

    summary[
        "cost"
    ] = float(
        cost
    )

    return summary


# ============================================================
# EVALUATE ONE GAIN VECTOR
# ============================================================

def evaluate_gains(
    gains,
    verbose=False,
):

    case_results = []

    for case in TRAINING_CASES:

        result = run_one_case(
            gains,
            case,
        )

        case_results.append(
            result
        )

        if verbose:

            status = (
                "PASS"
                if result["success"]
                else "FAIL"
            )

            print(
                f"    Case "
                f"{case['case']:02d}/"
                f"{N_TRAINING_CASES:02d} | "
                f"{status} | "
                f"traj "
                f"{result['trajectory_rmse_m']:.3f} m | "
                f"cross "
                f"{result['cross_track_rmse_m']:.3f} m | "
                f"final "
                f"{result['final_error_m']:.3f} m"
            )

    summary = summarize_candidate(
        case_results
    )

    return (
        summary,
        case_results,
    )


# ============================================================
# PRINT CANDIDATE SUMMARY
# ============================================================

def print_candidate_summary(
    name,
    gains,
    summary,
):

    print()

    print(
        name
    )

    print(
        "-" * 78
    )

    print(
        f"Gains:              "
        f"{np.round(gains, 4)}"
    )

    print(
        f"Cost:               "
        f"{summary['cost']:.5f}"
    )

    print(
        f"Success rate:       "
        f"{100.0 * summary['success_rate']:.1f} %"
    )

    print(
        f"Trajectory mean:    "
        f"{summary['trajectory_mean_m']:.4f} m"
    )

    print(
        f"Trajectory p95:     "
        f"{summary['trajectory_p95_m']:.4f} m"
    )

    print(
        f"Cross-track mean:   "
        f"{summary['cross_track_mean_m']:.4f} m"
    )

    print(
        f"Cross-track p95:    "
        f"{summary['cross_track_p95_m']:.4f} m"
    )

    print(
        f"Final-error mean:   "
        f"{summary['final_error_mean_m']:.4f} m"
    )

    print(
        f"Final-error p95:    "
        f"{summary['final_error_p95_m']:.4f} m"
    )


# ============================================================
# OPTIMIZER
# ============================================================

def optimize():

    print()

    print(
        "=" * 78
    )

    print(
        "ROBUST OUTER-LOOP GAIN OPTIMIZATION"
    )

    print(
        "=" * 78
    )

    print(
        f"Training Monte Carlo cases: "
        f"{N_TRAINING_CASES}"
    )

    print(
        f"Training random seed:       "
        f"{TRAINING_MASTER_SEED}"
    )

    print(
        f"Mass dispersion:            "
        f"{MASS_SCALE_MIN:.2f} to "
        f"{MASS_SCALE_MAX:.2f}"
    )

    print(
        f"Inertia dispersion:         "
        f"{INERTIA_SCALE_MIN:.2f} to "
        f"{INERTIA_SCALE_MAX:.2f}"
    )

    print(
        f"Wind XY range:              "
        f"+/- {WIND_XY_MAX:.3f} m/s^2"
    )

    print(
        f"Final-error success limit:  "
        f"{FINAL_ERROR_LIMIT_M:.3f} m"
    )


    # --------------------------------------------------------
    # START FROM CURRENT SAVED ROBUST GAINS
    # --------------------------------------------------------

    current_gains = (
        INITIAL_GAINS.copy()
    )

    step = (
        INITIAL_STEP.copy()
    )


    # --------------------------------------------------------
    # INITIAL EVALUATION
    # --------------------------------------------------------

    print()

    print(
        "Evaluating starting controller..."
    )

    (
        current_summary,
        current_cases,
    ) = evaluate_gains(
        current_gains,
        verbose=True,
    )

    print_candidate_summary(
        "STARTING ROBUST PERFORMANCE",
        current_gains,
        current_summary,
    )


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history = []

    history.append(
        {
            "iteration":
                0,

            "sweep":
                0,

            "parameter":
                "initial",

            "gains":
                current_gains.tolist(),

            **current_summary,
        }
    )


    # ========================================================
    # SEQUENTIAL SEARCH
    # ========================================================

    for sweep in range(
        1,
        MAX_SWEEPS + 1,
    ):

        print()

        print(
            "=" * 78
        )

        print(
            f"ROBUST OPTIMIZATION SWEEP "
            f"{sweep}/{MAX_SWEEPS}"
        )

        print(
            "=" * 78
        )

        improved_this_sweep = (
            False
        )


        # ----------------------------------------------------
        # TUNE ONE PARAMETER AT A TIME
        # ----------------------------------------------------

        for i in range(
            len(
                current_gains
            )
        ):

            parameter_name = (
                PARAMETER_NAMES[i]
            )

            base_value = (
                current_gains[i]
            )

            best_local_gains = (
                current_gains.copy()
            )

            best_local_summary = (
                current_summary.copy()
            )


            print()

            print(
                f"Testing "
                f"{parameter_name} "
                f"around "
                f"{base_value:.3f}"
            )


            # ------------------------------------------------
            # TRY LOWER AND HIGHER VALUE
            # ------------------------------------------------

            for direction in (
                -1.0,
                +1.0,
            ):

                candidate = (
                    current_gains.copy()
                )

                candidate[i] = np.clip(

                    base_value
                    + direction
                    * step[i],

                    LOWER_BOUNDS[i],

                    UPPER_BOUNDS[i],
                )


                if np.isclose(
                    candidate[i],
                    base_value,
                ):

                    continue


                (
                    candidate_summary,
                    candidate_cases,
                ) = evaluate_gains(
                    candidate,
                    verbose=False,
                )


                print(

                    f"  "
                    f"{parameter_name}"
                    f" = "
                    f"{candidate[i]:.3f}"
                    f" | J = "
                    f"{candidate_summary['cost']:.5f}"
                    f" | success = "
                    f"{100.0 * candidate_summary['success_rate']:.0f}%"
                    f" | traj mean = "
                    f"{candidate_summary['trajectory_mean_m']:.3f}"
                    f" | traj p95 = "
                    f"{candidate_summary['trajectory_p95_m']:.3f}"
                    f" | final p95 = "
                    f"{candidate_summary['final_error_p95_m']:.3f}"
                )


                if (
                    candidate_summary[
                        "cost"
                    ]
                    <
                    best_local_summary[
                        "cost"
                    ]
                ):

                    best_local_gains = (
                        candidate.copy()
                    )

                    best_local_summary = (
                        candidate_summary.copy()
                    )


            # ------------------------------------------------
            # ACCEPT / REJECT
            # ------------------------------------------------

            if (
                best_local_summary[
                    "cost"
                ]
                <
                current_summary[
                    "cost"
                ]
            ):

                current_gains = (
                    best_local_gains
                )

                current_summary = (
                    best_local_summary
                )

                improved_this_sweep = (
                    True
                )


                print(

                    f"  ACCEPTED "
                    f"{parameter_name}"
                    f" = "
                    f"{current_gains[i]:.3f}"
                )


                history.append(
                    {
                        "iteration":
                            len(history),

                        "sweep":
                            sweep,

                        "parameter":
                            parameter_name,

                        "gains":
                            current_gains.tolist(),

                        **current_summary,
                    }
                )

            else:

                print(
                    f"  REJECTED changes "
                    f"to {parameter_name}"
                )


        # ----------------------------------------------------
        # REDUCE STEP IF FULL SWEEP STALLS
        # ----------------------------------------------------

        if not improved_this_sweep:

            step *= 0.5

            print()

            print(
                "No accepted changes in this sweep."
            )

            print(
                f"Reduced step sizes to: "
                f"{step}"
            )


        # ----------------------------------------------------
        # STOP WHEN SEARCH RESOLUTION IS SMALL ENOUGH
        # ----------------------------------------------------

        if (
            np.max(step)
            <
            MINIMUM_STEP
        ):

            print()

            print(
                "Minimum search resolution reached."
            )

            break


    # ========================================================
    # FINAL TRAINING EVALUATION
    # ========================================================

    print()

    print(
        "=" * 78
    )

    print(
        "FINAL ROBUST TRAINING EVALUATION"
    )

    print(
        "=" * 78
    )

    (
        final_summary,
        final_cases,
    ) = evaluate_gains(
        current_gains,
        verbose=True,
    )

    print_candidate_summary(
        "ROBUST OPTIMIZATION COMPLETE",
        current_gains,
        final_summary,
    )


    # ========================================================
    # SAVE ROBUST GAINS
    # ========================================================

    kp_pos = (
        current_gains[
            0:3
        ]
    )

    kd_pos = (
        current_gains[
            3:6
        ]
    )


    output = {

        "tuning_method":
            "robust_sequential_coordinate_search",

        "kp_pos":
            kp_pos.tolist(),

        "kd_pos":
            kd_pos.tolist(),

        "training_cases":
            N_TRAINING_CASES,

        "training_seed":
            TRAINING_MASTER_SEED,

        "training_summary":
            final_summary,

        "uncertainty_model":
            {

                "mass_scale_min":
                    MASS_SCALE_MIN,

                "mass_scale_max":
                    MASS_SCALE_MAX,

                "inertia_scale_min":
                    INERTIA_SCALE_MIN,

                "inertia_scale_max":
                    INERTIA_SCALE_MAX,

                "wind_xy_max_mps2":
                    WIND_XY_MAX,

                "wind_z_max_mps2":
                    WIND_Z_MAX,
            },

        "search_bounds":
            {

                "lower":
                    LOWER_BOUNDS.tolist(),

                "upper":
                    UPPER_BOUNDS.tolist(),
            },

        "success_definition":
            {

                "all_waypoints_completed":
                    True,

                "mission_complete":
                    True,

                "final_error_limit_m":
                    FINAL_ERROR_LIMIT_M,
            },
    }


    GAIN_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    GAIN_FILE.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )


    # ========================================================
    # SAVE OPTIMIZATION HISTORY
    # ========================================================

    results_dir = (
        ROOT
        / "results"
        / "data"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    history_path = (
        results_dir
        / "robust_gain_optimization_history.json"
    )

    history_path.write_text(
        json.dumps(
            history,
            indent=2,
        )
    )


    # ========================================================
    # SAVE FINAL TRAINING CASE RESULTS
    # ========================================================

    training_case_path = (
        results_dir
        / "robust_gain_training_cases.json"
    )

    serializable_cases = []


    for case, result in zip(
        TRAINING_CASES,
        final_cases,
    ):

        serializable_cases.append(
            {
                "case":
                    case["case"],

                "mass_scale":
                    float(
                        case["mass_scale"]
                    ),

                "inertia_scale":
                    np.asarray(
                        case[
                            "inertia_scale"
                        ]
                    ).tolist(),

                "wind_mps2":
                    np.asarray(
                        case["wind"]
                    ).tolist(),

                "sensor_seed":
                    int(
                        case[
                            "sensor_seed"
                        ]
                    ),

                "result":
                    result,
            }
        )


    training_case_path.write_text(
        json.dumps(
            serializable_cases,
            indent=2,
        )
    )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()

    print(
        "=" * 78
    )

    print(
        "ROBUST OPTIMIZED GAINS"
    )

    print(
        "=" * 78
    )


    print(
        f"kp_pos = "
        f"{kp_pos}"
    )

    print(
        f"kd_pos = "
        f"{kd_pos}"
    )


    print()

    print(
        "TRAINING-SET ROBUSTNESS"
    )

    print(
        "-" * 78
    )


    print(
        f"Success rate:              "
        f"{100.0 * final_summary['success_rate']:.1f} %"
    )


    print(
        f"Trajectory RMSE mean:      "
        f"{final_summary['trajectory_mean_m']:.3f} m"
    )


    print(
        f"Trajectory RMSE p95:       "
        f"{final_summary['trajectory_p95_m']:.3f} m"
    )


    print(
        f"Cross-track RMSE mean:     "
        f"{final_summary['cross_track_mean_m']:.3f} m"
    )


    print(
        f"Cross-track RMSE p95:      "
        f"{final_summary['cross_track_p95_m']:.3f} m"
    )


    print(
        f"Final error mean:          "
        f"{final_summary['final_error_mean_m']:.3f} m"
    )


    print(
        f"Final error p95:           "
        f"{final_summary['final_error_p95_m']:.3f} m"
    )


    print()

    print(
        "Saved robust gains:"
    )

    print(
        GAIN_FILE
    )


    print()

    print(
        "Saved optimization history:"
    )

    print(
        history_path
    )


    print()

    print(
        "IMPORTANT:"
    )

    print(
        "These are training results only."
    )

    print(
        "After reviewing them, run "
        "run_monte_carlo.py for independent "
        "100-case validation."
    )


    print(
        "=" * 78
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    optimize()