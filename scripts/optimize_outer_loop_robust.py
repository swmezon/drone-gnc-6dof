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

from drone_gnc.vehicles import QuadrotorModel

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
# ROBUST TRAINING SET
# ============================================================

N_TRAINING_CASES = 10

TRAINING_MASTER_SEED = 314159


# ============================================================
# UNCERTAINTY MODEL
# ============================================================

MASS_SCALE_MIN = 0.90
MASS_SCALE_MAX = 1.10

INERTIA_SCALE_MIN = 0.90
INERTIA_SCALE_MAX = 1.10

WIND_XY_MAX = 0.35
WIND_Z_MAX = 0.175


# ============================================================
# SUCCESS DEFINITION
# ============================================================

FINAL_ERROR_LIMIT_M = 0.25


# ============================================================
# OPTIMIZATION VECTOR
# ============================================================

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
        0.30,
        0.30,
        0.50,

        0.60,
        0.60,
        0.80,

        0.00,
        0.00,
        0.00,
    ],
    dtype=float,
)


UPPER_BOUNDS = np.array(
    [
        4.00,
        4.00,
        5.00,

        7.00,
        7.00,
        7.00,

        1.00,
        1.00,
        1.50,
    ],
    dtype=float,
)


# ============================================================
# SEARCH STEP SIZES
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


MINIMUM_STEP = 0.025

MAX_SWEEPS = 8


# ============================================================
# LOAD CURRENT STARTING GAINS
# ============================================================

GAIN_FILE = (
    ROOT
    / "config"
    / "optimized_outer_loop_gains.json"
)


if GAIN_FILE.exists():

    gain_data = json.loads(
        GAIN_FILE.read_text()
    )

    kp_start = np.asarray(
        gain_data.get(
            "kp_pos",
            [0.9, 0.9, 2.0],
        ),
        dtype=float,
    )


    kv_start = np.asarray(
        gain_data.get(
            "kd_pos",
            [1.9, 1.9, 1.7],
        ),
        dtype=float,
    )


    ki_start = np.asarray(
        gain_data.get(
            "ki_pos",
            [0.05, 0.05, 0.10],
        ),
        dtype=float,
    )


    INITIAL_GAINS = np.r_[
        kp_start,
        kv_start,
        ki_start,
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

            0.05,
            0.05,
            0.10,
        ],
        dtype=float,
    )


# ============================================================
# SECONDARY PERFORMANCE COST
# ============================================================

W_TRAJ_MEAN = 1.00
W_TRAJ_P95 = 1.00

W_CROSS_MEAN = 0.50
W_CROSS_P95 = 0.50

W_FINAL_MEAN = 0.75
W_FINAL_P95 = 1.00

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
# CROSS-TRACK DISTANCE
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
# FIXED TRAINING CASE GENERATION
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

    if not hasattr(
        vehicle.params,
        "inertia",
    ):

        return


    scale = np.asarray(
        scale,
        dtype=float,
    )


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

        modified[0, 0] *= scale[0]
        modified[1, 1] *= scale[1]
        modified[2, 2] *= scale[2]

        vehicle.params.inertia = (
            modified
        )


# ============================================================
# RUN ONE ROBUST CASE
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
    # NOMINAL VALUES KNOWN TO CONTROLLER
    # --------------------------------------------------------

    nominal_vehicle = QuadrotorModel()

    nominal_mass = float(
        nominal_vehicle.params.mass
    )

    nominal_gravity = float(
        nominal_vehicle.params.gravity
    )


    # --------------------------------------------------------
    # ACTUAL PHYSICAL PLANT
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
    # SENSOR REALIZATION
    # --------------------------------------------------------

    sensors = ImuGpsSensorSuite(
        seed=case[
            "sensor_seed"
        ],
    )


    # --------------------------------------------------------
    # EKF
    # --------------------------------------------------------

    ekf = ErrorStateEKF15(
        dt=DT,
    )


    # --------------------------------------------------------
    # DISTURBANCE
    # --------------------------------------------------------

    wind = ConstantWindAcceleration(
        case[
            "wind"
        ]
    )


    # --------------------------------------------------------
    # RUN MISSION
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
            "success":
                False,

            "failure_reason":
                f"simulation_exception:{exc}",

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


    if not np.all(
        np.isfinite(
            logs["state"]
        )
    ):

        return {
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
    # ESTIMATOR METRICS
    # ========================================================

    estimator = estimation_metrics(
        logs["state"],
        logs["estimate"],
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
                    len(path_points) - 1
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
    # SUCCESS DEFINITION
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
                estimator[
                    "ekf_position_rmse_m"
                ]
            ),

        "torque_saturation_pct":
            torque_saturation_pct,
    }


# ============================================================
# SUMMARIZE CANDIDATE
# ============================================================

def summarize_candidate(
    results,
):

    trajectory = np.asarray(
        [
            r[
                "trajectory_rmse_m"
            ]
            for r in results
        ]
    )


    cross = np.asarray(
        [
            r[
                "cross_track_rmse_m"
            ]
            for r in results
        ]
    )


    final_error = np.asarray(
        [
            r[
                "final_error_m"
            ]
            for r in results
        ]
    )


    torque_sat = np.asarray(
        [
            r[
                "torque_saturation_pct"
            ]
            for r in results
        ]
    )


    success_array = np.asarray(
        [
            r["success"]
            for r in results
        ],
        dtype=float,
    )


    success_rate = float(
        np.mean(
            success_array
        )
    )


    performance_cost = (

        W_TRAJ_MEAN
        * float(
            np.mean(
                trajectory
            )
        )

        + W_TRAJ_P95
        * float(
            np.percentile(
                trajectory,
                95,
            )
        )

        + W_CROSS_MEAN
        * float(
            np.mean(
                cross
            )
        )

        + W_CROSS_P95
        * float(
            np.percentile(
                cross,
                95,
            )
        )

        + W_FINAL_MEAN
        * float(
            np.mean(
                final_error
            )
        )

        + W_FINAL_P95
        * float(
            np.percentile(
                final_error,
                95,
            )
        )

        + W_TORQUE_SAT
        * float(
            np.mean(
                torque_sat
            )
        )
    )


    failure_reasons = Counter(
        r[
            "failure_reason"
        ]
        for r in results
        if not r[
            "success"
        ]
    )


    return {

        "success_rate":
            success_rate,

        "successful_cases":
            int(
                np.sum(
                    success_array
                )
            ),

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
                    cross
                )
            ),

        "cross_track_p95_m":
            float(
                np.percentile(
                    cross,
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

        "performance_cost":
            float(
                performance_cost
            ),

        "failure_reasons":
            dict(
                failure_reasons
            ),
    }


# ============================================================
# IMPORTANT:
# PRIMARY OPTIMIZATION RULE
# ============================================================

def candidate_is_better(
    candidate,
    current,
):
    """
    Lexicographic robust optimization.

    Priority 1:
        Higher mission-success rate.

    Priority 2:
        If success rates are equal,
        lower performance cost wins.

    This prevents a fragile controller with prettier RMSE
    from defeating a controller that completes more missions.
    """

    candidate_success = (
        candidate[
            "success_rate"
        ]
    )

    current_success = (
        current[
            "success_rate"
        ]
    )


    if (
        candidate_success
        >
        current_success
        + 1e-12
    ):

        return True


    if (
        candidate_success
        <
        current_success
        - 1e-12
    ):

        return False


    return (
        candidate[
            "performance_cost"
        ]
        <
        current[
            "performance_cost"
        ]
    )


# ============================================================
# EVALUATE GAIN VECTOR
# ============================================================

def evaluate_gains(
    gains,
    verbose=False,
):

    results = []


    for case in TRAINING_CASES:

        result = run_one_case(
            gains,
            case,
        )


        results.append(
            result
        )


        if verbose:

            status = (
                "PASS"
                if result[
                    "success"
                ]
                else "FAIL"
            )


            print(

                f"Case "
                f"{case['case']:02d}"
                f"/{N_TRAINING_CASES:02d}"
                f" | {status}"
                f" | traj "
                f"{result['trajectory_rmse_m']:.3f} m"
                f" | cross "
                f"{result['cross_track_rmse_m']:.3f} m"
                f" | final "
                f"{result['final_error_m']:.3f} m"
                f" | reason "
                f"{result['failure_reason']}"
            )


    return (
        summarize_candidate(
            results
        ),
        results,
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


    print()

    print(
        "=" * 80
    )

    print(
        "ROBUST OUTER-LOOP PID GAIN OPTIMIZATION"
    )

    print(
        "=" * 80
    )


    print(
        f"Training cases: "
        f"{N_TRAINING_CASES}"
    )


    print(
        f"Starting gains:"
    )


    print(
        f"Kp = "
        f"{current_gains[0:3]}"
    )


    print(
        f"Kv = "
        f"{current_gains[3:6]}"
    )


    print(
        f"Ki = "
        f"{current_gains[6:9]}"
    )


    print()

    print(
        "Evaluating starting controller..."
    )


    (
        current_summary,
        current_results,
    ) = evaluate_gains(
        current_gains,
        verbose=True,
    )


    history = []


    history.append(
        {
            "iteration":
                0,

            "parameter":
                "initial",

            "gains":
                current_gains.tolist(),

            **current_summary,
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
            "=" * 80
        )

        print(
            f"SWEEP "
            f"{sweep}/{MAX_SWEEPS}"
        )

        print(
            "=" * 80
        )


        improved = False


        for i, name in enumerate(
            PARAMETER_NAMES
        ):

            base = (
                current_gains[i]
            )


            best_gains = (
                current_gains.copy()
            )


            best_summary = (
                current_summary.copy()
            )


            print()

            print(
                f"Testing {name} "
                f"around {base:.3f}"
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


                (
                    candidate_summary,
                    candidate_results,
                ) = evaluate_gains(
                    candidate,
                    verbose=False,
                )


                print(

                    f"  {name}"
                    f" = "
                    f"{candidate[i]:.3f}"
                    f" | success "
                    f"{100.0 * candidate_summary['success_rate']:.0f}%"
                    f" | J "
                    f"{candidate_summary['performance_cost']:.4f}"
                    f" | traj mean "
                    f"{candidate_summary['trajectory_mean_m']:.3f}"
                    f" | traj p95 "
                    f"{candidate_summary['trajectory_p95_m']:.3f}"
                    f" | final p95 "
                    f"{candidate_summary['final_error_p95_m']:.3f}"
                )


                if candidate_is_better(
                    candidate_summary,
                    best_summary,
                ):

                    best_gains = (
                        candidate.copy()
                    )

                    best_summary = (
                        candidate_summary.copy()
                    )


            if candidate_is_better(
                best_summary,
                current_summary,
            ):

                current_gains = (
                    best_gains
                )

                current_summary = (
                    best_summary
                )

                improved = True


                print(
                    f"  ACCEPTED "
                    f"{name}"
                    f" = "
                    f"{current_gains[i]:.3f}"
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

                        **current_summary,
                    }
                )


            else:

                print(
                    f"  REJECTED changes "
                    f"to {name}"
                )


        if not improved:

            step *= 0.5


            print()

            print(
                "No accepted change."
            )


            print(
                "Reducing search step:"
            )


            print(
                step
            )


        if (
            np.max(
                step
            )
            < MINIMUM_STEP
        ):

            print(
                "Minimum search resolution reached."
            )

            break


    # ========================================================
    # FINAL TRAINING EVALUATION
    # ========================================================

    (
        final_summary,
        final_results,
    ) = evaluate_gains(
        current_gains,
        verbose=True,
    )


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

    ki_pos = (
        current_gains[
            6:9
        ]
    )


    # ========================================================
    # SAVE GAINS
    # ========================================================

    output = {

        "tuning_method":
            "robust_outer_loop_pid_coordinate_search",

        "kp_pos":
            kp_pos.tolist(),

        "kd_pos":
            kd_pos.tolist(),

        "ki_pos":
            ki_pos.tolist(),

        "training_cases":
            N_TRAINING_CASES,

        "training_seed":
            TRAINING_MASTER_SEED,

        "training_summary":
            final_summary,

        "uncertainty_model":
            {

                "mass_scale":
                    [
                        MASS_SCALE_MIN,
                        MASS_SCALE_MAX,
                    ],

                "inertia_scale":
                    [
                        INERTIA_SCALE_MIN,
                        INERTIA_SCALE_MAX,
                    ],

                "wind_xy_max_mps2":
                    WIND_XY_MAX,

                "wind_z_max_mps2":
                    WIND_Z_MAX,
            },

        "final_error_limit_m":
            FINAL_ERROR_LIMIT_M,
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


    results_dir = (
        ROOT
        / "results"
        / "data"
    )


    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    (
        results_dir
        / "robust_gain_optimization_history.json"
    ).write_text(

        json.dumps(
            history,
            indent=2,
        )
    )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "ROBUST OPTIMIZED GAINS"
    )

    print(
        "=" * 80
    )


    print(
        f"kp_pos = {kp_pos}"
    )


    print(
        f"kd_pos = {kd_pos}"
    )


    print(
        f"ki_pos = {ki_pos}"
    )


    print()

    print(
        "TRAINING-SET ROBUSTNESS"
    )

    print(
        "-" * 80
    )


    print(
        f"Successful cases:          "
        f"{final_summary['successful_cases']} "
        f"/ {N_TRAINING_CASES}"
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
        "FAILURE REASONS"
    )

    print(
        "-" * 80
    )


    print(
        final_summary[
            "failure_reasons"
        ]
    )


    print()

    print(
        "Saved gains:"
    )

    print(
        GAIN_FILE
    )


    print(
        "=" * 80
    )


if __name__ == "__main__":

    optimize()