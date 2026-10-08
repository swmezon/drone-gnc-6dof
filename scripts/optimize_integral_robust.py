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

# IMPORTANT:
#
# These are exactly the same fixed training conditions
# used by the previous robust optimizer.
#
# Every Ki candidate sees the same 10 cases.

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
# LOAD CURRENT CONTROLLER GAINS
# ============================================================

GAIN_FILE = (
    ROOT
    / "config"
    / "optimized_outer_loop_gains.json"
)


if not GAIN_FILE.exists():

    raise FileNotFoundError(
        "Could not find "
        "config/optimized_outer_loop_gains.json"
    )


gain_data = json.loads(
    GAIN_FILE.read_text()
)


# ------------------------------------------------------------
# FREEZE Kp
# ------------------------------------------------------------

KP_POS = np.asarray(
    gain_data["kp_pos"],
    dtype=float,
)


# ------------------------------------------------------------
# FREEZE Kv
# ------------------------------------------------------------

KD_POS = np.asarray(
    gain_data["kd_pos"],
    dtype=float,
)


# ------------------------------------------------------------
# START Ki FROM CURRENT ROBUST RESULT
# ------------------------------------------------------------

INITIAL_KI = np.asarray(
    gain_data.get(
        "ki_pos",
        [
            0.45,
            0.15,
            0.15,
        ],
    ),
    dtype=float,
)


# ============================================================
# INTEGRAL-GAIN SEARCH BOUNDS
# ============================================================

# Parameter order:
#
# [Kix, Kiy, Kiz]

KI_NAMES = [
    "Kix",
    "Kiy",
    "Kiz",
]


KI_LOWER = np.array(
    [
        0.00,
        0.00,
        0.00,
    ],
    dtype=float,
)


KI_UPPER = np.array(
    [
        1.20,
        1.20,
        1.50,
    ],
    dtype=float,
)


# ============================================================
# SEARCH SETTINGS
# ============================================================

INITIAL_STEP = np.array(
    [
        0.10,
        0.10,
        0.10,
    ],
    dtype=float,
)


MINIMUM_STEP = 0.025

MAX_SWEEPS = 8


# ============================================================
# PERFORMANCE COST
# ============================================================

# Success rate is NOT embedded in this numerical cost.
#
# Success rate is compared FIRST using candidate_is_better().
#
# Only when success rate is tied do these metrics decide
# which controller is better.

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
# GENERATE EXACT TRAINING CASES
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


    inertia = np.asarray(
        vehicle.params.inertia,
        dtype=float,
    )


    scale = np.asarray(
        scale,
        dtype=float,
    )


    if inertia.shape == (3,):

        vehicle.params.inertia = (
            inertia
            * scale
        )


    elif inertia.shape == (3, 3):

        modified = (
            inertia.copy()
        )


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
# RUN ONE TRAINING CASE
# ============================================================

def run_one_case(
    ki_pos,
    case,
):

    ki_pos = np.asarray(
        ki_pos,
        dtype=float,
    )


    # --------------------------------------------------------
    # NOMINAL MODEL KNOWN TO CONTROLLER
    # --------------------------------------------------------

    nominal_vehicle = (
        QuadrotorModel()
    )


    nominal_mass = float(
        nominal_vehicle.params.mass
    )


    nominal_gravity = float(
        nominal_vehicle.params.gravity
    )


    # --------------------------------------------------------
    # ACTUAL PHYSICAL PLANT
    # --------------------------------------------------------

    vehicle = (
        QuadrotorModel()
    )


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


    # Kp remains frozen.

    controller.kp_pos = (
        KP_POS.copy()
    )


    # Kv remains frozen.

    controller.kd_pos = (
        KD_POS.copy()
    )


    # ONLY Ki changes.

    controller.ki_pos = (
        ki_pos.copy()
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
    # SENSORS
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
    # CONSTANT DISTURBANCE
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

            "torque_saturation_pct":
                100.0,

            "waypoints_reached":
                0,
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

            "torque_saturation_pct":
                100.0,

            "waypoints_reached":
                0,
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
    # CROSS-TRACK RMSE
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
    # TORQUE SATURATION
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


    final_error = float(
        metrics[
            "final_error_m"
        ]
    )


    success = bool(

        mission_complete

        and waypoints_reached
        == len(
            WAYPOINTS
        )

        and final_error
        < FINAL_ERROR_LIMIT_M
    )


    # ========================================================
    # FAILURE CLASSIFICATION
    # ========================================================

    if success:

        failure_reason = (
            "none"
        )


    elif not mission_complete:

        failure_reason = (
            "mission_not_complete"
        )


    elif (
        waypoints_reached
        != len(
            WAYPOINTS
        )
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
            "unknown"
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

        "torque_saturation_pct":
            torque_saturation_pct,

        "waypoints_reached":
            waypoints_reached,
    }


# ============================================================
# SUMMARIZE A Ki CANDIDATE
# ============================================================

def summarize_candidate(
    results,
):

    success_array = np.asarray(
        [
            result[
                "success"
            ]
            for result in results
        ],
        dtype=float,
    )


    trajectory = np.asarray(
        [
            result[
                "trajectory_rmse_m"
            ]
            for result in results
        ],
        dtype=float,
    )


    cross_track = np.asarray(
        [
            result[
                "cross_track_rmse_m"
            ]
            for result in results
        ],
        dtype=float,
    )


    final_error = np.asarray(
        [
            result[
                "final_error_m"
            ]
            for result in results
        ],
        dtype=float,
    )


    torque_sat = np.asarray(
        [
            result[
                "torque_saturation_pct"
            ]
            for result in results
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
                cross_track
            )
        )

        + W_CROSS_P95
        * float(
            np.percentile(
                cross_track,
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
        result[
            "failure_reason"
        ]
        for result in results
        if not result[
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

        "performance_cost":
            float(
                performance_cost
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

        "failure_reasons":
            dict(
                failure_reasons
            ),
    }


# ============================================================
# PRIMARY COMPARISON RULE
# ============================================================

def candidate_is_better(
    candidate,
    current,
):

    # --------------------------------------------------------
    # PRIORITY 1:
    # MORE SUCCESSFUL MISSIONS
    # --------------------------------------------------------

    if (
        candidate[
            "success_rate"
        ]
        >
        current[
            "success_rate"
        ]
        + 1e-12
    ):

        return True


    if (
        candidate[
            "success_rate"
        ]
        <
        current[
            "success_rate"
        ]
        - 1e-12
    ):

        return False


    # --------------------------------------------------------
    # PRIORITY 2:
    # LOWER TRACKING / TERMINAL COST
    # --------------------------------------------------------

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
# EVALUATE ONE Ki VECTOR
# ============================================================

def evaluate_ki(
    ki_pos,
    verbose=False,
):

    results = []


    for case in TRAINING_CASES:

        result = run_one_case(
            ki_pos,
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
                f"{case['case']:02d}/"
                f"{N_TRAINING_CASES:02d}"
                f" | {status}"
                f" | WP "
                f"{result['waypoints_reached']}/5"
                f" | traj "
                f"{result['trajectory_rmse_m']:.3f} m"
                f" | cross "
                f"{result['cross_track_rmse_m']:.3f} m"
                f" | final "
                f"{result['final_error_m']:.3f} m"
                f" | "
                f"{result['failure_reason']}"
            )


    summary = summarize_candidate(
        results
    )


    return (
        summary,
        results,
    )


# ============================================================
# OPTIMIZER
# ============================================================

def optimize():

    current_ki = (
        INITIAL_KI.copy()
    )


    step = (
        INITIAL_STEP.copy()
    )


    print()

    print(
        "=" * 80
    )

    print(
        "ROBUST INTEGRAL-GAIN OPTIMIZATION"
    )

    print(
        "=" * 80
    )


    print(
        "Kp is FIXED:"
    )


    print(
        KP_POS
    )


    print()

    print(
        "Kv is FIXED:"
    )


    print(
        KD_POS
    )


    print()

    print(
        "Starting Ki:"
    )


    print(
        current_ki
    )


    print()

    print(
        "Evaluating starting Ki..."
    )


    (
        current_summary,
        current_results,
    ) = evaluate_ki(
        current_ki,
        verbose=True,
    )


    print()

    print(
        "STARTING RESULT"
    )

    print(
        "-" * 80
    )


    print(
        f"Success:            "
        f"{current_summary['successful_cases']} "
        f"/ {N_TRAINING_CASES}"
    )


    print(
        f"Trajectory mean:    "
        f"{current_summary['trajectory_mean_m']:.3f} m"
    )


    print(
        f"Final error p95:    "
        f"{current_summary['final_error_p95_m']:.3f} m"
    )


    # ========================================================
    # HISTORY
    # ========================================================

    history = []


    history.append(
        {
            "iteration":
                0,

            "ki_pos":
                current_ki.tolist(),

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
            "=" * 80
        )


        print(
            f"INTEGRAL SWEEP "
            f"{sweep}/{MAX_SWEEPS}"
        )


        print(
            "=" * 80
        )


        improved = False


        # ----------------------------------------------------
        # Kix, then Kiy, then Kiz
        # ----------------------------------------------------

        for i, name in enumerate(
            KI_NAMES
        ):

            base_value = (
                current_ki[i]
            )


            best_ki = (
                current_ki.copy()
            )


            best_summary = (
                current_summary.copy()
            )


            print()

            print(
                f"Testing {name} "
                f"around "
                f"{base_value:.3f}"
            )


            # ------------------------------------------------
            # TRY LOWER AND HIGHER
            # ------------------------------------------------

            for direction in (
                -1.0,
                +1.0,
            ):

                candidate_ki = (
                    current_ki.copy()
                )


                candidate_ki[i] = (
                    np.clip(

                        base_value
                        + direction
                        * step[i],

                        KI_LOWER[i],

                        KI_UPPER[i],
                    )
                )


                if np.isclose(
                    candidate_ki[i],
                    base_value,
                ):

                    continue


                (
                    candidate_summary,
                    candidate_results,
                ) = evaluate_ki(
                    candidate_ki,
                    verbose=False,
                )


                print(

                    f"  {name}"
                    f" = "
                    f"{candidate_ki[i]:.3f}"
                    f" | success "
                    f"{candidate_summary['successful_cases']}"
                    f"/"
                    f"{N_TRAINING_CASES}"
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

                    best_ki = (
                        candidate_ki.copy()
                    )


                    best_summary = (
                        candidate_summary.copy()
                    )


            # ------------------------------------------------
            # ACCEPT OR REJECT
            # ------------------------------------------------

            if candidate_is_better(
                best_summary,
                current_summary,
            ):

                current_ki = (
                    best_ki
                )


                current_summary = (
                    best_summary
                )


                improved = True


                print(

                    f"  ACCEPTED "
                    f"{name}"
                    f" = "
                    f"{current_ki[i]:.3f}"
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

                        "ki_pos":
                            current_ki.tolist(),

                        **current_summary,
                    }
                )


            else:

                print(
                    f"  REJECTED changes "
                    f"to {name}"
                )


        # ----------------------------------------------------
        # IF A WHOLE SWEEP STALLS, REFINE THE STEP
        # ----------------------------------------------------

        if not improved:

            step *= 0.5


            print()

            print(
                "No Ki improvement in this sweep."
            )


            print(
                "Reducing Ki search steps to:"
            )


            print(
                step
            )


        # ----------------------------------------------------
        # EARLY STOP IF ALL TRAINING CASES PASS
        #
        # We do NOT stop immediately on 10/10 because a later
        # adjustment may improve p95 while maintaining 10/10.
        # ----------------------------------------------------


        if (
            np.max(
                step
            )
            < MINIMUM_STEP
        ):

            print()

            print(
                "Minimum Ki search resolution reached."
            )

            break


    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    print()

    print(
        "=" * 80
    )


    print(
        "FINAL INTEGRAL-GAIN EVALUATION"
    )


    print(
        "=" * 80
    )


    (
        final_summary,
        final_results,
    ) = evaluate_ki(
        current_ki,
        verbose=True,
    )


    # ========================================================
    # SAVE UPDATED GAINS
    # ========================================================

    updated_gain_data = dict(
        gain_data
    )


    updated_gain_data[
        "kp_pos"
    ] = KP_POS.tolist()


    updated_gain_data[
        "kd_pos"
    ] = KD_POS.tolist()


    updated_gain_data[
        "ki_pos"
    ] = current_ki.tolist()


    updated_gain_data[
        "integral_tuning_method"
    ] = (
        "robust_integral_only_coordinate_search"
    )


    updated_gain_data[
        "integral_training_summary"
    ] = final_summary


    GAIN_FILE.write_text(

        json.dumps(
            updated_gain_data,
            indent=2,
        )
    )


    # ========================================================
    # SAVE HISTORY
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
        / "integral_gain_optimization_history.json"
    )


    history_path.write_text(

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
        "INTEGRAL OPTIMIZATION COMPLETE"
    )


    print(
        "=" * 80
    )


    print(
        f"Fixed Kp: "
        f"{KP_POS}"
    )


    print(
        f"Fixed Kv: "
        f"{KD_POS}"
    )


    print(
        f"Optimized Ki: "
        f"{current_ki}"
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


    print()

    print(
        "Saved history:"
    )


    print(
        history_path
    )


    print(
        "=" * 80
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    optimize()