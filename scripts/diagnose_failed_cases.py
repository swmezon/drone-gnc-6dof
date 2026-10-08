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


# Test the original mission duration plus longer durations.
#
# This lets us distinguish:
#
#   "controller cannot complete"
#
# from:
#
#   "controller completes, but needs more than 40 seconds"

TEST_DURATIONS = [
    40.0,
    50.0,
    60.0,
]


# ============================================================
# EXACT WAYPOINT MISSION
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
# SAME ROBUST-TRAINING RANDOMIZATION
# ============================================================

N_TRAINING_CASES = 10

TRAINING_MASTER_SEED = 314159


MASS_SCALE_MIN = 0.90
MASS_SCALE_MAX = 1.10


INERTIA_SCALE_MIN = 0.90
INERTIA_SCALE_MAX = 1.10


WIND_XY_MAX = 0.35
WIND_Z_MAX = 0.175


FINAL_ERROR_LIMIT_M = 0.25


# Cases that failed your robust-training run.

FAILED_CASE_NUMBERS = [
    1,
    7,
]


# ============================================================
# LOAD CURRENT ROBUST GAINS
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


KP_POS = np.asarray(
    gain_data["kp_pos"],
    dtype=float,
)


KD_POS = np.asarray(
    gain_data["kd_pos"],
    dtype=float,
)


KI_POS = np.asarray(
    gain_data.get(
        "ki_pos",
        [0.0, 0.0, 0.0],
    ),
    dtype=float,
)


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
# RECREATE EXACT TRAINING CASES
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

        modified = inertia.copy()

        modified[0, 0] *= scale[0]
        modified[1, 1] *= scale[1]
        modified[2, 2] *= scale[2]

        vehicle.params.inertia = (
            modified
        )


# ============================================================
# RUN ONE EXACT CASE
# ============================================================

def run_case(
    case,
    duration,
):

    # --------------------------------------------------------
    # NOMINAL CONTROLLER MODEL
    # --------------------------------------------------------

    nominal_vehicle = QuadrotorModel()


    nominal_mass = float(
        nominal_vehicle.params.mass
    )


    nominal_gravity = float(
        nominal_vehicle.params.gravity
    )


    # --------------------------------------------------------
    # PHYSICAL VEHICLE
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
    # FIXED ROBUST CONTROLLER
    # --------------------------------------------------------

    controller = CascadedController(
        nominal_mass,
        nominal_gravity,
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
    # EXACT SENSOR REALIZATION
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
    # EXACT DISTURBANCE
    # --------------------------------------------------------

    wind = ConstantWindAcceleration(
        case["wind"]
    )


    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    logs = run_waypoint_mission(
        vehicle,
        controller,
        guidance,
        sensors,
        ekf,
        duration=duration,
        dt=DT,
        gps_hz=GPS_HZ,
        wind=wind,
    )


    return (
        logs,
        vehicle,
        controller,
    )


# ============================================================
# ANALYZE ONE RUN
# ============================================================

def analyze_run(
    case,
    duration,
):

    (
        logs,
        vehicle,
        controller,
    ) = run_case(
        case,
        duration,
    )


    # ========================================================
    # BASIC MISSION STATE
    # ========================================================

    final_position = (
        logs[
            "state"
        ][-1, 0:3]
    )


    final_velocity = (
        logs[
            "state"
        ][-1, 3:6]
    )


    final_estimate = (
        logs[
            "estimate"
        ][-1, 0:3]
    )


    final_ref = (
        logs[
            "ref"
        ][-1]
    )


    final_wp_index = int(
        logs[
            "wp_index"
        ][-1]
    )


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


    current_waypoint = (
        WAYPOINTS[
            final_wp_index
        ]
    )


    distance_to_current_waypoint = float(
        np.linalg.norm(
            current_waypoint
            - final_position
        )
    )


    final_target_error = float(
        np.linalg.norm(
            WAYPOINTS[-1]
            - final_position
        )
    )


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
            waypoints_reached,
    )


    estimator = estimation_metrics(
        logs["state"],
        logs["estimate"],
    )


    # ========================================================
    # FIND FIRST COMPLETION TIME
    # ========================================================

    complete_indices = np.where(
        logs[
            "mission_complete"
        ]
    )[0]


    if len(
        complete_indices
    ) > 0:

        completion_time = float(
            logs[
                "t"
            ][
                complete_indices[0]
            ]
        )

    else:

        completion_time = None


    # ========================================================
    # DISTANCE TREND OVER LAST 5 SECONDS
    # ========================================================

    last_window_samples = int(
        round(
            5.0 / DT
        )
    )


    last_window_samples = min(
        last_window_samples,
        len(
            logs["t"]
        ),
    )


    recent_position = (
        logs[
            "state"
        ][
            -last_window_samples:,
            0:3
        ]
    )


    recent_distance = np.linalg.norm(

        recent_position
        - current_waypoint,

        axis=1,
    )


    distance_5s_ago = float(
        recent_distance[0]
    )


    distance_now = float(
        recent_distance[-1]
    )


    minimum_recent_distance = float(
        np.min(
            recent_distance
        )
    )


    distance_change = (
        distance_now
        - distance_5s_ago
    )


    # ========================================================
    # POSITION ERROR AXES
    # ========================================================

    final_position_error_vector = (
        current_waypoint
        - final_position
    )


    # ========================================================
    # INTEGRATOR STATE
    # ========================================================

    integral_state = np.asarray(
        controller.int_pos,
        dtype=float,
    )


    # ========================================================
    # DETERMINE SIMPLE DIAGNOSTIC CLASSIFICATION
    # ========================================================

    if mission_complete:

        diagnosis = (
            "COMPLETED"
        )


    elif (
        minimum_recent_distance
        <= 0.30
    ):

        diagnosis = (
            "NEAR WAYPOINT / LIKELY TIME-LIMIT ISSUE"
        )


    elif (
        distance_change
        < -0.05
    ):

        diagnosis = (
            "STILL CONVERGING WHEN TIME EXPIRED"
        )


    elif (
        abs(
            distance_change
        )
        <= 0.05
    ):

        diagnosis = (
            "LITTLE PROGRESS / POSSIBLE STEADY DISTURBANCE OFFSET"
        )


    else:

        diagnosis = (
            "MOVING AWAY FROM CURRENT WAYPOINT"
        )


    # ========================================================
    # PRINT REPORT
    # ========================================================

    print()

    print(
        "=" * 82
    )


    print(
        f"CASE {case['case']:02d} "
        f"— DURATION {duration:.1f} s"
    )


    print(
        "=" * 82
    )


    print(
        f"Diagnosis:                     "
        f"{diagnosis}"
    )


    print()

    print(
        "MISSION STATUS"
    )

    print(
        "-" * 82
    )


    print(
        f"Mission complete:              "
        f"{mission_complete}"
    )


    print(
        f"Waypoints reached:             "
        f"{waypoints_reached} / "
        f"{len(WAYPOINTS)}"
    )


    print(
        f"Current waypoint index:        "
        f"{final_wp_index + 1}"
    )


    print(
        f"Current waypoint XYZ [m]:      "
        f"{np.round(current_waypoint, 4)}"
    )


    print(
        f"Final actual XYZ [m]:          "
        f"{np.round(final_position, 4)}"
    )


    print(
        f"Final estimated XYZ [m]:       "
        f"{np.round(final_estimate, 4)}"
    )


    print(
        f"Final desired XYZ [m]:         "
        f"{np.round(final_ref, 4)}"
    )


    print(
        f"Final velocity [m/s]:          "
        f"{np.round(final_velocity, 4)}"
    )


    print(
        f"Position error vector [m]:     "
        f"{np.round(final_position_error_vector, 4)}"
    )


    print(
        f"Distance to current WP [m]:    "
        f"{distance_to_current_waypoint:.4f}"
    )


    print(
        f"Distance to final WP [m]:      "
        f"{final_target_error:.4f}"
    )


    if completion_time is None:

        print(
            "Mission completion time:       "
            "NOT COMPLETED"
        )

    else:

        print(
            f"Mission completion time [s]:   "
            f"{completion_time:.2f}"
        )


    print()

    print(
        "LAST 5-SECOND TREND"
    )

    print(
        "-" * 82
    )


    print(
        f"Distance 5 s ago [m]:          "
        f"{distance_5s_ago:.4f}"
    )


    print(
        f"Distance at end [m]:           "
        f"{distance_now:.4f}"
    )


    print(
        f"Distance change [m]:           "
        f"{distance_change:+.4f}"
    )


    print(
        f"Closest in last 5 s [m]:       "
        f"{minimum_recent_distance:.4f}"
    )


    print()

    print(
        "TRACKING / ESTIMATION"
    )

    print(
        "-" * 82
    )


    print(
        f"Trajectory RMSE [m]:           "
        f"{metrics['trajectory_position_rmse_m']:.4f}"
    )


    print(
        f"EKF position RMSE [m]:         "
        f"{estimator['ekf_position_rmse_m']:.4f}"
    )


    print()

    print(
        "UNCERTAINTY FOR THIS CASE"
    )

    print(
        "-" * 82
    )


    print(
        f"Mass scale:                    "
        f"{case['mass_scale']:.4f}"
    )


    print(
        f"Actual mass [kg]:              "
        f"{vehicle.params.mass:.4f}"
    )


    print(
        f"Inertia scales:                "
        f"{np.round(case['inertia_scale'], 4)}"
    )


    print(
        f"Wind acceleration [m/s^2]:     "
        f"{np.round(case['wind'], 4)}"
    )


    print(
        f"Sensor seed:                   "
        f"{case['sensor_seed']}"
    )


    print()

    print(
        "CONTROLLER"
    )

    print(
        "-" * 82
    )


    print(
        f"Kp:                            "
        f"{KP_POS}"
    )


    print(
        f"Kv:                            "
        f"{KD_POS}"
    )


    print(
        f"Ki:                            "
        f"{KI_POS}"
    )


    print(
        f"Final integral state [m*s]:    "
        f"{np.round(integral_state, 4)}"
    )


    return {

        "case":
            int(
                case[
                    "case"
                ]
            ),

        "duration_s":
            float(
                duration
            ),

        "mission_complete":
            mission_complete,

        "waypoints_reached":
            waypoints_reached,

        "completion_time_s":
            completion_time,

        "current_waypoint":
            int(
                final_wp_index
                + 1
            ),

        "distance_to_current_waypoint_m":
            distance_to_current_waypoint,

        "distance_to_final_waypoint_m":
            final_target_error,

        "distance_change_last_5s_m":
            float(
                distance_change
            ),

        "minimum_distance_last_5s_m":
            minimum_recent_distance,

        "trajectory_rmse_m":
            float(
                metrics[
                    "trajectory_position_rmse_m"
                ]
            ),

        "ekf_position_rmse_m":
            float(
                estimator[
                    "ekf_position_rmse_m"
                ]
            ),

        "diagnosis":
            diagnosis,

        "mass_scale":
            float(
                case[
                    "mass_scale"
                ]
            ),

        "inertia_scale":
            np.asarray(
                case[
                    "inertia_scale"
                ]
            ).tolist(),

        "wind_mps2":
            np.asarray(
                case[
                    "wind"
                ]
            ).tolist(),

        "final_position_m":
            final_position.tolist(),

        "final_reference_m":
            final_ref.tolist(),

        "integral_state":
            integral_state.tolist(),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "=" * 82
    )

    print(
        "FAILED MONTE CARLO CASE DIAGNOSTICS"
    )

    print(
        "=" * 82
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


    all_results = []


    for case_number in FAILED_CASE_NUMBERS:

        case = TRAINING_CASES[
            case_number - 1
        ]


        for duration in TEST_DURATIONS:

            result = analyze_run(
                case,
                duration,
            )


            all_results.append(
                result
            )


    # ========================================================
    # SAVE DIAGNOSTIC REPORT
    # ========================================================

    output_path = (
        ROOT
        / "results"
        / "data"
        / "failed_case_diagnostics.json"
    )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    output_path.write_text(
        json.dumps(
            all_results,
            indent=2,
        )
    )


    print()

    print(
        "=" * 82
    )

    print(
        "DIAGNOSTIC SUMMARY"
    )

    print(
        "=" * 82
    )


    for result in all_results:

        print(

            f"Case "
            f"{result['case']:02d}"
            f" | "
            f"{result['duration_s']:.0f} s"
            f" | "
            f"{result['waypoints_reached']}/"
            f"{len(WAYPOINTS)} waypoints"
            f" | "
            f"distance "
            f"{result['distance_to_current_waypoint_m']:.3f} m"
            f" | "
            f"{result['diagnosis']}"
        )


    print()

    print(
        "Saved diagnostic report:"
    )


    print(
        output_path
    )


    print(
        "=" * 82
    )


if __name__ == "__main__":

    main()