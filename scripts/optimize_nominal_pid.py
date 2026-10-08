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

from drone_gnc.simulation.mission import (
    run_waypoint_mission,
)

from drone_gnc.evaluation.metrics import (
    trajectory_metrics,
    attitude_tracking_metrics,
)


# ============================================================
# NOMINAL CONTROL-ONLY SETTINGS
# ============================================================

DT = 0.01

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
# OUTPUT FILE
# ============================================================

OUTPUT_GAIN_FILE = (
    ROOT
    / "config"
    / "nominal_pid_gains.json"
)


# ============================================================
# OPTIMIZATION VECTOR
# ============================================================

# Parameter order:
#
# [Kpx, Kpy, Kpz,
#  Kvx, Kvy, Kvz,
#  Kix, Kiy, Kiz]

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
# INITIAL GAINS
# ============================================================

INITIAL_GAINS = np.array(
    [
        1.20,
        1.20,
        2.00,

        2.00,
        2.00,
        2.00,

        0.05,
        0.05,
        0.10,
    ],
    dtype=float,
)


# ============================================================
# SEARCH BOUNDS
# ============================================================

LOWER_BOUNDS = np.array(
    [
        0.20,
        0.20,
        0.50,

        0.40,
        0.40,
        0.60,

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

        2.00,
        2.00,
        2.00,
    ],
    dtype=float,
)


# ============================================================
# SEARCH STEPS
# ============================================================

INITIAL_STEP = np.array(
    [
        0.50,
        0.50,
        0.50,

        0.50,
        0.50,
        0.50,

        0.10,
        0.10,
        0.10,
    ],
    dtype=float,
)


MINIMUM_STEP = 0.025

MAX_SWEEPS = 10


# ============================================================
# OBJECTIVE WEIGHTS
# ============================================================

# Primary objective:
#
# make the actual vehicle path stay close to the commanded
# trajectory and geometric path.

W_TRAJECTORY_RMSE = 3.0

W_CROSS_TRACK_RMSE = 5.0

W_CROSS_TRACK_MAX = 3.0

W_FINAL_ERROR = 2.0

W_ATTITUDE_RMSE = 0.05

W_TORQUE_SAT = 0.10

W_THRUST_SAT = 0.10


# ============================================================
# CROSS-TRACK HELPER
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
# RUN ONE NOMINAL CANDIDATE
# ============================================================

def evaluate_candidate(
    gains,
):

    gains = np.asarray(
        gains,
        dtype=float,
    )


    # --------------------------------------------------------
    # NOMINAL VEHICLE
    # --------------------------------------------------------

    vehicle = QuadrotorModel()


    # --------------------------------------------------------
    # CONTROLLER
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
    # PERFECT-STATE FEEDBACK
    # --------------------------------------------------------
    #
    # sensor_suite = None
    # ekf          = None
    #
    # mission.py will therefore use the true simulated state
    # directly for control feedback.

    try:

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

    except Exception:

        return {
            "cost":
                1e6,

            "success":
                False,

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

            "attitude_rmse_deg":
                100.0,

            "torque_saturation_pct":
                100.0,

            "thrust_saturation_pct":
                100.0,
        }


    # --------------------------------------------------------
    # NUMERICAL VALIDITY
    # --------------------------------------------------------

    if not np.all(
        np.isfinite(
            logs["state"]
        )
    ):

        return {
            "cost":
                1e6,

            "success":
                False,

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

            "attitude_rmse_deg":
                100.0,

            "torque_saturation_pct":
                100.0,

            "thrust_saturation_pct":
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
    # ATTITUDE METRICS
    # ========================================================

    attitude = attitude_tracking_metrics(

        logs["state"][:, 6:9],

        logs["att_ref"],
    )


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


    yaw_rmse = float(
        attitude[
            "yaw_tracking_rmse_deg"
        ]
    )


    attitude_rmse = float(
        np.sqrt(
            (
                roll_rmse**2
                + pitch_rmse**2
                + yaw_rmse**2
            )
            / 3.0
        )
    )


    # ========================================================
    # GEOMETRIC PATH
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


    cross_track_max = float(
        np.max(
            cross_errors
        )
    )


    # ========================================================
    # ACTUATOR METRICS
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
    # MISSION SUCCESS
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


    trajectory_rmse = float(
        metrics[
            "trajectory_position_rmse_m"
        ]
    )


    final_error = float(
        metrics[
            "final_error_m"
        ]
    )


    # ========================================================
    # OBJECTIVE FUNCTION
    # ========================================================

    cost = (

        W_TRAJECTORY_RMSE
        * trajectory_rmse

        + W_CROSS_TRACK_RMSE
        * cross_track_rmse

        + W_CROSS_TRACK_MAX
        * cross_track_max

        + W_FINAL_ERROR
        * final_error

        + W_ATTITUDE_RMSE
        * attitude_rmse

        + W_TORQUE_SAT
        * torque_saturation_pct

        + W_THRUST_SAT
        * thrust_saturation_pct
    )


    # Mission completion has absolute priority.

    if not success:

        missing_waypoints = (
            len(WAYPOINTS)
            - waypoints_reached
        )

        cost += (
            100.0
            + 25.0
            * missing_waypoints
        )


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
            trajectory_rmse,

        "cross_track_rmse_m":
            cross_track_rmse,

        "cross_track_max_m":
            cross_track_max,

        "final_error_m":
            final_error,

        "attitude_rmse_deg":
            attitude_rmse,

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


    return (
        candidate["cost"]
        <
        current["cost"]
    )


# ============================================================
# PRINT RESULT
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


    print(
        f"Waypoints reached:       "
        f"{result['waypoints_reached']} "
        f"/ {len(WAYPOINTS)}"
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
        f"Final waypoint error:    "
        f"{result['final_error_m']:.4f} m"
    )


    print(
        f"Attitude RMSE:           "
        f"{result['attitude_rmse_deg']:.4f} deg"
    )


    print(
        f"Torque saturation:       "
        f"{result['torque_saturation_pct']:.4f} %"
    )


    print(
        f"Thrust saturation:       "
        f"{result['thrust_saturation_pct']:.4f} %"
    )


    print(
        f"Objective cost:          "
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


    print()

    print(
        "=" * 84
    )

    print(
        "NOMINAL CONTROL-ONLY PID TRAJECTORY OPTIMIZER"
    )

    print(
        "=" * 84
    )


    print(
        "No EKF."
    )


    print(
        "No sensor noise."
    )


    print(
        "No Monte Carlo."
    )


    print(
        "No wind."
    )


    print(
        "No plant uncertainty."
    )


    print(
        "Feedback uses the true simulated vehicle state."
    )


    print()


    # ========================================================
    # INITIAL RESULT
    # ========================================================

    current_result = evaluate_candidate(
        current_gains
    )


    print(
        "INITIAL CONTROLLER"
    )


    print(
        "-" * 84
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
    # SEQUENTIAL COORDINATE SEARCH
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
            f"{sweep}/"
            f"{MAX_SWEEPS}"
        )


        print(
            "=" * 84
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


            best_result = (
                current_result.copy()
            )


            print()

            print(
                f"Testing "
                f"{name} around "
                f"{base:.3f}"
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
                    f"{candidate[i]:.3f}"
                    f" | {status}"
                    f" | WP "
                    f"{result['waypoints_reached']}/5"
                    f" | traj "
                    f"{result['trajectory_rmse_m']:.4f}"
                    f" | cross "
                    f"{result['cross_track_rmse_m']:.4f}"
                    f" | max "
                    f"{result['cross_track_max_m']:.4f}"
                    f" | final "
                    f"{result['final_error_m']:.4f}"
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
                    f"{current_gains[i]:.3f}"
                )


                history.append(
                    {
                        "iteration":
                            len(history),

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
        # STEP REFINEMENT
        # ====================================================

        if not improved:

            step *= 0.5


            print()

            print(
                "No improvement in this sweep."
            )


            print(
                "Reducing search steps:"
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

            print()

            print(
                "Minimum search resolution reached."
            )

            break


    # ========================================================
    # FINAL RESULT
    # ========================================================

    final_result = evaluate_candidate(
        current_gains
    )


    print()

    print(
        "=" * 84
    )


    print(
        "NOMINAL PID OPTIMIZATION COMPLETE"
    )


    print(
        "=" * 84
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
            "nominal_perfect_state_pid_coordinate_search",

        "feedback":
            "perfect_true_state",

        "ekf_enabled":
            False,

        "sensor_noise_enabled":
            False,

        "monte_carlo_enabled":
            False,

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

        "nominal_result":
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

    history_file = (
        ROOT
        / "results"
        / "data"
        / "nominal_pid_optimization_history.json"
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
        "Saved nominal PID gains:"
    )


    print(
        OUTPUT_GAIN_FILE
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