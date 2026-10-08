import json
import sys
from pathlib import Path

import numpy as np


# ============================================================
# PROJECT IMPORT PATH
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
)


# ============================================================
# FIXED TEST CONDITIONS
# ============================================================

DT = 0.01

GPS_HZ = 10.0

DURATION = 40.0


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
# OPTIMIZATION BOUNDS
# ============================================================

# Parameter order:
#
# [Kpx, Kpy, Kpz, Kvx, Kvy, Kvz]

LOWER_BOUNDS = np.array(
    [
        0.40,
        0.40,
        0.80,
        0.80,
        0.80,
        0.80,
    ],
    dtype=float,
)


UPPER_BOUNDS = np.array(
    [
        3.00,
        3.00,
        4.00,
        4.00,
        4.00,
        4.00,
    ],
    dtype=float,
)


# ============================================================
# INITIAL GAINS
# ============================================================

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
# INITIAL SEARCH STEP SIZE
# ============================================================

INITIAL_STEP = np.array(
    [
        0.20,
        0.20,
        0.20,
        0.20,
        0.20,
        0.20,
    ],
    dtype=float,
)


# ============================================================
# OBJECTIVE WEIGHTS
# ============================================================

W_TRAJECTORY = 1.00

W_CROSS_TRACK = 0.75

W_FINAL = 0.50

W_TORQUE_SAT = 0.02


# ============================================================
# CROSS-TRACK GEOMETRY
# ============================================================

def point_segment_distance(
    p,
    a,
    b,
):

    ab = b - a

    den = float(ab @ ab)

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
# RUN ONE CANDIDATE CONTROLLER
# ============================================================

def evaluate_candidate(
    gains,
    sensor_seed=24,
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
    # CONTROLLER
    # --------------------------------------------------------

    controller = CascadedController(
        vehicle.params.mass,
        vehicle.params.gravity,
    )

    # Candidate gains supplied by optimizer.

    controller.kp_pos = np.array(
        gains[0:3],
        dtype=float,
    )

    controller.kd_pos = np.array(
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
    # SENSORS / EKF
    # --------------------------------------------------------

    sensors = ImuGpsSensorSuite(
        seed=sensor_seed,
    )

    ekf = ErrorStateEKF15(
        dt=DT,
    )


    # --------------------------------------------------------
    # RUN CLOSED-LOOP MISSION
    # --------------------------------------------------------

    logs = run_waypoint_mission(
        vehicle,
        controller,
        guidance,
        sensors,
        ekf,
        duration=DURATION,
        dt=DT,
        gps_hz=GPS_HZ,
    )


    # --------------------------------------------------------
    # TRAJECTORY METRICS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # CROSS-TRACK METRIC
    # --------------------------------------------------------

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

        error = min(

            point_segment_distance(
                position,
                path_points[j],
                path_points[j + 1],
            )

            for j in range(
                len(path_points) - 1
            )
        )

        path_error.append(
            error
        )


    path_error = np.asarray(
        path_error
    )


    cross_track_rmse = float(

        np.sqrt(

            np.mean(
                path_error**2
            )
        )
    )


    # --------------------------------------------------------
    # TORQUE SATURATION
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # MISSION SUCCESS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # OBJECTIVE FUNCTION
    # --------------------------------------------------------

    trajectory_rmse = (
        metrics[
            "trajectory_position_rmse_m"
        ]
    )

    final_error = (
        metrics[
            "final_error_m"
        ]
    )


    cost = (

        W_TRAJECTORY
        * trajectory_rmse

        + W_CROSS_TRACK
        * cross_track_rmse

        + W_FINAL
        * final_error

        + W_TORQUE_SAT
        * torque_saturation_pct
    )


    # Massive penalty for mission failure.

    if (
        not mission_complete
        or waypoints_reached
        < len(WAYPOINTS)
    ):

        cost += 100.0


    return {

        "cost":
            float(cost),

        "trajectory_rmse_m":
            float(
                trajectory_rmse
            ),

        "cross_track_rmse_m":
            float(
                cross_track_rmse
            ),

        "final_error_m":
            float(
                final_error
            ),

        "torque_saturation_pct":
            float(
                torque_saturation_pct
            ),

        "waypoints_reached":
            int(
                waypoints_reached
            ),

        "mission_complete":
            bool(
                mission_complete
            ),
    }


# ============================================================
# SEQUENTIAL COORDINATE SEARCH
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
        "=" * 78
    )

    print(
        "OUTER-LOOP GAIN OPTIMIZATION"
    )

    print(
        "=" * 78
    )


    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    current_result = (
        evaluate_candidate(
            current_gains
        )
    )


    print(
        "\nINITIAL CONTROLLER"
    )

    print(
        "-" * 78
    )

    print(
        "Gains:"
    )

    print(
        current_gains
    )

    print(
        f"Cost:               "
        f"{current_result['cost']:.6f}"
    )

    print(
        f"Trajectory RMSE:    "
        f"{current_result['trajectory_rmse_m']:.4f} m"
    )

    print(
        f"Cross-track RMSE:   "
        f"{current_result['cross_track_rmse_m']:.4f} m"
    )

    print(
        f"Final error:        "
        f"{current_result['final_error_m']:.4f} m"
    )


    # --------------------------------------------------------
    # SEARCH SETTINGS
    # --------------------------------------------------------

    max_sweeps = 12

    minimum_step = 0.025


    parameter_names = [
        "Kpx",
        "Kpy",
        "Kpz",
        "Kvx",
        "Kvy",
        "Kvz",
    ]


    history = []


    history.append(
        {
            "iteration": 0,
            "gains":
                current_gains.tolist(),
            **current_result,
        }
    )


    # --------------------------------------------------------
    # OPTIMIZATION SWEEPS
    # --------------------------------------------------------

    for sweep in range(
        1,
        max_sweeps + 1,
    ):

        print()

        print(
            "=" * 78
        )

        print(
            f"SWEEP {sweep}"
        )

        print(
            "=" * 78
        )


        improved_this_sweep = False


        # ----------------------------------------------------
        # UPDATE ONE GAIN AT A TIME
        # ----------------------------------------------------

        for i in range(
            len(current_gains)
        ):

            base_value = (
                current_gains[i]
            )

            best_local_gains = (
                current_gains.copy()
            )

            best_local_result = (
                current_result
            )


            # Try decreasing and increasing
            # this one parameter.

            for direction in [
                -1.0,
                +1.0,
            ]:

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


                # If clipping produced no
                # actual change, skip it.

                if np.isclose(
                    candidate[i],
                    base_value,
                ):
                    continue


                result = (
                    evaluate_candidate(
                        candidate
                    )
                )


                print(
                    f"{parameter_names[i]:>3s} "
                    f"= {candidate[i]:.3f} | "
                    f"J = {result['cost']:.5f} | "
                    f"traj = "
                    f"{result['trajectory_rmse_m']:.4f} m | "
                    f"cross = "
                    f"{result['cross_track_rmse_m']:.4f} m | "
                    f"final = "
                    f"{result['final_error_m']:.4f} m"
                )


                if (
                    result["cost"]
                    < best_local_result["cost"]
                ):

                    best_local_gains = (
                        candidate.copy()
                    )

                    best_local_result = (
                        result
                    )


            # ------------------------------------------------
            # ACCEPT THE BEST CHANGE FOR THIS PARAMETER
            # ------------------------------------------------

            if (
                best_local_result["cost"]
                < current_result["cost"]
            ):

                current_gains = (
                    best_local_gains
                )

                current_result = (
                    best_local_result
                )

                improved_this_sweep = (
                    True
                )


                print(
                    f"  ACCEPTED "
                    f"{parameter_names[i]} "
                    f"= "
                    f"{current_gains[i]:.3f}"
                )


                history.append(
                    {
                        "iteration":
                            len(history),

                        "sweep":
                            sweep,

                        "parameter":
                            parameter_names[i],

                        "gains":
                            current_gains.tolist(),

                        **current_result,
                    }
                )


        # ----------------------------------------------------
        # IF NOTHING IMPROVED, REDUCE STEP SIZE
        # ----------------------------------------------------

        if not improved_this_sweep:

            step *= 0.5


            print()

            print(
                "No improvement in this sweep."
            )

            print(
                "Reducing search step sizes:"
            )

            print(
                step
            )


        # ----------------------------------------------------
        # STOP WHEN SEARCH RESOLUTION IS SMALL
        # ----------------------------------------------------

        if np.max(step) < minimum_step:

            print()

            print(
                "Minimum search step reached."
            )

            break


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()

    print(
        "=" * 78
    )

    print(
        "OPTIMIZATION COMPLETE"
    )

    print(
        "=" * 78
    )


    print(
        "\nOPTIMIZED GAINS"
    )

    print(
        "-" * 78
    )


    print(
        "kp_pos = "
        f"{current_gains[0:3]}"
    )

    print(
        "kd_pos = "
        f"{current_gains[3:6]}"
    )


    print(
        "\nFINAL PERFORMANCE"
    )

    print(
        "-" * 78
    )


    print(
        f"Objective cost:          "
        f"{current_result['cost']:.6f}"
    )

    print(
        f"Trajectory RMSE:         "
        f"{current_result['trajectory_rmse_m']:.4f} m"
    )

    print(
        f"Cross-track RMSE:        "
        f"{current_result['cross_track_rmse_m']:.4f} m"
    )

    print(
        f"Final waypoint error:    "
        f"{current_result['final_error_m']:.4f} m"
    )

    print(
        f"Torque saturation:       "
        f"{current_result['torque_saturation_pct']:.4f} %"
    )

    print(
        f"Waypoints completed:     "
        f"{current_result['waypoints_reached']} "
        f"/ {len(WAYPOINTS)}"
    )


    # ========================================================
    # SAVE OPTIMIZED GAINS
    # ========================================================

    config_dir = (
        ROOT
        / "config"
    )

    config_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    output = {

        "kp_pos":
            current_gains[
                0:3
            ].tolist(),

        "kd_pos":
            current_gains[
                3:6
            ].tolist(),

        "objective":
            current_result,

        "objective_weights": {

            "trajectory_rmse":
                W_TRAJECTORY,

            "cross_track_rmse":
                W_CROSS_TRACK,

            "final_error":
                W_FINAL,

            "torque_saturation":
                W_TORQUE_SAT,
        },
    }


    output_path = (
        config_dir
        / "optimized_outer_loop_gains.json"
    )


    output_path.write_text(

        json.dumps(
            output,
            indent=2,
        )
    )


    history_path = (

        ROOT
        / "results"
        / "data"
        / "gain_optimization_history.json"
    )


    history_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    history_path.write_text(

        json.dumps(
            history,
            indent=2,
        )
    )


    print()

    print(
        "Saved optimized gains to:"
    )

    print(
        output_path
    )


    print()

    print(
        "Saved optimization history to:"
    )

    print(
        history_path
    )


if __name__ == "__main__":

    optimize()