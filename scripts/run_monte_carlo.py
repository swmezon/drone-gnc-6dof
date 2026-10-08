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
# SETTINGS
# ============================================================

DT = 0.01

GPS_HZ = 10.0

DURATION = 40.0

N_RUNS = 100


# IMPORTANT:
#
# Different from robust optimizer training seed.

VALIDATION_MASTER_SEED = 20261007


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
# UNCERTAINTY MODEL
# ============================================================

MASS_SCALE_MIN = 0.90
MASS_SCALE_MAX = 1.10

INERTIA_SCALE_MIN = 0.90
INERTIA_SCALE_MAX = 1.10

WIND_XY_MAX = 0.35
WIND_Z_MAX = 0.175

FINAL_ERROR_LIMIT_M = 0.25


# ============================================================
# LOAD ROBUST GAINS
# ============================================================

GAIN_FILE = (
    ROOT
    / "config"
    / "optimized_outer_loop_gains.json"
)


if not GAIN_FILE.exists():

    raise FileNotFoundError(
        "Run optimize_outer_loop_robust.py first."
    )


gain_data = json.loads(
    GAIN_FILE.read_text()
)


KP_POS = np.asarray(
    gain_data[
        "kp_pos"
    ],
    dtype=float,
)


KD_POS = np.asarray(
    gain_data[
        "kd_pos"
    ],
    dtype=float,
)


KI_POS = np.asarray(
    gain_data.get(
        "ki_pos",
        [
            0.0,
            0.0,
            0.0,
        ],
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
# ONE VALIDATION RUN
# ============================================================

def run_case(
    run_number,
    rng,
):

    nominal_vehicle = QuadrotorModel()

    nominal_mass = float(
        nominal_vehicle.params.mass
    )

    nominal_gravity = float(
        nominal_vehicle.params.gravity
    )


    # --------------------------------------------------------
    # RANDOM PLANT
    # --------------------------------------------------------

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


    wind_vector = np.array(
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


    vehicle = QuadrotorModel()


    vehicle.params.mass = (
        nominal_mass
        * mass_scale
    )


    apply_inertia_scale(
        vehicle,
        inertia_scale,
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
    # SENSORS / EKF
    # --------------------------------------------------------

    sensors = ImuGpsSensorSuite(
        seed=sensor_seed,
    )


    ekf = ErrorStateEKF15(
        dt=DT,
    )


    # --------------------------------------------------------
    # WIND
    # --------------------------------------------------------

    wind = ConstantWindAcceleration(
        wind_vector
    )


    # --------------------------------------------------------
    # RUN MISSION
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
        wind=wind,
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


    estimator = estimation_metrics(
        logs["state"],
        logs["estimate"],
    )


    # --------------------------------------------------------
    # CROSS-TRACK
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # ACTUATORS
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

        "run":
            int(
                run_number
            ),

        "success":
            success,

        "failure_reason":
            failure_reason,

        "waypoints_reached":
            waypoints_reached,

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

        "ekf_velocity_rmse_mps":
            float(
                estimator[
                    "ekf_velocity_rmse_mps"
                ]
            ),

        "torque_saturation_pct":
            torque_saturation_pct,

        "mass_scale":
            mass_scale,

        "inertia_x_scale":
            float(
                inertia_scale[0]
            ),

        "inertia_y_scale":
            float(
                inertia_scale[1]
            ),

        "inertia_z_scale":
            float(
                inertia_scale[2]
            ),

        "wind_x_mps2":
            float(
                wind_vector[0]
            ),

        "wind_y_mps2":
            float(
                wind_vector[1]
            ),

        "wind_z_mps2":
            float(
                wind_vector[2]
            ),

        "sensor_seed":
            sensor_seed,
    }


# ============================================================
# SUMMARY HELPER
# ============================================================

def metric_summary(
    results,
    key,
):

    values = np.asarray(
        [
            result[key]
            for result in results
        ],
        dtype=float,
    )


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

        "p95":
            float(
                np.percentile(
                    values,
                    95,
                )
            ),

        "worst":
            float(
                np.max(
                    values
                )
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    rng = np.random.default_rng(
        VALIDATION_MASTER_SEED
    )


    print()

    print(
        "=" * 78
    )

    print(
        "INDEPENDENT 100-CASE MONTE CARLO VALIDATION"
    )

    print(
        "=" * 78
    )


    print(
        f"kp_pos = {KP_POS}"
    )


    print(
        f"kd_pos = {KD_POS}"
    )


    print(
        f"ki_pos = {KI_POS}"
    )


    print()


    results = []


    for run_number in range(
        1,
        N_RUNS + 1,
    ):

        result = run_case(
            run_number,
            rng,
        )


        results.append(
            result
        )


        status = (
            "PASS"
            if result[
                "success"
            ]
            else "FAIL"
        )


        print(

            f"Run "
            f"{run_number:03d}/"
            f"{N_RUNS:03d}"
            f" | {status}"
            f" | traj "
            f"{result['trajectory_rmse_m']:.3f} m"
            f" | cross "
            f"{result['cross_track_rmse_m']:.3f} m"
            f" | final "
            f"{result['final_error_m']:.3f} m"
            f" | "
            f"{result['failure_reason']}"
        )


    successful = sum(
        result[
            "success"
        ]
        for result in results
    )


    success_rate = (
        100.0
        * successful
        / N_RUNS
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


    summary = {

        "runs":
            N_RUNS,

        "successful_runs":
            int(
                successful
            ),

        "success_rate_pct":
            float(
                success_rate
            ),

        "failure_reasons":
            dict(
                failure_reasons
            ),

        "trajectory_rmse_m":
            metric_summary(
                results,
                "trajectory_rmse_m",
            ),

        "cross_track_rmse_m":
            metric_summary(
                results,
                "cross_track_rmse_m",
            ),

        "final_error_m":
            metric_summary(
                results,
                "final_error_m",
            ),

        "ekf_position_rmse_m":
            metric_summary(
                results,
                "ekf_position_rmse_m",
            ),

        "ekf_velocity_rmse_mps":
            metric_summary(
                results,
                "ekf_velocity_rmse_mps",
            ),

        "torque_saturation_pct":
            metric_summary(
                results,
                "torque_saturation_pct",
            ),
    }


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_dir = (
        ROOT
        / "results"
        / "data"
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    (
        output_dir
        / "monte_carlo_summary.json"
    ).write_text(

        json.dumps(
            summary,
            indent=2,
        )
    )


    with (
        output_dir
        / "monte_carlo_runs.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=
                list(
                    results[0].keys()
                ),
        )


        writer.writeheader()

        writer.writerows(
            results
        )


    # ========================================================
    # REPORT
    # ========================================================

    print()

    print(
        "=" * 78
    )

    print(
        "MONTE CARLO SUMMARY"
    )

    print(
        "=" * 78
    )


    print(
        f"Successful runs:            "
        f"{successful} / {N_RUNS}"
    )


    print(
        f"Mission success rate:       "
        f"{success_rate:.1f} %"
    )


    print()

    print(
        "FAILURE REASONS"
    )

    print(
        failure_reasons
    )


    for title, key, units in [

        (
            "TRAJECTORY RMSE",
            "trajectory_rmse_m",
            "m",
        ),

        (
            "CROSS-TRACK RMSE",
            "cross_track_rmse_m",
            "m",
        ),

        (
            "FINAL WAYPOINT ERROR",
            "final_error_m",
            "m",
        ),

        (
            "EKF POSITION RMSE",
            "ekf_position_rmse_m",
            "m",
        ),

    ]:

        data = (
            summary[
                key
            ]
        )


        print()

        print(
            title
        )


        print(
            f"Mean:                       "
            f"{data['mean']:.3f} {units}"
        )


        print(
            f"Median:                     "
            f"{data['median']:.3f} {units}"
        )


        print(
            f"95th percentile:            "
            f"{data['p95']:.3f} {units}"
        )


        print(
            f"Worst case:                 "
            f"{data['worst']:.3f} {units}"
        )


    print(
        "=" * 78
    )


if __name__ == "__main__":

    main()