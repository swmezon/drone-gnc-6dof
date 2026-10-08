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

MASTER_SEED = 2026


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
# LOAD OPTIMIZED GAINS
# ============================================================

gain_path = (
    ROOT
    / "config"
    / "optimized_outer_loop_gains.json"
)


if not gain_path.exists():
    raise FileNotFoundError(
        "optimized_outer_loop_gains.json not found. "
        "Run optimize_outer_loop.py first."
    )


gain_data = json.loads(
    gain_path.read_text()
)


KP_POS = np.asarray(
    gain_data["kp_pos"],
    dtype=float,
)


KD_POS = np.asarray(
    gain_data["kd_pos"],
    dtype=float,
)


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
# SIMPLE WIND MODEL
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
# RUN ONE MONTE CARLO CASE
# ============================================================

def run_case(
    case_index,
    rng,
):

    # --------------------------------------------------------
    # RANDOMIZED VEHICLE MASS
    # --------------------------------------------------------

    mass_scale = rng.uniform(
        0.90,
        1.10,
    )


    vehicle = QuadrotorModel()

    nominal_mass = (
        vehicle.params.mass
    )

    vehicle.params.mass = (
        nominal_mass
        * mass_scale
    )


    # --------------------------------------------------------
    # RANDOMIZED INERTIA
    # --------------------------------------------------------

    inertia_scale = rng.uniform(
        0.90,
        1.10,
        size=3,
    )


    if hasattr(
        vehicle.params,
        "inertia",
    ):

        inertia = np.asarray(
            vehicle.params.inertia,
            dtype=float,
        )

        if inertia.ndim == 1:

            vehicle.params.inertia = (
                inertia
                * inertia_scale
            )

        elif inertia.shape == (3, 3):

            modified = (
                inertia.copy()
            )

            modified[0, 0] *= (
                inertia_scale[0]
            )

            modified[1, 1] *= (
                inertia_scale[1]
            )

            modified[2, 2] *= (
                inertia_scale[2]
            )

            vehicle.params.inertia = (
                modified
            )


    # --------------------------------------------------------
    # FIXED OPTIMIZED CONTROLLER
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # GUIDANCE
    # --------------------------------------------------------

    guidance = WaypointGuidance(
        WAYPOINTS,
        acceptance_radius=0.25,
        cruise_speed=0.60,
    )


    # --------------------------------------------------------
    # RANDOM SENSOR REALIZATION
    # --------------------------------------------------------

    sensor_seed = int(
        rng.integers(
            0,
            2**31 - 1,
        )
    )


    sensors = ImuGpsSensorSuite(
        seed=sensor_seed,
    )


    ekf = ErrorStateEKF15(
        dt=DT,
    )


    # --------------------------------------------------------
    # RANDOM WIND ACCELERATION
    # --------------------------------------------------------

    wind_accel = rng.uniform(
        -0.35,
        0.35,
        size=3,
    )


    # Keep vertical disturbance smaller.
    wind_accel[2] *= 0.5


    wind = ConstantWindAcceleration(
        wind_accel
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


    # --------------------------------------------------------
    # EKF METRICS
    # --------------------------------------------------------

    metrics.update(

        estimation_metrics(

            logs["state"],

            logs["estimate"],
        )
    )


    # --------------------------------------------------------
    # CROSS-TRACK METRICS
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
        path_error
    )


    cross_track_rmse = float(

        np.sqrt(

            np.mean(
                path_error**2
            )
        )
    )


    cross_track_max = float(

        np.max(
            path_error
        )
    )


    # --------------------------------------------------------
    # ACTUATOR METRICS
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


    # --------------------------------------------------------
    # SUCCESS CRITERION
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


    success = bool(

        mission_complete

        and waypoints_reached
        == len(WAYPOINTS)

        and metrics[
            "final_error_m"
        ] < 0.25
    )


    # --------------------------------------------------------
    # RETURN CASE RESULTS
    # --------------------------------------------------------

    return {

        "case":
            int(case_index),

        "success":
            success,

        "waypoints_reached":
            waypoints_reached,

        "trajectory_rmse_m":
            float(
                metrics[
                    "trajectory_position_rmse_m"
                ]
            ),

        "cross_track_rmse_m":
            float(
                cross_track_rmse
            ),

        "cross_track_max_m":
            float(
                cross_track_max
            ),

        "final_error_m":
            float(
                metrics[
                    "final_error_m"
                ]
            ),

        "ekf_position_rmse_m":
            float(
                metrics[
                    "ekf_position_rmse_m"
                ]
            ),

        "ekf_velocity_rmse_mps":
            float(
                metrics[
                    "ekf_velocity_rmse_mps"
                ]
            ),

        "torque_saturation_pct":
            torque_saturation_pct,

        "thrust_saturation_pct":
            thrust_saturation_pct,

        "mass_scale":
            float(
                mass_scale
            ),

        "inertia_scale_x":
            float(
                inertia_scale[0]
            ),

        "inertia_scale_y":
            float(
                inertia_scale[1]
            ),

        "inertia_scale_z":
            float(
                inertia_scale[2]
            ),

        "wind_ax_mps2":
            float(
                wind_accel[0]
            ),

        "wind_ay_mps2":
            float(
                wind_accel[1]
            ),

        "wind_az_mps2":
            float(
                wind_accel[2]
            ),

        "sensor_seed":
            int(
                sensor_seed
            ),
    }


# ============================================================
# SUMMARY HELPER
# ============================================================

def summarize_metric(
    results,
    key,
):

    values = np.asarray(
        [
            r[key]
            for r in results
        ],
        dtype=float,
    )


    return {

        "mean":
            float(
                np.mean(values)
            ),

        "median":
            float(
                np.median(values)
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
                np.max(values)
            ),
    }


# ============================================================
# MAIN MONTE CARLO LOOP
# ============================================================

def main():

    rng = np.random.default_rng(
        MASTER_SEED
    )


    print()

    print(
        "=" * 72
    )

    print(
        "QUADROTOR GNC MONTE CARLO ROBUSTNESS STUDY"
    )

    print(
        "=" * 72
    )


    print(
        f"Runs:              {N_RUNS}"
    )

    print(
        f"kp_pos:            {KP_POS}"
    )

    print(
        f"kd_pos:            {KD_POS}"
    )

    print()


    results = []


    for i in range(
        N_RUNS
    ):

        result = run_case(
            i + 1,
            rng,
        )


        results.append(
            result
        )


        status = (
            "PASS"
            if result["success"]
            else "FAIL"
        )


        print(

            f"Run {i + 1:03d}/{N_RUNS:03d} | "
            f"{status} | "

            f"traj "
            f"{result['trajectory_rmse_m']:.3f} m | "

            f"cross "
            f"{result['cross_track_rmse_m']:.3f} m | "

            f"final "
            f"{result['final_error_m']:.3f} m"
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    success_count = sum(
        r["success"]
        for r in results
    )


    success_rate = (
        100.0
        * success_count
        / len(results)
    )


    summary = {

        "runs":
            N_RUNS,

        "successful_runs":
            int(
                success_count
            ),

        "success_rate_pct":
            float(
                success_rate
            ),

        "trajectory_rmse_m":
            summarize_metric(
                results,
                "trajectory_rmse_m",
            ),

        "cross_track_rmse_m":
            summarize_metric(
                results,
                "cross_track_rmse_m",
            ),

        "final_error_m":
            summarize_metric(
                results,
                "final_error_m",
            ),

        "ekf_position_rmse_m":
            summarize_metric(
                results,
                "ekf_position_rmse_m",
            ),

        "ekf_velocity_rmse_mps":
            summarize_metric(
                results,
                "ekf_velocity_rmse_mps",
            ),

        "torque_saturation_pct":
            summarize_metric(
                results,
                "torque_saturation_pct",
            ),
    }


    # ========================================================
    # SAVE JSON
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
        / "monte_carlo_runs.json"
    ).write_text(

        json.dumps(
            results,
            indent=2,
        )
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


    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = (
        output_dir
        / "monte_carlo_runs.csv"
    )


    keys = list(
        results[0].keys()
    )


    with csv_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            ",".join(keys)
            + "\n"
        )


        for result in results:

            f.write(

                ",".join(

                    str(
                        result[key]
                    )

                    for key in keys
                )

                + "\n"
            )


    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()

    print(
        "=" * 72
    )

    print(
        "MONTE CARLO SUMMARY"
    )

    print(
        "=" * 72
    )


    print(
        f"Successful runs:             "
        f"{success_count} / {N_RUNS}"
    )


    print(
        f"Mission success rate:        "
        f"{success_rate:.1f} %"
    )


    traj = (
        summary[
            "trajectory_rmse_m"
        ]
    )


    print(
        "\nTRAJECTORY RMSE"
    )

    print(
        f"Mean:                        "
        f"{traj['mean']:.3f} m"
    )

    print(
        f"Median:                      "
        f"{traj['median']:.3f} m"
    )

    print(
        f"95th percentile:             "
        f"{traj['p95']:.3f} m"
    )

    print(
        f"Worst case:                  "
        f"{traj['worst']:.3f} m"
    )


    cross = (
        summary[
            "cross_track_rmse_m"
        ]
    )


    print(
        "\nCROSS-TRACK RMSE"
    )

    print(
        f"Mean:                        "
        f"{cross['mean']:.3f} m"
    )

    print(
        f"95th percentile:             "
        f"{cross['p95']:.3f} m"
    )

    print(
        f"Worst case:                  "
        f"{cross['worst']:.3f} m"
    )


    final = (
        summary[
            "final_error_m"
        ]
    )


    print(
        "\nFINAL WAYPOINT ERROR"
    )

    print(
        f"Mean:                        "
        f"{final['mean']:.3f} m"
    )

    print(
        f"95th percentile:             "
        f"{final['p95']:.3f} m"
    )

    print(
        f"Worst case:                  "
        f"{final['worst']:.3f} m"
    )


    ekf_pos = (
        summary[
            "ekf_position_rmse_m"
        ]
    )


    print(
        "\nEKF POSITION RMSE"
    )

    print(
        f"Mean:                        "
        f"{ekf_pos['mean']:.3f} m"
    )

    print(
        f"95th percentile:             "
        f"{ekf_pos['p95']:.3f} m"
    )


    print()

    print(
        "Saved:"
    )

    print(
        output_dir
        / "monte_carlo_runs.csv"
    )

    print(
        output_dir
        / "monte_carlo_summary.json"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":

    main()