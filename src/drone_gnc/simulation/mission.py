import numpy as np

from drone_gnc.simulation.integrators import (
    rk4_step,
)


def run_waypoint_mission(
    vehicle,
    controller,
    guidance,
    sensor_suite=None,
    ekf=None,
    duration=32.0,
    dt=0.01,
    gps_hz=10.0,
    wind=None,
):

    n = (
        int(
            round(
                duration / dt
            )
        )
        + 1
    )


    t = (
        np.arange(
            n,
            dtype=float,
        )
        * dt
    )


    # ========================================================
    # TRUE 12-STATE VEHICLE
    # ========================================================
    #
    # [
    #   x, y, z,
    #   u, v, w,
    #   phi, theta, psi,
    #   p, q, r
    # ]

    x = np.zeros(
        12,
        dtype=float,
    )


    x[2] = 0.0


    # ========================================================
    # RESET CONTROLLER / GUIDANCE
    # ========================================================

    controller.reset()


    if hasattr(
        guidance,
        "reset",
    ):

        guidance.reset(
            x[:3]
        )


    # ========================================================
    # INITIALIZE EKF
    # ========================================================

    if ekf is not None:

        ekf.initialize(
            x[
                0:3
            ],

            x[
                3:6
            ],

            x[
                6:9
            ],
        )


    # ========================================================
    # LOG STORAGE
    # ========================================================

    logs = {
        key: []
        for key in [
            "t",
            "state",
            "control",

            "ref",
            "vel_ref",
            "acc_ref",

            "att_ref",
            "rate_ref",

            "wp_index",
            "distance",
            "waypoints_reached",
            "mission_complete",

            "estimate",
            "gps_pos",
        ]
    }


    # ========================================================
    # GPS UPDATE PERIOD
    # ========================================================

    gps_period = max(
        1,
        int(
            round(
                1.0
                / (
                    gps_hz
                    * dt
                )
            )
        ),
    )


    gps_for_current_time = np.full(
        3,
        np.nan,
    )


    # ========================================================
    # MAIN SIMULATION LOOP
    # ========================================================

    for k, tk in enumerate(
        t
    ):

        # ====================================================
        # FEEDBACK STATE
        # ====================================================
        #
        # No EKF:
        #
        #     controller receives true state.
        #
        # EKF:
        #
        #     controller receives the complete EKF
        #     12-state estimate.

        if ekf is None:

            feedback = (
                x.copy()
            )

        else:

            feedback = (
                ekf.state12()
            )


        # ====================================================
        # GUIDANCE
        # ====================================================

        guidance_dt = (
            0.0
            if k == 0
            else dt
        )


        (
            pref,
            vref,
            yaw_ref,
            idx,
            dist,
        ) = guidance.update(
            feedback[
                :3
            ],
            guidance_dt,
        )


        aref = np.asarray(
            getattr(
                guidance,
                "acc_ref",
                np.zeros(
                    3
                ),
            ),
            dtype=float,
        )


        # ====================================================
        # CONTROL
        # ====================================================

        (
            u,
            ctrl_info,
        ) = controller.command(
            feedback,
            pref,
            vref,
            yaw_ref,
            dt,
            acc_ref=aref,
        )


        # ====================================================
        # CURRENT ESTIMATE FOR LOGGING
        # ====================================================

        if ekf is None:

            estimate = (
                x.copy()
            )

        else:

            estimate = (
                ekf.state12()
            )


        # ====================================================
        # LOG CURRENT TIME
        # ====================================================

        logs[
            "t"
        ].append(
            tk
        )


        logs[
            "state"
        ].append(
            x.copy()
        )


        logs[
            "control"
        ].append(
            u.copy()
        )


        logs[
            "ref"
        ].append(
            pref.copy()
        )


        logs[
            "vel_ref"
        ].append(
            vref.copy()
        )


        logs[
            "acc_ref"
        ].append(
            aref.copy()
        )


        logs[
            "att_ref"
        ].append(
            ctrl_info[
                "att_ref"
            ].copy()
        )


        logs[
            "rate_ref"
        ].append(
            ctrl_info[
                "rate_ref"
            ].copy()
        )


        logs[
            "wp_index"
        ].append(
            idx
        )


        logs[
            "distance"
        ].append(
            dist
        )


        logs[
            "waypoints_reached"
        ].append(
            int(
                getattr(
                    guidance,
                    "reached_count",
                    idx,
                )
            )
        )


        logs[
            "mission_complete"
        ].append(
            bool(
                getattr(
                    guidance,
                    "complete",
                    False,
                )
            )
        )


        logs[
            "estimate"
        ].append(
            estimate
        )


        logs[
            "gps_pos"
        ].append(
            gps_for_current_time.copy()
        )


        # Final sample is logged but not propagated.

        if k == n - 1:

            break


        # ====================================================
        # TRUE DYNAMICS AT CURRENT TIME
        # ====================================================

        xdot_nominal = (
            vehicle.derivatives(
                x,
                u,
            )
        )


        acc_i = (
            xdot_nominal[
                3:6
            ].copy()
        )


        # ====================================================
        # IMU MEASUREMENT
        # ====================================================
        #
        # IMPORTANT:
        #
        # IMU measurement is generated using:
        #
        #   current state x_k
        #   current acceleration a_k
        #
        # BEFORE propagating the true vehicle.
        #
        # This keeps truth and estimator timing consistent.

        imu_measurement = None


        if (
            sensor_suite is not None
            and ekf is not None
        ):

            imu_measurement = (
                sensor_suite.imu(
                    x,
                    acc_i,
                    dt,
                )
            )


        # ====================================================
        # PROPAGATE TRUE VEHICLE
        # ====================================================

        if wind is None:

            x_next = rk4_step(
                lambda xx, uu:
                    vehicle.derivatives(
                        xx,
                        uu,
                    ),
                x,
                u,
                dt,
            )


        else:

            wind_acc = np.asarray(
                wind.acceleration(),
                dtype=float,
            )


            def dynamics_with_wind(
                xx,
                uu,
            ):

                dx = (
                    vehicle.derivatives(
                        xx,
                        uu,
                    ).copy()
                )


                dx[
                    3:6
                ] += (
                    wind_acc
                )


                return dx


            x_next = rk4_step(
                dynamics_with_wind,
                x,
                u,
                dt,
            )


        # ====================================================
        # EKF PROPAGATION
        # ====================================================

        gps_for_current_time = np.full(
            3,
            np.nan,
        )


        if imu_measurement is not None:

            (
                accel_m,
                gyro_m,
            ) = imu_measurement


            ekf.predict(
                accel_m,
                gyro_m,
            )


            # =================================================
            # GPS UPDATE AT NEW TIME
            # =================================================
            #
            # GPS measurement belongs to x_{k+1},
            # because the EKF prediction has now advanced
            # from k to k+1.

            if (
                (
                    k + 1
                )
                % gps_period
                == 0
            ):

                (
                    gps_p,
                    gps_v,
                ) = sensor_suite.gps(
                    x_next
                )


                ekf.update_gps(
                    gps_p,
                    gps_v,
                )


                gps_for_current_time = (
                    gps_p.copy()
                )


        # ====================================================
        # COMMIT NEXT TRUE STATE
        # ====================================================

        x = (
            x_next
        )


    # ========================================================
    # CONVERT LOGS TO NUMPY ARRAYS
    # ========================================================

    for key in logs:

        logs[
            key
        ] = np.asarray(
            logs[
                key
            ]
        )


    return logs