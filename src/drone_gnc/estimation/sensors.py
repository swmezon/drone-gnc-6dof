from dataclasses import dataclass

import numpy as np

from drone_gnc.vehicles.quadrotor import (
    rotation_matrix,
)


# ============================================================
# SENSOR CONFIGURATION
# ============================================================

@dataclass
class SensorConfig:

    # --------------------------------------------------------
    # IMU
    # --------------------------------------------------------

    accel_noise_std: float = 0.05

    gyro_noise_std: float = 0.003


    # --------------------------------------------------------
    # ABSOLUTE POSITION SENSOR
    # --------------------------------------------------------
    #
    # This project uses a small 2 m x 2 m flight workspace.
    #
    # A 0.45 m conventional-GPS position standard deviation is
    # not appropriate for trajectory-control evaluation at this
    # physical scale.
    #
    # The new absolute-position sensor model is:
    #
    #     sigma_x = 0.03 m
    #     sigma_y = 0.03 m
    #     sigma_z = 0.04 m
    #
    # The model may represent a high-accuracy positioning
    # source such as RTK/UWB/motion-tracking-class absolute
    # position measurements rather than conventional GPS.
    # --------------------------------------------------------

    position_noise_std_x: float = 0.03

    position_noise_std_y: float = 0.03

    position_noise_std_z: float = 0.04


    # --------------------------------------------------------
    # ABSOLUTE VELOCITY MEASUREMENT
    # --------------------------------------------------------
    #
    # Keep the previous velocity assumption unchanged for this
    # experiment so we isolate the effect of position accuracy.
    # --------------------------------------------------------

    gps_vel_std: float = 0.08


    # --------------------------------------------------------
    # IMU BIAS RANDOM WALKS
    # --------------------------------------------------------

    accel_bias_rw_std: float = 0.002

    gyro_bias_rw_std: float = 0.0002


    # --------------------------------------------------------
    # CONVENIENCE PROPERTY
    # --------------------------------------------------------

    @property
    def gps_pos_std(self):

        return np.array(
            [
                self.position_noise_std_x,
                self.position_noise_std_y,
                self.position_noise_std_z,
            ],
            dtype=float,
        )


# ============================================================
# IMU + ABSOLUTE-POSITION SENSOR SUITE
# ============================================================

class ImuGpsSensorSuite:

    def __init__(
        self,
        config=None,
        seed=42,
    ):

        self.cfg = (
            config
            if config is not None
            else SensorConfig()
        )


        self.rng = np.random.default_rng(
            seed
        )


        # True simulated sensor biases.

        self.ba = np.zeros(
            3,
            dtype=float,
        )


        self.bg = np.zeros(
            3,
            dtype=float,
        )


    # ========================================================
    # IMU
    # ========================================================

    def imu(
        self,
        state,
        acc_i,
        dt,
    ):

        c = self.cfg


        # ----------------------------------------------------
        # BIAS RANDOM WALK
        # ----------------------------------------------------

        self.ba += self.rng.normal(
            0.0,
            c.accel_bias_rw_std
            * np.sqrt(
                dt
            ),
            3,
        )


        self.bg += self.rng.normal(
            0.0,
            c.gyro_bias_rw_std
            * np.sqrt(
                dt
            ),
            3,
        )


        # ----------------------------------------------------
        # BODY ROTATION MATRIX
        # ----------------------------------------------------

        R = rotation_matrix(
            *state[
                6:9
            ]
        )


        # ----------------------------------------------------
        # SPECIFIC FORCE
        # ----------------------------------------------------
        #
        # Accelerometer measures specific force:
        #
        #     f_b = R^T (a_i - g_i)
        #
        # where
        #
        #     g_i = [0, 0, -9.81]^T
        # ----------------------------------------------------

        gravity_i = np.array(
            [
                0.0,
                0.0,
                -9.81,
            ],
            dtype=float,
        )


        specific_b = (
            R.T
            @ (
                np.asarray(
                    acc_i,
                    dtype=float,
                )
                - gravity_i
            )
        )


        # ----------------------------------------------------
        # ACCELEROMETER
        # ----------------------------------------------------

        accel = (
            specific_b

            + self.ba

            + self.rng.normal(
                0.0,
                c.accel_noise_std,
                3,
            )
        )


        # ----------------------------------------------------
        # GYROSCOPE
        # ----------------------------------------------------

        gyro = (
            np.asarray(
                state[
                    9:12
                ],
                dtype=float,
            )

            + self.bg

            + self.rng.normal(
                0.0,
                c.gyro_noise_std,
                3,
            )
        )


        return (
            accel,
            gyro,
        )


    # ========================================================
    # ABSOLUTE POSITION + VELOCITY
    # ========================================================

    def gps(
        self,
        state,
    ):

        c = self.cfg


        # ----------------------------------------------------
        # POSITION
        # ----------------------------------------------------

        position_std = np.asarray(
            c.gps_pos_std,
            dtype=float,
        )


        position_noise = self.rng.normal(
            loc=0.0,
            scale=position_std,
            size=3,
        )


        position_measurement = (
            np.asarray(
                state[
                    0:3
                ],
                dtype=float,
            )
            + position_noise
        )


        # ----------------------------------------------------
        # VELOCITY
        # ----------------------------------------------------

        velocity_noise = self.rng.normal(
            loc=0.0,
            scale=c.gps_vel_std,
            size=3,
        )


        velocity_measurement = (
            np.asarray(
                state[
                    3:6
                ],
                dtype=float,
            )
            + velocity_noise
        )


        return (
            position_measurement,
            velocity_measurement,
        )