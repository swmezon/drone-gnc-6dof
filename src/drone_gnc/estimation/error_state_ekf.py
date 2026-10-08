import numpy as np

from drone_gnc.vehicles.quadrotor import (
    rotation_matrix,
)


GRAVITY_I = np.array(
    [
        0.0,
        0.0,
        -9.81,
    ],
    dtype=float,
)


# ============================================================
# HELPERS
# ============================================================

def skew(
    v,
):

    x, y, z = np.asarray(
        v,
        dtype=float,
    )


    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def wrap_angle(
    angle,
):

    return (
        angle
        + np.pi
    ) % (
        2.0
        * np.pi
    ) - np.pi


def wrap_euler(
    att,
):

    att = np.asarray(
        att,
        dtype=float,
    ).copy()


    for i in range(
        3
    ):

        att[
            i
        ] = wrap_angle(
            att[
                i
            ]
        )


    return att


def body_rates_to_euler_rates(
    att,
    omega_body,
):

    phi, theta, _ = np.asarray(
        att,
        dtype=float,
    )


    p, q, r = np.asarray(
        omega_body,
        dtype=float,
    )


    cphi = np.cos(
        phi
    )


    sphi = np.sin(
        phi
    )


    ctheta = np.cos(
        theta
    )


    if abs(
        ctheta
    ) < 1e-6:

        ctheta = (
            1e-6
            if ctheta >= 0.0
            else -1e-6
        )


    ttheta = (
        np.sin(
            theta
        )
        / ctheta
    )


    return np.array(
        [
            p
            + q
            * sphi
            * ttheta
            + r
            * cphi
            * ttheta,

            q
            * cphi
            - r
            * sphi,

            q
            * sphi
            / ctheta
            + r
            * cphi
            / ctheta,
        ],
        dtype=float,
    )


# ============================================================
# 15-STATE ERROR-STATE EKF
# ============================================================

class ErrorStateEKF15:

    def __init__(
        self,
        dt,
        gps_pos_std=0.03,
        gps_vel_std=0.08,
        accel_noise_std=0.05,
        gyro_noise_std=0.003,
        accel_bias_rw_std=0.002,
        gyro_bias_rw_std=0.0002,
    ):

        self.dt = float(
            dt
        )


        # ====================================================
        # NOMINAL STATE
        # ====================================================

        self.p = np.zeros(
            3,
            dtype=float,
        )


        self.v = np.zeros(
            3,
            dtype=float,
        )


        self.att = np.zeros(
            3,
            dtype=float,
        )


        self.ba = np.zeros(
            3,
            dtype=float,
        )


        self.bg = np.zeros(
            3,
            dtype=float,
        )


        self.omega_body = np.zeros(
            3,
            dtype=float,
        )


        # ====================================================
        # INITIAL COVARIANCE
        # ====================================================

        p0_std = 0.02

        v0_std = 0.02

        att0_std = np.deg2rad(
            0.5
        )

        ba0_std = 0.02

        bg0_std = 0.002


        initial_std = np.r_[
            np.full(
                3,
                p0_std,
            ),

            np.full(
                3,
                v0_std,
            ),

            np.full(
                3,
                att0_std,
            ),

            np.full(
                3,
                ba0_std,
            ),

            np.full(
                3,
                bg0_std,
            ),
        ]


        self.P = np.diag(
            initial_std**2
        )


        # ====================================================
        # PROCESS NOISE
        # ====================================================

        self.accel_noise_std = float(
            accel_noise_std
        )


        self.gyro_noise_std = float(
            gyro_noise_std
        )


        self.accel_bias_rw_std = float(
            accel_bias_rw_std
        )


        self.gyro_bias_rw_std = float(
            gyro_bias_rw_std
        )


        # ====================================================
        # POSITION MEASUREMENT STD
        # ====================================================
        #
        # Accept either:
        #
        # scalar:
        #     0.03
        #
        # or vector:
        #     [0.03, 0.03, 0.04]
        # ====================================================

        gps_pos_std = np.asarray(
            gps_pos_std,
            dtype=float,
        )


        if gps_pos_std.ndim == 0:

            gps_pos_std = np.full(
                3,
                float(
                    gps_pos_std
                ),
                dtype=float,
            )


        if gps_pos_std.shape != (
            3,
        ):

            raise ValueError(
                "gps_pos_std must be a scalar "
                "or a 3-element vector."
            )


        # ====================================================
        # VELOCITY MEASUREMENT STD
        # ====================================================

        gps_vel_std = np.asarray(
            gps_vel_std,
            dtype=float,
        )


        if gps_vel_std.ndim == 0:

            gps_vel_std = np.full(
                3,
                float(
                    gps_vel_std
                ),
                dtype=float,
            )


        if gps_vel_std.shape != (
            3,
        ):

            raise ValueError(
                "gps_vel_std must be a scalar "
                "or a 3-element vector."
            )


        # ====================================================
        # MEASUREMENT COVARIANCE
        # ====================================================

        self.R = np.diag(
            np.r_[
                gps_pos_std**2,
                gps_vel_std**2,
            ]
        )


        # ====================================================
        # MEASUREMENT JACOBIAN
        # ====================================================

        self.H = np.zeros(
            (
                6,
                15,
            ),
            dtype=float,
        )


        self.H[
            0:3,
            0:3,
        ] = np.eye(
            3
        )


        self.H[
            3:6,
            3:6,
        ] = np.eye(
            3
        )


        self.last_innovation = np.zeros(
            6,
            dtype=float,
        )


        self.last_gain = np.zeros(
            (
                15,
                6,
            ),
            dtype=float,
        )


    # ========================================================
    # INITIALIZE
    # ========================================================

    def initialize(
        self,
        p,
        v,
        att,
    ):

        self.p = np.asarray(
            p,
            dtype=float,
        ).copy()


        self.v = np.asarray(
            v,
            dtype=float,
        ).copy()


        self.att = wrap_euler(
            att
        )


        self.ba[:] = 0.0

        self.bg[:] = 0.0

        self.omega_body[:] = 0.0


    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(
        self,
        accel_m,
        gyro_m,
    ):

        dt = (
            self.dt
        )


        accel_m = np.asarray(
            accel_m,
            dtype=float,
        )


        gyro_m = np.asarray(
            gyro_m,
            dtype=float,
        )


        # ----------------------------------------------------
        # BIAS-CORRECTED IMU
        # ----------------------------------------------------

        f_b = (
            accel_m
            - self.ba
        )


        omega = (
            gyro_m
            - self.bg
        )


        self.omega_body = (
            omega.copy()
        )


        # ----------------------------------------------------
        # ATTITUDE
        # ----------------------------------------------------

        euler_dot = (
            body_rates_to_euler_rates(
                self.att,
                omega,
            )
        )


        self.att = wrap_euler(
            self.att
            + euler_dot
            * dt
        )


        # ----------------------------------------------------
        # ACCELERATION
        # ----------------------------------------------------

        R_bi = rotation_matrix(
            *self.att
        )


        acc_i = (
            R_bi
            @ f_b

            + GRAVITY_I
        )


        # ----------------------------------------------------
        # POSITION / VELOCITY
        # ----------------------------------------------------

        self.p = (
            self.p

            + self.v
            * dt

            + 0.5
            * acc_i
            * dt**2
        )


        self.v = (
            self.v
            + acc_i
            * dt
        )


        # ====================================================
        # ERROR-STATE JACOBIAN
        # ====================================================

        F_c = np.zeros(
            (
                15,
                15,
            ),
            dtype=float,
        )


        F_c[
            0:3,
            3:6,
        ] = np.eye(
            3
        )


        F_c[
            3:6,
            6:9,
        ] = (
            -R_bi
            @ skew(
                f_b
            )
        )


        F_c[
            3:6,
            9:12,
        ] = (
            -R_bi
        )


        F_c[
            6:9,
            6:9,
        ] = (
            -skew(
                omega
            )
        )


        F_c[
            6:9,
            12:15,
        ] = (
            -np.eye(
                3
            )
        )


        F_d = (
            np.eye(
                15
            )

            + F_c
            * dt
        )


        # ====================================================
        # PROCESS NOISE MAPPING
        # ====================================================

        G = np.zeros(
            (
                15,
                12,
            ),
            dtype=float,
        )


        G[
            3:6,
            0:3,
        ] = R_bi


        G[
            6:9,
            3:6,
        ] = np.eye(
            3
        )


        G[
            9:12,
            6:9,
        ] = np.eye(
            3
        )


        G[
            12:15,
            9:12,
        ] = np.eye(
            3
        )


        noise_variances = np.r_[
            np.full(
                3,
                self.accel_noise_std**2,
            ),

            np.full(
                3,
                self.gyro_noise_std**2,
            ),

            np.full(
                3,
                self.accel_bias_rw_std**2,
            ),

            np.full(
                3,
                self.gyro_bias_rw_std**2,
            ),
        ]


        Q_c = np.diag(
            noise_variances
        )


        Q_d = (
            G
            @ Q_c
            @ G.T
        ) * dt


        # ====================================================
        # COVARIANCE PREDICTION
        # ====================================================

        self.P = (
            F_d
            @ self.P
            @ F_d.T

            + Q_d
        )


        self.P = 0.5 * (
            self.P
            + self.P.T
        )


    # ========================================================
    # ABSOLUTE POSITION / VELOCITY UPDATE
    # ========================================================

    def update_gps(
        self,
        p_gps,
        v_gps,
    ):

        z = np.r_[
            np.asarray(
                p_gps,
                dtype=float,
            ),

            np.asarray(
                v_gps,
                dtype=float,
            ),
        ]


        h = np.r_[
            self.p,
            self.v,
        ]


        innovation = (
            z - h
        )


        S = (
            self.H
            @ self.P
            @ self.H.T

            + self.R
        )


        PHt = (
            self.P
            @ self.H.T
        )


        K = np.linalg.solve(
            S.T,
            PHt.T,
        ).T


        dx = (
            K
            @ innovation
        )


        # ----------------------------------------------------
        # INJECT ERROR STATE
        # ----------------------------------------------------

        self.p += (
            dx[
                0:3
            ]
        )


        self.v += (
            dx[
                3:6
            ]
        )


        self.att = wrap_euler(
            self.att
            + dx[
                6:9
            ]
        )


        self.ba += (
            dx[
                9:12
            ]
        )


        self.bg += (
            dx[
                12:15
            ]
        )


        self.omega_body -= (
            dx[
                12:15
            ]
        )


        # ----------------------------------------------------
        # JOSEPH FORM COVARIANCE UPDATE
        # ----------------------------------------------------

        I = np.eye(
            15
        )


        I_KH = (
            I
            - K
            @ self.H
        )


        self.P = (

            I_KH
            @ self.P
            @ I_KH.T

            + K
            @ self.R
            @ K.T
        )


        self.P = 0.5 * (
            self.P
            + self.P.T
        )


        self.last_innovation = (
            innovation.copy()
        )


        self.last_gain = (
            K.copy()
        )


        return (
            innovation,
            K,
        )


    # ========================================================
    # ESTIMATED 12-STATE VECTOR
    # ========================================================

    def state12(
        self,
    ):

        return np.r_[
            self.p,
            self.v,
            self.att,
            self.omega_body,
        ]