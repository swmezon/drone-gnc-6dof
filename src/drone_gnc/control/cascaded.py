from dataclasses import dataclass

import numpy as np


# ============================================================
# CONTROLLER LIMITS
# ============================================================

@dataclass
class ControlLimits:
    max_tilt_deg: float = 25.0
    max_rate_rad_s: float = 3.0
    max_torque: float = 1.2
    max_thrust: float = 30.0


# ============================================================
# CASCADED QUADROTOR CONTROLLER
# ============================================================

class CascadedController:
    """
    Cascaded quadrotor controller.

    Outer-loop law:

        a_cmd =
            a_ref
            + Kp_pos * e_pos
            + Kv_pos * e_vel
            + a_integral

    where the integral contribution is limited directly
    in acceleration units rather than by an arbitrary
    integral-state magnitude.

    This makes disturbance-rejection authority physically
    interpretable.

    Control flow:

        position / velocity / integral error
                    |
                    v
            acceleration command
                    |
                    v
             desired attitude
                    |
                    v
              attitude loop
                    |
                    v
             body-rate loop
                    |
                    v
            thrust + torque
    """

    def __init__(
        self,
        mass=1.5,
        gravity=9.81,
        limits=None,
    ):

        self.mass = float(
            mass
        )

        self.g = float(
            gravity
        )

        self.limits = (
            limits
            if limits is not None
            else ControlLimits()
        )


        # ====================================================
        # OUTER-LOOP POSITION GAINS
        # ====================================================

        self.kp_pos = np.array(
            [
                0.90,
                0.90,
                2.00,
            ],
            dtype=float,
        )


        # ====================================================
        # OUTER-LOOP VELOCITY GAINS
        # ====================================================

        self.kd_pos = np.array(
            [
                1.90,
                1.90,
                1.70,
            ],
            dtype=float,
        )


        # ====================================================
        # OUTER-LOOP INTEGRAL GAINS
        # ====================================================

        self.ki_pos = np.array(
            [
                0.05,
                0.15,
                0.15,
            ],
            dtype=float,
        )


        # ====================================================
        # POSITION-INTEGRAL STATE
        # ====================================================

        self.int_pos = np.zeros(
            3,
            dtype=float,
        )


        # ====================================================
        # INTEGRAL CONTRIBUTION LIMIT
        # ====================================================

        # Instead of clipping int_pos directly, limit the
        # acceleration contribution:
        #
        #     a_i = Ki * integral(position error)
        #
        # Units:
        #
        #     m/s^2
        #
        # Horizontal axes are allowed up to +/- 0.60 m/s^2.
        #
        # Vertical axis is allowed up to +/- 1.30 m/s^2.
        #
        # This gives the controller enough vertical authority
        # to compensate for the current +/-10% mass mismatch
        # plus the vertical disturbance acceleration used in
        # the robustness study.

        self.integral_accel_limit = np.array(
            [
                0.60,
                0.60,
                1.30,
            ],
            dtype=float,
        )


        # ====================================================
        # ATTITUDE LOOP
        # ====================================================

        self.kp_att = np.array(
            [
                4.0,
                4.0,
                2.5,
            ],
            dtype=float,
        )


        # ====================================================
        # BODY-RATE PID GAINS
        # ====================================================

        self.kp_rate = np.array(
            [
                0.25,
                0.25,
                0.26,
            ],
            dtype=float,
        )


        self.ki_rate = np.array(
            [
                0.02,
                0.02,
                0.018,
            ],
            dtype=float,
        )


        self.kd_rate = np.array(
            [
                0.015,
                0.015,
                0.009,
            ],
            dtype=float,
        )


        # ====================================================
        # BODY-RATE INTEGRATOR
        # ====================================================

        self.int_rate = np.zeros(
            3,
            dtype=float,
        )


        self.prev_rate_err = np.zeros(
            3,
            dtype=float,
        )


    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.int_pos[:] = 0.0

        self.int_rate[:] = 0.0

        self.prev_rate_err[:] = 0.0


    # ========================================================
    # INTEGRAL CONTRIBUTION
    # ========================================================

    def _compute_integral_acceleration(
        self,
    ):

        raw_integral_accel = (
            self.ki_pos
            * self.int_pos
        )


        limited_integral_accel = np.clip(
            raw_integral_accel,
            -self.integral_accel_limit,
            self.integral_accel_limit,
        )


        return (
            raw_integral_accel,
            limited_integral_accel,
        )


    # ========================================================
    # MAIN CONTROL LAW
    # ========================================================

    def command(
        self,
        state,
        pos_ref,
        vel_ref,
        yaw_ref,
        dt,
        acc_ref=None,
    ):

        state = np.asarray(
            state,
            dtype=float,
        )


        pos_ref = np.asarray(
            pos_ref,
            dtype=float,
        )


        vel_ref = np.asarray(
            vel_ref,
            dtype=float,
        )


        # ====================================================
        # CURRENT STATE
        # ====================================================

        pos = state[
            0:3
        ]


        vel = state[
            3:6
        ]


        att = state[
            6:9
        ]


        omega = state[
            9:12
        ]


        if acc_ref is None:

            acc_ref = np.zeros(
                3,
                dtype=float,
            )

        else:

            acc_ref = np.asarray(
                acc_ref,
                dtype=float,
            )


        # ====================================================
        # POSITION ERROR
        # ====================================================

        pos_err = (
            pos_ref
            - pos
        )


        # ====================================================
        # VELOCITY ERROR
        # ====================================================

        vel_err = (
            vel_ref
            - vel
        )


        # ====================================================
        # PROPORTIONAL TERM
        # ====================================================

        a_p = (
            self.kp_pos
            * pos_err
        )


        # ====================================================
        # VELOCITY / DAMPING TERM
        # ====================================================

        a_v = (
            self.kd_pos
            * vel_err
        )


        # ====================================================
        # INTEGRAL UPDATE WITH ANTI-WINDUP
        # ====================================================

        if dt > 0.0:

            previous_int_pos = (
                self.int_pos.copy()
            )


            self.int_pos += (
                pos_err
                * dt
            )


            (
                raw_a_i,
                limited_a_i,
            ) = self._compute_integral_acceleration()


            # ------------------------------------------------
            # CONDITIONAL INTEGRATION
            # ------------------------------------------------
            #
            # If the integral contribution is already at its
            # allowed acceleration limit and the current error
            # would push it farther into saturation, undo that
            # axis's latest integration step.
            #
            # This prevents integrator windup while still
            # allowing enough persistent disturbance authority.

            for axis in range(3):

                upper_limit = (
                    self.integral_accel_limit[
                        axis
                    ]
                )


                lower_limit = (
                    -upper_limit
                )


                pushing_positive = (
                    raw_a_i[axis]
                    > upper_limit
                    and pos_err[axis]
                    > 0.0
                )


                pushing_negative = (
                    raw_a_i[axis]
                    < lower_limit
                    and pos_err[axis]
                    < 0.0
                )


                if (
                    pushing_positive
                    or pushing_negative
                ):

                    self.int_pos[
                        axis
                    ] = (
                        previous_int_pos[
                            axis
                        ]
                    )


        # ====================================================
        # FINAL INTEGRAL CONTRIBUTION
        # ====================================================

        (
            raw_a_i,
            a_i,
        ) = self._compute_integral_acceleration()


        # ====================================================
        # ACCELERATION COMMAND
        # ====================================================

        a_cmd = (

            acc_ref

            + a_p

            + a_v

            + a_i
        )


        # ====================================================
        # ACCELERATION -> DESIRED ATTITUDE
        # ====================================================

        tilt_limit = np.deg2rad(
            self.limits.max_tilt_deg
        )


        theta_d = np.clip(
            a_cmd[0]
            / self.g,
            -tilt_limit,
            tilt_limit,
        )


        phi_d = np.clip(
            -a_cmd[1]
            / self.g,
            -tilt_limit,
            tilt_limit,
        )


        # ====================================================
        # VERTICAL THRUST
        # ====================================================

        thrust_unsat = (
            self.mass
            * (
                self.g
                + a_cmd[2]
            )
        )


        thrust = np.clip(
            thrust_unsat,
            0.0,
            self.limits.max_thrust,
        )


        # ====================================================
        # THRUST-SATURATION ANTI-WINDUP
        # ====================================================

        # If the thrust actuator itself saturates and the
        # vertical integral is attempting to push farther into
        # saturation, roll back the latest Z integration step.

        if dt > 0.0:

            if (
                thrust_unsat
                > self.limits.max_thrust
                and pos_err[2] > 0.0
            ):

                self.int_pos[2] -= (
                    pos_err[2]
                    * dt
                )


            elif (
                thrust_unsat
                < 0.0
                and pos_err[2] < 0.0
            ):

                self.int_pos[2] -= (
                    pos_err[2]
                    * dt
                )


            # Recalculate integral contribution after any
            # anti-windup rollback.

            (
                raw_a_i,
                a_i,
            ) = self._compute_integral_acceleration()


        # ====================================================
        # DESIRED ATTITUDE VECTOR
        # ====================================================

        att_ref = np.array(
            [
                phi_d,
                theta_d,
                yaw_ref,
            ],
            dtype=float,
        )


        # ====================================================
        # ATTITUDE ERROR
        # ====================================================

        att_err = (
            att_ref
            - att
        )


        # Wrap yaw error to [-pi, pi].

        att_err[2] = (
            att_err[2]
            + np.pi
        ) % (
            2.0 * np.pi
        ) - np.pi


        # ====================================================
        # ATTITUDE -> BODY-RATE COMMAND
        # ====================================================

        rate_ref = np.clip(
            self.kp_att
            * att_err,
            -self.limits.max_rate_rad_s,
            self.limits.max_rate_rad_s,
        )


        # ====================================================
        # BODY-RATE ERROR
        # ====================================================

        rate_err = (
            rate_ref
            - omega
        )


        # ====================================================
        # RATE INTEGRATOR
        # ====================================================

        self.int_rate += (
            rate_err
            * dt
        )


        self.int_rate = np.clip(
            self.int_rate,
            -1.0,
            1.0,
        )


        # ====================================================
        # RATE DERIVATIVE
        # ====================================================

        rate_der = (
            rate_err
            - self.prev_rate_err
        ) / max(
            dt,
            1e-9,
        )


        self.prev_rate_err = (
            rate_err.copy()
        )


        # ====================================================
        # BODY TORQUE COMMAND
        # ====================================================

        tau_unsat = (

            self.kp_rate
            * rate_err

            + self.ki_rate
            * self.int_rate

            + self.kd_rate
            * rate_der
        )


        tau = np.clip(
            tau_unsat,
            -self.limits.max_torque,
            self.limits.max_torque,
        )


        # ====================================================
        # FINAL CONTROL VECTOR
        # ====================================================

        control = np.r_[
            thrust,
            tau,
        ]


        # ====================================================
        # DIAGNOSTICS
        # ====================================================

        info = {

            "pos_err":
                pos_err.copy(),

            "vel_err":
                vel_err.copy(),

            "int_pos":
                self.int_pos.copy(),

            "acc_ref":
                acc_ref.copy(),

            "a_p":
                a_p.copy(),

            "a_v":
                a_v.copy(),

            "a_i_raw":
                raw_a_i.copy(),

            "a_i":
                a_i.copy(),

            "a_cmd":
                a_cmd.copy(),

            "att_ref":
                att_ref.copy(),

            "rate_ref":
                rate_ref.copy(),

            "rate_err":
                rate_err.copy(),

            "thrust_unsat":
                float(
                    thrust_unsat
                ),

            "tau_unsat":
                tau_unsat.copy(),
        }


        return (
            control,
            info,
        )