from dataclasses import dataclass
import numpy as np


@dataclass
class ControlLimits:
    max_tilt_deg: float = 25.0
    max_rate_rad_s: float = 3.0
    max_torque: float = 1.2
    max_thrust: float = 30.0


class CascadedController:
    """
    Position -> attitude -> body-rate cascade
    for a z-up quadrotor.
    """

    def __init__(
        self,
        mass=1.5,
        gravity=9.81,
        limits=None,
    ):
        self.mass = mass
        self.g = gravity

        self.limits = (
            limits or ControlLimits()
        )

        # Position / velocity feedback gains.
        #
        # IMPORTANT:
        # These are intentionally unchanged from
        # the current baseline.
        self.kp_pos = np.array(
            [0.9, 0.9, 2.0]
        )

        self.kd_pos = np.array(
            [1.9, 1.9, 1.7]
        )

        # Attitude loop.
        self.kp_att = np.array(
            [4.0, 4.0, 2.5]
        )

        # Body-rate PID loop.
        self.kp_rate = np.array(
            [0.25, 0.25, 0.26]
        )

        self.ki_rate = np.array(
            [0.02, 0.02, 0.018]
        )

        self.kd_rate = np.array(
            [0.015, 0.015, 0.009]
        )

        self.int_rate = np.zeros(3)
        self.prev_rate_err = np.zeros(3)

    def reset(self):
        self.int_rate[:] = 0.0
        self.prev_rate_err[:] = 0.0

    def command(
        self,
        state,
        pos_ref,
        vel_ref,
        yaw_ref,
        dt,
        acc_ref=None,
    ):
        pos = state[0:3]
        vel = state[3:6]
        att = state[6:9]
        omega = state[9:12]

        if acc_ref is None:
            acc_ref = np.zeros(3)
        else:
            acc_ref = np.asarray(
                acc_ref,
                dtype=float,
            )

        # -------------------------------------------------
        # OUTER POSITION / VELOCITY LOOP
        # -------------------------------------------------

        pos_err = (
            np.asarray(pos_ref)
            - pos
        )

        vel_err = (
            np.asarray(vel_ref)
            - vel
        )

        a_feedback = (
            self.kp_pos * pos_err
            + self.kd_pos * vel_err
        )

        # Desired trajectory acceleration
        # plus feedback correction.
        a_cmd = (
            acc_ref
            + a_feedback
        )

        # -------------------------------------------------
        # ACCELERATION -> DESIRED ATTITUDE
        # -------------------------------------------------

        tilt_lim = np.deg2rad(
            self.limits.max_tilt_deg
        )

        theta_d = np.clip(
            a_cmd[0] / self.g,
            -tilt_lim,
            tilt_lim,
        )

        phi_d = np.clip(
            -a_cmd[1] / self.g,
            -tilt_lim,
            tilt_lim,
        )

        thrust = np.clip(
            self.mass
            * (
                self.g
                + a_cmd[2]
            ),
            0.0,
            self.limits.max_thrust,
        )

        att_ref = np.array(
            [
                phi_d,
                theta_d,
                yaw_ref,
            ]
        )

        # -------------------------------------------------
        # ATTITUDE LOOP
        # -------------------------------------------------

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

        rate_ref = np.clip(
            self.kp_att
            * att_err,
            -self.limits.max_rate_rad_s,
            self.limits.max_rate_rad_s,
        )

        # -------------------------------------------------
        # BODY-RATE PID LOOP
        # -------------------------------------------------

        rate_err = (
            rate_ref
            - omega
        )

        self.int_rate = np.clip(
            self.int_rate
            + rate_err * dt,
            -1.0,
            1.0,
        )

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

        tau = (
            self.kp_rate
            * rate_err

            + self.ki_rate
            * self.int_rate

            + self.kd_rate
            * rate_der
        )

        tau = np.clip(
            tau,
            -self.limits.max_torque,
            self.limits.max_torque,
        )

        return (
            np.r_[thrust, tau],
            {
                "pos_err": pos_err,
                "vel_err": vel_err,
                "acc_ref": acc_ref.copy(),
                "a_feedback": a_feedback,
                "a_cmd": a_cmd,
                "att_ref": att_ref,
                "rate_ref": rate_ref,
                "rate_err": rate_err,
            },
        )