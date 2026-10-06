from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def wrap_angle_rad(angle_rad: float) -> float:
    return (float(angle_rad) + np.pi) % (2.0*np.pi) - np.pi

def wrap_attitude_error(error_rad: np.ndarray) -> np.ndarray:
    error_rad = np.asarray(error_rad, dtype=float)
    if error_rad.shape != (3,):
        raise ValueError("attitude error must have shape (3,)")
    return np.array([wrap_angle_rad(v) for v in error_rad], dtype=float)

@dataclass(frozen=True)
class AttitudeControlGains:
    attitude_kp: np.ndarray
    rate_kp: np.ndarray
    rate_ki: np.ndarray
    rate_kd: np.ndarray
    max_body_rate: np.ndarray
    max_torque: np.ndarray
    integral_limit: np.ndarray

def nominal_attitude_control_gains() -> AttitudeControlGains:
    return AttitudeControlGains(
        attitude_kp=np.array([4.0, 4.0, 2.5]),
        rate_kp=np.array([0.18, 0.18, 0.12]),
        rate_ki=np.array([0.020, 0.020, 0.010]),
        rate_kd=np.array([0.015, 0.015, 0.010]),
        max_body_rate=np.deg2rad(np.array([180.0, 180.0, 120.0])),
        max_torque=np.array([0.35, 0.35, 0.20]),
        integral_limit=np.array([0.50, 0.50, 0.50]),
    )

class CascadedAttitudeController:
    """
    Outer loop: Euler attitude error -> desired body rates.
    Inner loop: body-rate PID -> body torque command.
    """
    def __init__(self, gains: AttitudeControlGains):
        self.gains = gains
        self.rate_integral = np.zeros(3)
        self.previous_rate_error = np.zeros(3)
        self.initialized = False

    def reset(self) -> None:
        self.rate_integral[:] = 0.0
        self.previous_rate_error[:] = 0.0
        self.initialized = False

    def update(self, attitude_command_rad, attitude_rad, body_rate_rad_s, dt):
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        attitude_command_rad = np.asarray(attitude_command_rad, dtype=float)
        attitude_rad = np.asarray(attitude_rad, dtype=float)
        body_rate_rad_s = np.asarray(body_rate_rad_s, dtype=float)

        g = self.gains
        attitude_error = wrap_attitude_error(attitude_command_rad - attitude_rad)

        desired_body_rate = g.attitude_kp * attitude_error
        desired_body_rate = np.clip(
            desired_body_rate, -g.max_body_rate, g.max_body_rate
        )

        rate_error = desired_body_rate - body_rate_rad_s
        self.rate_integral += rate_error * dt
        self.rate_integral = np.clip(
            self.rate_integral, -g.integral_limit, g.integral_limit
        )

        if self.initialized:
            rate_error_dot = (rate_error - self.previous_rate_error) / dt
        else:
            rate_error_dot = np.zeros(3)
            self.initialized = True

        unsat_torque = (
            g.rate_kp * rate_error
            + g.rate_ki * self.rate_integral
            + g.rate_kd * rate_error_dot
        )
        torque = np.clip(unsat_torque, -g.max_torque, g.max_torque)

        saturated = np.abs(unsat_torque) > g.max_torque
        same_direction = np.sign(rate_error) == np.sign(unsat_torque)
        undo = saturated & same_direction
        self.rate_integral[undo] -= rate_error[undo] * dt

        self.previous_rate_error = rate_error.copy()
        return torque, desired_body_rate, attitude_error, rate_error
