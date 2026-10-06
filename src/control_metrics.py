from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TrackingMetrics:
    axis: str
    rmse_deg: float
    peak_abs_error_deg: float


@dataclass(frozen=True)
class StepResponseMetrics:
    axis: str
    full_rmse_deg: float
    post_transient_rmse_deg: float
    steady_state_error_deg: float
    percent_overshoot: float
    rise_time_s: float
    settling_time_s: float
    torque_saturation_fraction: float
    sustained_oscillation: bool


def attitude_tracking_metrics(command_rad, response_rad):
    command = np.asarray(command_rad, dtype=float)
    response = np.asarray(response_rad, dtype=float)

    axes = ("Roll", "Pitch", "Yaw")
    out = []

    for i, axis in enumerate(axes):
        err_deg = np.rad2deg(command[:, i] - response[:, i])
        out.append(
            TrackingMetrics(
                axis=axis,
                rmse_deg=float(np.sqrt(np.mean(err_deg**2))),
                peak_abs_error_deg=float(np.max(np.abs(err_deg))),
            )
        )
    return out


def _first_crossing_index(signal, threshold, direction):
    if direction > 0:
        idx = np.where(signal >= threshold)[0]
    else:
        idx = np.where(signal <= threshold)[0]
    return None if len(idx) == 0 else int(idx[0])


def compute_step_response_metrics(
    time_s,
    command_rad,
    response_rad,
    torque_nm,
    torque_limit_nm,
    *,
    step_time_s: float,
    axis_name: str,
):
    time_s = np.asarray(time_s, dtype=float)
    command_rad = np.asarray(command_rad, dtype=float)
    response_rad = np.asarray(response_rad, dtype=float)
    torque_nm = np.asarray(torque_nm, dtype=float)

    start_idx = int(np.searchsorted(time_s, step_time_s))
    t = time_s[start_idx:] - step_time_s
    cmd_deg = np.rad2deg(command_rad[start_idx:])
    y_deg = np.rad2deg(response_rad[start_idx:])

    final_cmd = float(cmd_deg[-1])
    initial_value = float(y_deg[0])
    step_size = final_cmd - initial_value
    error_deg = cmd_deg - y_deg

    full_rmse = float(np.sqrt(np.mean(error_deg**2)))
    tail_start = int(0.90 * len(error_deg))
    ss_error = float(abs(np.mean(error_deg[tail_start:])))

    if abs(step_size) < 1e-9:
        rise_time = float("nan")
        overshoot = 0.0
        idx90 = 0
    else:
        direction = 1.0 if step_size > 0 else -1.0
        idx10 = _first_crossing_index(
            y_deg,
            initial_value + 0.10 * step_size,
            direction,
        )
        idx90 = _first_crossing_index(
            y_deg,
            initial_value + 0.90 * step_size,
            direction,
        )

        rise_time = (
            float(t[idx90] - t[idx10])
            if idx10 is not None and idx90 is not None
            else float("nan")
        )

        if step_size > 0:
            overshoot = max(
                0.0,
                100.0 * (float(np.max(y_deg)) - final_cmd) / abs(step_size),
            )
        else:
            overshoot = max(
                0.0,
                100.0 * (final_cmd - float(np.min(y_deg))) / abs(step_size),
            )

    post_rmse = (
        float("inf")
        if idx90 is None
        else float(np.sqrt(np.mean(error_deg[idx90:] ** 2)))
    )

    band = max(0.1, 0.02 * abs(step_size))
    inside = np.abs(y_deg - final_cmd) <= band
    settling = float("nan")
    for i in range(len(t)):
        if np.all(inside[i:]):
            settling = float(t[i])
            break

    saturation = float(
        np.mean(
            np.abs(torque_nm[start_idx:])
            >= (abs(torque_limit_nm) - 1e-9)
        )
    )

    tail = error_deg[int(0.70 * len(error_deg)):]
    if len(tail) >= 5:
        detrended = tail - np.mean(tail)
        signs = np.sign(detrended)
        crossings = np.sum(signs[1:] * signs[:-1] < 0)
        amplitude = float(np.max(detrended) - np.min(detrended))
        oscillation = bool(crossings >= 6 and amplitude > 0.2)
    else:
        oscillation = False

    return StepResponseMetrics(
        axis=axis_name,
        full_rmse_deg=full_rmse,
        post_transient_rmse_deg=post_rmse,
        steady_state_error_deg=ss_error,
        percent_overshoot=float(overshoot),
        rise_time_s=rise_time,
        settling_time_s=settling,
        torque_saturation_fraction=saturation,
        sustained_oscillation=oscillation,
    )
