from __future__ import annotations

from dataclasses import dataclass, field, replace
import numpy as np

from src.attitude_controller import (
    AttitudeControlGains,
    CascadedAttitudeController,
)
from src.closed_loop_simulation import simulate_closed_loop_attitude
from src.control_metrics import compute_step_response_metrics


@dataclass(frozen=True)
class StrongPortfolioTargets:
    """
    Project design targets for simulated quadrotor attitude control.

    These are portfolio targets, not universal flight-certification limits.
    """

    post_transient_rmse_deg: np.ndarray = field(
        default_factory=lambda: np.array([0.25, 0.25, 0.50], dtype=float)
    )
    steady_state_error_deg: float = 0.10
    percent_overshoot: float = 5.0
    settling_time_s: float = 1.0
    rise_time_min_s: float = 0.30
    rise_time_max_s: float = 0.70
    max_saturation_fraction: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.02, 0.05], dtype=float)
    )
    require_no_sustained_oscillation: bool = True

    def __post_init__(self) -> None:
        rmse = np.asarray(self.post_transient_rmse_deg, dtype=float)
        saturation = np.asarray(self.max_saturation_fraction, dtype=float)

        if rmse.shape != (3,):
            raise ValueError(
                "post_transient_rmse_deg must be a 3-element vector."
            )
        if saturation.shape != (3,):
            raise ValueError(
                "max_saturation_fraction must be a 3-element vector."
            )
        if np.any(rmse <= 0.0):
            raise ValueError("RMSE targets must be positive.")
        if np.any((saturation < 0.0) | (saturation > 1.0)):
            raise ValueError(
                "Saturation fractions must lie between 0 and 1."
            )

        object.__setattr__(self, "post_transient_rmse_deg", rmse)
        object.__setattr__(self, "max_saturation_fraction", saturation)

    @property
    def rmse_deg(self) -> np.ndarray:
        """
        Backward-compatible alias retained for older repository code/tests.

        New code should use `post_transient_rmse_deg`.
        """
        return self.post_transient_rmse_deg


# Backward-compatible alias for older imports.
AutoTuneTargets = StrongPortfolioTargets


@dataclass(frozen=True)
class AutoTuneSearch:
    """
    Coordinate-search settings.

    Exactly one scalar gain is changed at a time.
    Both upward and downward candidates are evaluated.
    """
    rate_kp_step: float = 1.12
    rate_ki_step: float = 1.10
    rate_kd_step: float = 1.12
    attitude_kp_step: float = 1.10
    max_iterations_per_scalar: int = 16


@dataclass
class AutoTuneResult:
    tuned_gains: AttitudeControlGains
    history: list[dict]
    converged: bool
    final_metrics: dict


def _single_axis_reference(axis, command_deg, step_time_s):
    def reference(t: float):
        command = np.zeros(3, dtype=float)
        if t >= step_time_s:
            command[axis] = np.deg2rad(command_deg)
        return command
    return reference


def _evaluate_axis(
    gains,
    axis,
    params,
    *,
    dt,
    t_final,
    step_time,
    command_deg,
):
    controller = CascadedAttitudeController(gains)
    reference = _single_axis_reference(
        axis,
        command_deg,
        step_time,
    )

    result = simulate_closed_loop_attitude(
        initial_state=np.zeros(12, dtype=float),
        attitude_reference=reference,
        controller=controller,
        params=params,
        t_final=t_final,
        dt=dt,
    )

    metric = compute_step_response_metrics(
        time_s=result.time,
        command_rad=result.attitude_command[:, axis],
        response_rad=result.state[:, 6 + axis],
        torque_nm=result.control[:, 1 + axis],
        torque_limit_nm=gains.max_torque[axis],
        step_time_s=step_time,
        axis_name=("Roll", "Pitch", "Yaw")[axis],
    )

    return result, metric


def _metric_penalty(metric, axis, targets):
    score = 0.0

    rmse_excess = max(
        0.0,
        metric.post_transient_rmse_deg
        - targets.post_transient_rmse_deg[axis],
    )
    score += 80.0 * rmse_excess**2

    ss_excess = max(
        0.0,
        metric.steady_state_error_deg
        - targets.steady_state_error_deg,
    )
    score += 100.0 * ss_excess**2

    overshoot_excess = max(
        0.0,
        metric.percent_overshoot
        - targets.percent_overshoot,
    )
    score += 2.0 * overshoot_excess**2

    if np.isfinite(metric.settling_time_s):
        settling_excess = max(
            0.0,
            metric.settling_time_s - targets.settling_time_s,
        )
        score += 30.0 * settling_excess**2
    else:
        score += 500.0

    if np.isfinite(metric.rise_time_s):
        if metric.rise_time_s < targets.rise_time_min_s:
            too_fast = (
                targets.rise_time_min_s - metric.rise_time_s
            )
            score += 250.0 * too_fast**2
        elif metric.rise_time_s > targets.rise_time_max_s:
            too_slow = (
                metric.rise_time_s - targets.rise_time_max_s
            )
            score += 80.0 * too_slow**2
    else:
        score += 500.0

    saturation_excess = max(
        0.0,
        metric.torque_saturation_fraction
        - targets.max_saturation_fraction[axis],
    )
    score += 600.0 * saturation_excess**2

    if (
        targets.require_no_sustained_oscillation
        and metric.sustained_oscillation
    ):
        score += 500.0

    return float(score)


def _metric_passes(metric, axis, targets):
    rise_ok = (
        np.isfinite(metric.rise_time_s)
        and targets.rise_time_min_s
        <= metric.rise_time_s
        <= targets.rise_time_max_s
    )
    settling_ok = (
        np.isfinite(metric.settling_time_s)
        and metric.settling_time_s <= targets.settling_time_s
    )

    return bool(
        metric.post_transient_rmse_deg
        <= targets.post_transient_rmse_deg[axis]
        and metric.steady_state_error_deg
        <= targets.steady_state_error_deg
        and metric.percent_overshoot
        <= targets.percent_overshoot
        and settling_ok
        and rise_ok
        and metric.torque_saturation_fraction
        <= targets.max_saturation_fraction[axis]
        and (
            not targets.require_no_sustained_oscillation
            or not metric.sustained_oscillation
        )
    )


def _replace_gain(gains, field_name, axis, value):
    vector = np.array(
        getattr(gains, field_name),
        dtype=float,
        copy=True,
    )
    vector[axis] = float(value)
    return replace(gains, **{field_name: vector})


def _evaluate_candidate(
    gains,
    field_name,
    axis,
    multiplier,
    params,
    targets,
    *,
    dt,
    t_final,
    step_time,
    command_deg,
):
    current_value = float(getattr(gains, field_name)[axis])
    candidate_value = max(1e-8, current_value * multiplier)

    candidate_gains = _replace_gain(
        gains,
        field_name,
        axis,
        candidate_value,
    )

    _, candidate_metric = _evaluate_axis(
        candidate_gains,
        axis,
        params,
        dt=dt,
        t_final=t_final,
        step_time=step_time,
        command_deg=command_deg,
    )

    candidate_score = _metric_penalty(
        candidate_metric,
        axis,
        targets,
    )

    return (
        candidate_gains,
        candidate_value,
        candidate_metric,
        candidate_score,
    )


def _tune_scalar(
    gains,
    field_name,
    axis,
    step_factor,
    params,
    targets,
    search,
    *,
    dt,
    t_final,
    step_time,
    command_deg,
    history,
):
    _, best_metric = _evaluate_axis(
        gains,
        axis,
        params,
        dt=dt,
        t_final=t_final,
        step_time=step_time,
        command_deg=command_deg,
    )
    best_score = _metric_penalty(best_metric, axis, targets)
    best_gains = gains

    for iteration in range(search.max_iterations_per_scalar):
        if _metric_passes(best_metric, axis, targets):
            break

        current_value = float(
            getattr(best_gains, field_name)[axis]
        )

        candidates = []
        for direction, multiplier in (
            ("down", 1.0 / step_factor),
            ("up", step_factor),
        ):
            (
                candidate_gains,
                candidate_value,
                candidate_metric,
                candidate_score,
            ) = _evaluate_candidate(
                best_gains,
                field_name,
                axis,
                multiplier,
                params,
                targets,
                dt=dt,
                t_final=t_final,
                step_time=step_time,
                command_deg=command_deg,
            )

            candidates.append(
                (
                    candidate_score,
                    direction,
                    candidate_gains,
                    candidate_value,
                    candidate_metric,
                )
            )

            history.append(
                {
                    "gain_family": field_name,
                    "axis": ("roll", "pitch", "yaw")[axis],
                    "iteration": iteration + 1,
                    "direction": direction,
                    "gain_before": current_value,
                    "gain_candidate": candidate_value,
                    "score_before": best_score,
                    "score_candidate": candidate_score,
                    "full_rmse_deg":
                        candidate_metric.full_rmse_deg,
                    "post_transient_rmse_deg":
                        candidate_metric.post_transient_rmse_deg,
                    "steady_state_error_deg":
                        candidate_metric.steady_state_error_deg,
                    "overshoot_percent":
                        candidate_metric.percent_overshoot,
                    "rise_time_s":
                        candidate_metric.rise_time_s,
                    "settling_time_s":
                        candidate_metric.settling_time_s,
                    "saturation_fraction":
                        candidate_metric.torque_saturation_fraction,
                    "sustained_oscillation":
                        candidate_metric.sustained_oscillation,
                    "accepted": False,
                }
            )

        candidates.sort(key=lambda item: item[0])
        (
            candidate_score,
            _,
            candidate_gains,
            _,
            candidate_metric,
        ) = candidates[0]

        if candidate_score + 1e-12 < best_score:
            best_gains = candidate_gains
            best_metric = candidate_metric
            best_score = candidate_score

            for row in reversed(history):
                if (
                    row["gain_family"] == field_name
                    and row["axis"]
                    == ("roll", "pitch", "yaw")[axis]
                    and row["iteration"] == iteration + 1
                    and abs(
                        row["score_candidate"] - candidate_score
                    ) < 1e-12
                ):
                    row["accepted"] = True
                    break
        else:
            break

    return best_gains, best_metric


def sequential_strong_target_autotune(
    initial_gains,
    params,
    *,
    targets=None,
    search=None,
    controller_hz=1000.0,
    t_final=3.5,
    step_time=0.5,
):
    if targets is None:
        targets = StrongPortfolioTargets()
    if search is None:
        search = AutoTuneSearch()

    if controller_hz <= 0.0:
        raise ValueError("controller_hz must be positive.")

    dt = 1.0 / controller_hz
    gains = initial_gains
    history = []

    command_deg_by_axis = (10.0, 10.0, 20.0)

    stages = (
        ("rate_kp", search.rate_kp_step),
        ("rate_ki", search.rate_ki_step),
        ("rate_kd", search.rate_kd_step),
        ("attitude_kp", search.attitude_kp_step),
    )

    final_metrics = {}

    for axis in range(3):
        for field_name, step_factor in stages:
            gains, _ = _tune_scalar(
                gains=gains,
                field_name=field_name,
                axis=axis,
                step_factor=step_factor,
                params=params,
                targets=targets,
                search=search,
                dt=dt,
                t_final=t_final,
                step_time=step_time,
                command_deg=command_deg_by_axis[axis],
                history=history,
            )

        _, final_metric = _evaluate_axis(
            gains,
            axis,
            params,
            dt=dt,
            t_final=t_final,
            step_time=step_time,
            command_deg=command_deg_by_axis[axis],
        )
        final_metrics[("Roll", "Pitch", "Yaw")[axis]] = final_metric

    converged = all(
        _metric_passes(final_metrics[name], idx, targets)
        for idx, name in enumerate(("Roll", "Pitch", "Yaw"))
    )

    return AutoTuneResult(
        tuned_gains=gains,
        history=history,
        converged=converged,
        final_metrics=final_metrics,
    )


def sequential_pid_autotune(
    initial_gains,
    initial_state,
    attitude_reference,
    params,
    *,
    targets=None,
    search=None,
    t_final=3.5,
    dt=0.001,
):
    del initial_state, attitude_reference

    return sequential_strong_target_autotune(
        initial_gains=initial_gains,
        params=params,
        targets=targets,
        search=search,
        controller_hz=1.0 / dt,
        t_final=t_final,
        step_time=0.5,
    )
