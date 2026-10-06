from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import numpy as np

from src.attitude_controller import (
    CascadedAttitudeController,
    nominal_attitude_control_gains,
)
from src.attitude_visualization import (
    save_attitude_animation,
    save_attitude_tracking_plot,
    save_body_rate_tracking_plot,
    save_torque_command_plot,
)
from src.closed_loop_simulation import simulate_closed_loop_attitude
from src.control_metrics import attitude_tracking_metrics
from src.parameters import nominal_parameters


CONTROLLER_HZ = 1000.0
DT = 1.0 / CONTROLLER_HZ
TUNED_GAINS_PATH = Path("results/autotune/tuned_pid_gains.json")


def commanded_attitude(t: float) -> np.ndarray:
    if t < 1.0:
        cmd = [0.0, 0.0, 0.0]
    elif t < 3.0:
        cmd = [10.0, 0.0, 0.0]
    elif t < 5.0:
        cmd = [10.0, -7.0, 0.0]
    else:
        cmd = [10.0, -7.0, 20.0]
    return np.deg2rad(np.array(cmd, dtype=float))


def load_tuned_gains():
    baseline = nominal_attitude_control_gains()
    if not TUNED_GAINS_PATH.exists():
        return baseline, False

    data = json.loads(TUNED_GAINS_PATH.read_text(encoding="utf-8"))
    tuned = replace(
        baseline,
        attitude_kp=np.asarray(data.get("attitude_kp", baseline.attitude_kp), dtype=float),
        rate_kp=np.asarray(data.get("rate_kp", baseline.rate_kp), dtype=float),
        rate_ki=np.asarray(data.get("rate_ki", baseline.rate_ki), dtype=float),
        rate_kd=np.asarray(data.get("rate_kd", baseline.rate_kd), dtype=float),
    )
    return tuned, True


def run_case(gains):
    params = nominal_parameters()
    controller = CascadedAttitudeController(gains)
    return simulate_closed_loop_attitude(
        initial_state=np.zeros(12),
        attitude_reference=commanded_attitude,
        controller=controller,
        params=params,
        t_final=8.0,
        dt=DT,
    )


def metric_dict(result):
    items = attitude_tracking_metrics(
        result.attitude_command,
        result.state[:, 6:9],
    )
    return {m.axis: m.rmse_deg for m in items}


def main():
    baseline = nominal_attitude_control_gains()
    selected, tuned_found = load_tuned_gains()

    baseline_result = run_case(baseline)
    tuned_result = run_case(selected)

    before = metric_dict(baseline_result)
    after = metric_dict(tuned_result)

    print("=== 1000 Hz Closed-Loop Quadrotor Attitude Validation ===")
    print(f"Controller rate: {CONTROLLER_HZ:.0f} Hz")
    print(f"Controller dt:   {DT:.6f} s")
    print(f"Tuned gains loaded: {tuned_found}")
    print()
    print("Before vs After")
    print("---------------")

    for axis in ("Roll", "Pitch", "Yaw"):
        reduction = 100.0 * (before[axis] - after[axis]) / before[axis]
        print(
            f"{axis:>5s} | baseline={before[axis]:6.3f} deg | "
            f"tuned={after[axis]:6.3f} deg | reduction={reduction:6.1f}%"
        )

    save_attitude_tracking_plot(
        tuned_result,
        "results/figures/closed_loop_attitude_tracking.png",
    )
    save_body_rate_tracking_plot(
        tuned_result,
        "results/figures/closed_loop_body_rate_tracking.png",
    )
    save_torque_command_plot(
        tuned_result,
        "results/figures/closed_loop_torque_commands.png",
    )
    save_attitude_animation(
        tuned_result,
        "results/animations/closed_loop_attitude_control.gif",
        fps=24,
    )

    print()
    print("Regenerated plots and GIF using the 1000 Hz tuned controller.")


if __name__ == "__main__":
    main()
