from __future__ import annotations

from pathlib import Path
import csv
import json

from src.attitude_controller import nominal_attitude_control_gains
from src.parameters import nominal_parameters
from src.pid_autotuner import (
    AutoTuneSearch,
    StrongPortfolioTargets,
    sequential_strong_target_autotune,
)

CONTROLLER_HZ = 1000.0


def main() -> None:
    params = nominal_parameters()
    initial_gains = nominal_attitude_control_gains()
    targets = StrongPortfolioTargets()
    search = AutoTuneSearch()

    result = sequential_strong_target_autotune(
        initial_gains=initial_gains,
        params=params,
        targets=targets,
        search=search,
        controller_hz=CONTROLLER_HZ,
        t_final=3.5,
        step_time=0.5,
    )

    out_dir = Path("results/autotune")
    out_dir.mkdir(parents=True, exist_ok=True)

    gains_json = out_dir / "tuned_pid_gains.json"
    gains_json.write_text(
        json.dumps(
            {
                "converged": result.converged,
                "controller_hz": CONTROLLER_HZ,
                "attitude_kp": result.tuned_gains.attitude_kp.tolist(),
                "rate_kp": result.tuned_gains.rate_kp.tolist(),
                "rate_ki": result.tuned_gains.rate_ki.tolist(),
                "rate_kd": result.tuned_gains.rate_kd.tolist(),
                "targets": {
                    "post_transient_rmse_deg":
                        targets.post_transient_rmse_deg.tolist(),
                    "steady_state_error_deg":
                        targets.steady_state_error_deg,
                    "percent_overshoot": targets.percent_overshoot,
                    "settling_time_s": targets.settling_time_s,
                    "rise_time_min_s": targets.rise_time_min_s,
                    "rise_time_max_s": targets.rise_time_max_s,
                    "max_saturation_fraction":
                        targets.max_saturation_fraction.tolist(),
                    "require_no_sustained_oscillation":
                        targets.require_no_sustained_oscillation,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    history_csv = out_dir / "pid_autotune_history.csv"
    if result.history:
        with history_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=result.history[0].keys())
            writer.writeheader()
            writer.writerows(result.history)

    print("=== 1000 Hz Attitude-Control Optimization ===")
    print(f"Converged: {result.converged}")
    print()
    print("Tuned gains")
    print("-----------")
    print("attitude_kp =", result.tuned_gains.attitude_kp)
    print("rate_kp     =", result.tuned_gains.rate_kp)
    print("rate_ki     =", result.tuned_gains.rate_ki)
    print("rate_kd     =", result.tuned_gains.rate_kd)
    print()
    print("Final metrics")
    print("-------------")

    for axis_name in ("Roll", "Pitch", "Yaw"):
        m = result.final_metrics[axis_name]
        print(
            f"{axis_name:>5s} | "
            f"fullRMSE={m.full_rmse_deg:6.3f} deg | "
            f"postRMSE={m.post_transient_rmse_deg:6.3f} deg | "
            f"SS={m.steady_state_error_deg:6.3f} deg | "
            f"OS={m.percent_overshoot:6.2f}% | "
            f"Tr={m.rise_time_s:6.3f}s | "
            f"Ts={m.settling_time_s:6.3f}s | "
            f"sat={100*m.torque_saturation_fraction:5.2f}% | "
            f"osc={m.sustained_oscillation}"
        )


if __name__ == "__main__":
    main()
