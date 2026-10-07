from pathlib import Path
import argparse
import csv
import json
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

from src.aircraft import nominal_aircraft
from src.trim import trim_state_control
from src.dynamics import rk4_step
from src.linearize import linearize, modal_summary
from src.stability import local_rate_loop_analysis
from src.experiment import run_mission, dispersed_aircraft

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
DATA = ROOT / "results" / "data"
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)


def save_trim_validation(p, x0, u0):
    T = np.linspace(0, 20, 2001)
    x = x0.copy()
    X = []
    dt = T[1] - T[0]
    for _ in T:
        X.append(x.copy())
        x = rk4_step(x, u0, dt, p)
    X = np.array(X)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(T, -X[:, 2], label="Altitude")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title("Open-Loop Trim Validation")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "01_trim_validation.png", dpi=180)
    plt.close(fig)


def save_eigen_analysis(A):
    eig, rows = modal_summary(A)
    with (DATA / "flying_quality_modes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["real", "imag", "wn_rad_s", "zeta"])
        w.writerows(rows)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(eig.real, eig.imag)
    ax.axvline(0, ls="--", alpha=0.4)
    ax.set_xlabel("Real [1/s]")
    ax.set_ylabel("Imaginary [rad/s]")
    ax.set_title("Trim Linearization Eigenvalues")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "02_eigenvalue_map.png", dpi=180)
    plt.close(fig)
    return eig


def analyze_rate_loops(A, B):
    roll = local_rate_loop_analysis(A[9, 9], B[9, 0], 0.20)
    pitch = local_rate_loop_analysis(A[10, 10], B[10, 1], -0.23)
    with (DATA / "stability_margins.csv").open("w", newline="") as f:
        fields = [
            "loop",
            "gain_margin_db",
            "phase_margin_deg",
            "gain_cross_rad_s",
            "phase_cross_rad_s",
            "closed_loop_bandwidth_rad_s",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({"loop": "roll-rate", **roll})
        w.writerow({"loop": "pitch-rate", **pitch})
    return roll, pitch


def save_mission_plots(mission):
    T = mission["time"]
    X = mission["truth"]
    W = mission["waypoints"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(X[:, 1], X[:, 0], label="Aircraft")
    ax.plot(W[:, 1], W[:, 0], "o--", label="Waypoints")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title("Waypoint Trajectory Guidance")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "03_waypoint_tracking.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(T, mission["pos_error"])
    ax.axhline(1.0, ls="--", alpha=0.5, label="1 m convergence threshold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("3D Position Error [m]")
    ax.set_title("IMU/GPS EKF Position Error")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "04_ekf_position_error.png", dpi=180)
    plt.close(fig)


def monte_carlo_case(i):
    rng = np.random.default_rng(1000 + i)
    wind = np.array([rng.normal(0, 2.0), rng.normal(0, 2.0), rng.normal(0, 0.5)])
    p_case = dispersed_aircraft(rng)
    r = run_mission(seed=1000 + i, t_final=12.0, wind_ned=wind, aircraft=p_case)

    finite = bool(np.all(np.isfinite(r["truth"])) and np.all(np.isfinite(r["estimate"])))
    requirement_pos = r["pos_rmse"] < 5.0
    requirement_sat = r["saturation_fraction"] < 0.10
    requirement_conv = np.isfinite(r["convergence_s"]) and r["convergence_s"] < 8.0
    passed = finite and requirement_pos and requirement_sat and requirement_conv

    return [
        i, r["pos_rmse"], r["att_rmse_deg"], r["convergence_s"],
        r["saturation_fraction"], np.linalg.norm(wind), p_case.mass,
        p_case.Jx, p_case.Jy, p_case.Jz, int(passed),
    ]


def run_monte_carlo(N, jobs=1):
    if jobs > 1:
        # Spawn works on Windows as well as Linux/macOS and keeps each trial independent.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=jobs) as pool:
            rows = pool.map(monte_carlo_case, range(N))
    else:
        rows = [monte_carlo_case(i) for i in range(N)]

    arr = np.array(rows, float)
    with (DATA / "monte_carlo_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "case", "position_rmse_m", "attitude_rmse_deg", "convergence_s",
            "saturation_fraction", "wind_speed_m_s", "mass_kg", "Jx", "Jy", "Jz", "passed",
        ])
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(arr[:, 1], bins=30)
    ax.axvline(5.0, ls="--", label="5 m RMSE requirement")
    ax.set_xlabel("Position RMSE [m]")
    ax.set_ylabel("Count")
    ax.set_title(f"Monte Carlo Navigation Performance — N={N}")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "05_monte_carlo_position_rmse.png", dpi=180)
    plt.close(fig)
    return arr


def main():
    parser = argparse.ArgumentParser(description="Run the full fixed-wing UAV GNC proof-of-concept.")
    parser.add_argument("--mc-cases", type=int, default=500, help="Monte Carlo case count (default: 500)")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel Monte Carlo workers (use 1 for easiest debugging)")
    args = parser.parse_args()

    p = nominal_aircraft()
    x0, u0, sol = trim_state_control(p)
    save_trim_validation(p, x0, u0)

    A, B = linearize(x0, u0, p)
    save_eigen_analysis(A)
    roll, pitch = analyze_rate_loops(A, B)

    mission = run_mission(seed=7)
    save_mission_plots(mission)

    arr = run_monte_carlo(args.mc_cases, jobs=args.jobs)
    pass_rate = 100.0 * np.mean(arr[:, -1])

    crossover_values = np.array([roll["gain_cross_rad_s"], pitch["gain_cross_rad_s"]])
    bandwidth_values = np.array([roll["closed_loop_bandwidth_rad_s"], pitch["closed_loop_bandwidth_rad_s"]])

    summary = {
        "ekf_states": 15,
        "position_rmse_m": mission["pos_rmse"],
        "attitude_rmse_deg": mission["att_rmse_deg"],
        "ekf_convergence_s": mission["convergence_s"],
        "minimum_gain_margin_db": min(roll["gain_margin_db"], pitch["gain_margin_db"]),
        "minimum_phase_margin_deg": min(roll["phase_margin_deg"], pitch["phase_margin_deg"]),
        "gain_crossover_range_rad_s": [float(np.min(crossover_values)), float(np.max(crossover_values))],
        "closed_loop_bandwidth_range_rad_s": [float(np.min(bandwidth_values)), float(np.max(bandwidth_values))],
        "monte_carlo_cases": int(args.mc_cases),
        "monte_carlo_pass_rate_percent": float(pass_rate),
        "monte_carlo_position_rmse_95_m": float(np.percentile(arr[:, 1], 95)),
        "monte_carlo_worst_position_rmse_m": float(np.max(arr[:, 1])),
        "requirements": {
            "position_rmse_m_max": 5.0,
            "ekf_convergence_s_max": 8.0,
            "surface_saturation_fraction_max": 0.10,
            "numerically_finite": True,
        },
        "uncertainty_model": {
            "aircraft_parameter_sigma_fraction": 0.05,
            "aircraft_parameter_clip_fraction": 0.15,
            "horizontal_wind_sigma_m_s": 2.0,
            "vertical_wind_sigma_m_s": 0.5,
            "sensor_noise_and_bias": True,
        },
        "trim_success": bool(sol.success),
        "trim_residual_norm": float(np.linalg.norm(sol.fun)),
    }
    (DATA / "resume_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
