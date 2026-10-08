import csv
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CSV_FILE = (
    ROOT
    / "results"
    / "data"
    / "monte_carlo_2cm_runs.csv"
)

SUMMARY_FILE = (
    ROOT
    / "results"
    / "data"
    / "monte_carlo_2cm_summary.json"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "robustness"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# VERIFY INPUTS
# ============================================================

if not CSV_FILE.exists():

    raise FileNotFoundError(
        f"Missing file: {CSV_FILE}"
    )


if not SUMMARY_FILE.exists():

    raise FileNotFoundError(
        f"Missing file: {SUMMARY_FILE}"
    )


# ============================================================
# LOAD DATA
# ============================================================

with CSV_FILE.open(
    "r",
    encoding="utf-8",
    newline="",
) as file:

    rows = list(
        csv.DictReader(file)
    )


summary = json.loads(
    SUMMARY_FILE.read_text(
        encoding="utf-8"
    )
)


def column(
    name,
):

    values = []

    for row in rows:

        try:

            value = float(
                row[name]
            )

        except (
            ValueError,
            TypeError,
            KeyError,
        ):

            continue


        if np.isfinite(
            value
        ):

            values.append(
                value
            )


    return np.asarray(
        values,
        dtype=float,
    )


trajectory_rmse = column(
    "trajectory_rmse_m"
)

cross_track_rmse = column(
    "cross_track_rmse_m"
)

final_error = column(
    "final_error_m"
)

ekf_position_rmse = column(
    "ekf_position_rmse_m"
)


limits = summary[
    "performance_limits"
]


trajectory_limit = float(
    limits[
        "trajectory_rmse_m"
    ]
)

cross_track_limit = float(
    limits[
        "cross_track_rmse_m"
    ]
)

final_error_limit = float(
    limits[
        "final_error_m"
    ]
)


# ============================================================
# SAVE HELPER
# ============================================================
#
# Matplotlib/Pillow was failing when handed the complete
# Windows path.
#
# This helper changes temporarily into the destination
# directory and saves using only the short filename.
# ============================================================

def save_figure(
    fig,
    filename,
):

    original_directory = os.getcwd()

    try:

        os.chdir(
            OUTPUT_DIR
        )


        fig.savefig(
            filename,
            dpi=220,
            bbox_inches="tight",
            format="png",
        )


    finally:

        os.chdir(
            original_directory
        )


    output_path = (
        OUTPUT_DIR
        / filename
    )


    if not output_path.exists():

        raise RuntimeError(
            f"Figure was not created: "
            f"{output_path}"
        )


    return output_path


# ============================================================
# HISTOGRAM
# ============================================================

def create_histogram(
    values,
    title,
    xlabel,
    filename,
    limit=None,
):

    fig, ax = plt.subplots(
        figsize=(
            8.5,
            5.5,
        )
    )


    ax.hist(
        values,
        bins=12,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.80,
    )


    mean_value = float(
        np.mean(
            values
        )
    )

    median_value = float(
        np.median(
            values
        )
    )

    p95_value = float(
        np.percentile(
            values,
            95.0,
        )
    )

    worst_value = float(
        np.max(
            values
        )
    )


    ax.axvline(
        mean_value,
        linestyle="--",
        linewidth=2.0,
        label=(
            f"Mean = "
            f"{mean_value:.4f} m"
        ),
    )


    ax.axvline(
        p95_value,
        linestyle="-.",
        linewidth=2.0,
        label=(
            f"P95 = "
            f"{p95_value:.4f} m"
        ),
    )


    if limit is not None:

        ax.axvline(
            limit,
            linestyle=":",
            linewidth=2.5,
            label=(
                f"Limit = "
                f"{limit:.4f} m"
            ),
        )


    ax.set_title(
        title,
        fontsize=15,
        pad=12,
    )


    ax.set_xlabel(
        xlabel,
        fontsize=11,
    )


    ax.set_ylabel(
        "Number of Trials",
        fontsize=11,
    )


    ax.grid(
        True,
        alpha=0.25,
    )


    ax.legend(
        loc="upper center"
    )


    text = (
        f"N = {len(values)}\n"
        f"Mean = {mean_value:.4f} m\n"
        f"Median = {median_value:.4f} m\n"
        f"P95 = {p95_value:.4f} m\n"
        f"Worst = {worst_value:.4f} m"
    )


    ax.text(
        0.98,
        0.96,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.90,
        ),
    )


    fig.tight_layout()


    output_path = save_figure(
        fig,
        filename,
    )


    plt.close(
        fig
    )


    return output_path


# ============================================================
# INDIVIDUAL FIGURES
# ============================================================

trajectory_plot = create_histogram(
    trajectory_rmse,
    "Monte Carlo Trajectory RMSE",
    "Trajectory RMSE [m]",
    "trajectory_rmse.png",
    trajectory_limit,
)


cross_track_plot = create_histogram(
    cross_track_rmse,
    "Monte Carlo Cross-Track RMSE",
    "Cross-Track RMSE [m]",
    "cross_track_rmse.png",
    cross_track_limit,
)


final_error_plot = create_histogram(
    final_error,
    "Monte Carlo Final Waypoint Error",
    "Final Waypoint Error [m]",
    "final_waypoint_error.png",
    final_error_limit,
)


ekf_plot = create_histogram(
    ekf_position_rmse,
    "Monte Carlo EKF Position RMSE",
    "EKF Position RMSE [m]",
    "ekf_position_rmse.png",
)


# ============================================================
# SUMMARY FIGURE
# ============================================================

labels = [
    "Trajectory\nRMSE",
    "Cross-track\nRMSE",
    "Final waypoint\nerror",
    "EKF position\nRMSE",
]


means = np.array(
    [
        np.mean(
            trajectory_rmse
        ),

        np.mean(
            cross_track_rmse
        ),

        np.mean(
            final_error
        ),

        np.mean(
            ekf_position_rmse
        ),
    ]
)


p95 = np.array(
    [
        np.percentile(
            trajectory_rmse,
            95
        ),

        np.percentile(
            cross_track_rmse,
            95
        ),

        np.percentile(
            final_error,
            95
        ),

        np.percentile(
            ekf_position_rmse,
            95
        ),
    ]
)


fig, ax = plt.subplots(
    figsize=(
        9.5,
        7.0,
    )
)


x = np.arange(
    len(
        labels
    )
)

width = 0.34


mean_bars = ax.bar(
    x - width / 2,
    means,
    width,
    label="Mean",
)


p95_bars = ax.bar(
    x + width / 2,
    p95,
    width,
    label="95th percentile",
)


# ============================================================
# PROFESSIONAL TITLE
# ============================================================

ax.set_title(
    "Monte Carlo GNC Validation",
    fontsize=16,
    pad=28,
)


ax.text(
    0.5,
    1.01,
    (
        "100 trials | "
        "Position measurement sigma = 0.020 m"
    ),
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=10,
)


# ============================================================
# AXES
# ============================================================

ax.set_xticks(
    x
)


ax.set_xticklabels(
    labels,
    fontsize=10,
)


ax.set_ylabel(
    "Error [m]",
    fontsize=11,
)


ax.grid(
    True,
    axis="y",
    alpha=0.25,
)


ax.legend(
    loc="upper right"
)


# ============================================================
# VALUE LABELS
# ============================================================

for bar in mean_bars:

    value = (
        bar.get_height()
    )


    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 0.0005,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )


for bar in p95_bars:

    value = (
        bar.get_height()
    )


    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 0.0005,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )


# ============================================================
# FOOTER
# ============================================================

mission_rate = float(
    summary[
        "mission_success_rate_pct"
    ]
)


pass_rate = float(
    summary[
        "performance_pass_rate_pct"
    ]
)


fig.text(
    0.5,
    0.025,
    (
        f"Mission completion: "
        f"{mission_rate:.1f}%"
        f"  |  "
        f"Tracking-envelope pass rate: "
        f"{pass_rate:.1f}%"
    ),
    ha="center",
    fontsize=10,
)


fig.tight_layout(
    rect=[
        0.0,
        0.06,
        1.0,
        0.96,
    ]
)


summary_plot = save_figure(
    fig,
    "monte_carlo_validation_summary.png",
)


plt.close(
    fig
)


# ============================================================
# FINISHED
# ============================================================

print()

print(
    "=" * 80
)


print(
    "MONTE CARLO FIGURES GENERATED"
)


print(
    "=" * 80
)


print(
    trajectory_plot
)


print(
    cross_track_plot
)


print(
    final_error_plot
)


print(
    ekf_plot
)


print(
    summary_plot
)


print(
    "=" * 80
)