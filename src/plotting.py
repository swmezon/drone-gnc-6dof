from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.simulation import SimulationResult


def _prepare_output(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def save_state_histories(
    result: SimulationResult,
    output_path: str | Path,
    title: str,
) -> None:
    """Save position, velocity, attitude, and angular-rate histories."""
    output = _prepare_output(output_path)
    t = result.time
    x = result.state

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    axes[0].plot(t, x[:, 0], label="x")
    axes[0].plot(t, x[:, 1], label="y")
    axes[0].plot(t, x[:, 2], label="z")
    axes[0].set_ylabel("Position [m]")
    axes[0].legend(ncol=3)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, x[:, 3], label="vx")
    axes[1].plot(t, x[:, 4], label="vy")
    axes[1].plot(t, x[:, 5], label="vz")
    axes[1].set_ylabel("Velocity [m/s]")
    axes[1].legend(ncol=3)
    axes[1].grid(True, alpha=0.3)

    attitude_deg = np.rad2deg(x[:, 6:9])
    axes[2].plot(t, attitude_deg[:, 0], label="roll")
    axes[2].plot(t, attitude_deg[:, 1], label="pitch")
    axes[2].plot(t, attitude_deg[:, 2], label="yaw")
    axes[2].set_ylabel("Attitude [deg]")
    axes[2].legend(ncol=3)
    axes[2].grid(True, alpha=0.3)

    rates_deg_s = np.rad2deg(x[:, 9:12])
    axes[3].plot(t, rates_deg_s[:, 0], label="p")
    axes[3].plot(t, rates_deg_s[:, 1], label="q")
    axes[3].plot(t, rates_deg_s[:, 2], label="r")
    axes[3].set_ylabel("Body rate [deg/s]")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend(ncol=3)
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_control_histories(result, output_path, title):
    """Save collective-thrust and body-torque histories."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t = result.time
    u = result.control

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
    )

    # ---------------------------------------------------------
    # Collective thrust
    # ---------------------------------------------------------
    axes[0].plot(
        t,
        u[:, 0],
        linewidth=2.0,
    )

    axes[0].set_ylabel("Thrust [N]")
    axes[0].grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # Body torques
    #
    # u[:, 1] = roll torque
    # u[:, 2] = pitch torque
    # u[:, 3] = yaw torque
    #
    # Matplotlib math text is used so tau appears as the
    # actual Greek symbol rather than the text "tau".
    # ---------------------------------------------------------
    axes[1].plot(
        t,
        u[:, 1],
        linewidth=2.0,
        label=r"$\tau_x$",
    )

    axes[1].plot(
        t,
        u[:, 2],
        linewidth=2.0,
        label=r"$\tau_y$",
    )

    axes[1].plot(
        t,
        u[:, 3],
        linewidth=2.0,
        label=r"$\tau_z$",
    )

    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel(r"Torque [N$\cdot$m]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_trajectory_3d(
    result: SimulationResult,
    output_path: str | Path,
    title: str,
) -> None:
    """Save the inertial 3-D center-of-mass trajectory."""
    output = _prepare_output(output_path)
    position = result.state[:, 0:3]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(position[:, 0], position[:, 1], position[:, 2], linewidth=2)
    ax.scatter(
        position[0, 0], position[0, 1], position[0, 2], marker="o", label="start"
    )
    ax.scatter(
        position[-1, 0],
        position[-1, 1],
        position[-1, 2],
        marker="x",
        label="end",
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
