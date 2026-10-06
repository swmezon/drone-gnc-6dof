from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from src.rotations import body_to_inertial


EVENT_TIMES = [1.0, 3.0, 5.0]

EVENT_LABELS = [
    "Roll step",
    "Pitch step",
    "Yaw step",
]


def _prepare_output_path(
    output_path: str | Path,
) -> Path:
    """
    Prepare a safe absolute output path for Windows.

    - Converts the path to an absolute path.
    - Creates the parent directory.
    - Removes an existing file before rewriting it.
    """

    output_path = Path(output_path).resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        output_path.unlink()

    return output_path


def add_command_events(ax) -> None:
    """
    Add command-transition markers.

    t = 1 s : roll command
    t = 3 s : pitch command
    t = 5 s : yaw command
    """

    ymin, ymax = ax.get_ylim()

    for time_s, label in zip(
        EVENT_TIMES,
        EVENT_LABELS,
    ):
        ax.axvline(
            time_s,
            linestyle=":",
            linewidth=1.0,
            alpha=0.45,
        )

        ax.text(
            time_s + 0.03,
            ymax - 0.05 * (ymax - ymin),
            label,
            rotation=90,
            verticalalignment="top",
            fontsize=8,
            alpha=0.70,
        )


def save_attitude_tracking_plot(
    result,
    output_path: str | Path,
) -> None:
    """
    Save commanded-versus-actual roll, pitch, and yaw tracking.

    Command and actual signals use the same color for each axis:
        dashed = command
        solid  = actual
    """

    output_path = _prepare_output_path(
        output_path
    )

    time = result.time

    command_deg = np.rad2deg(
        result.attitude_command
    )

    actual_deg = np.rad2deg(
        result.state[:, 6:9]
    )

    fig, ax = plt.subplots(
        figsize=(10.0, 5.8)
    )

    axis_labels = [
        "Roll",
        "Pitch",
        "Yaw",
    ]

    for index, label in enumerate(
        axis_labels
    ):
        actual_line, = ax.plot(
            time,
            actual_deg[:, index],
            linewidth=2.0,
            label=f"{label} actual",
        )

        ax.plot(
            time,
            command_deg[:, index],
            linestyle="--",
            linewidth=1.6,
            color=actual_line.get_color(),
            label=f"{label} command",
        )

    ax.set_title(
        "Closed-Loop Attitude Tracking — 1000 Hz"
    )

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_ylabel(
        "Attitude [deg]"
    )

    ax.grid(
        True,
        alpha=0.20,
    )

    add_command_events(
        ax
    )

    handles, labels = ax.get_legend_handles_labels()

    desired_order = [
        "Roll command",
        "Roll actual",
        "Pitch command",
        "Pitch actual",
        "Yaw command",
        "Yaw actual",
    ]

    ordered_handles = []
    ordered_labels = []

    for desired_label in desired_order:
        for handle, label in zip(
            handles,
            labels,
        ):
            if label == desired_label:
                ordered_handles.append(
                    handle
                )
                ordered_labels.append(
                    label
                )

    ax.legend(
        ordered_handles,
        ordered_labels,
        ncol=3,
        frameon=False,
        loc="upper center",
    )

    fig.tight_layout()

    fig.savefig(
        str(output_path),
        format="png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_body_rate_tracking_plot(
    result,
    output_path: str | Path,
) -> None:
    """
    Save desired-versus-actual body angular-rate tracking.

    p = roll rate
    q = pitch rate
    r = yaw rate
    """

    output_path = _prepare_output_path(
        output_path
    )

    time = result.time

    desired_deg_s = np.rad2deg(
        result.desired_body_rate
    )

    actual_deg_s = np.rad2deg(
        result.state[:, 9:12]
    )

    fig, ax = plt.subplots(
        figsize=(10.0, 5.8)
    )

    axis_labels = [
        "p",
        "q",
        "r",
    ]

    for index, label in enumerate(
        axis_labels
    ):
        actual_line, = ax.plot(
            time,
            actual_deg_s[:, index],
            linewidth=2.0,
            label=f"{label} actual",
        )

        ax.plot(
            time,
            desired_deg_s[:, index],
            linestyle="--",
            linewidth=1.6,
            color=actual_line.get_color(),
            label=f"{label} desired",
        )

    ax.set_title(
        "Body-Rate Tracking — 1000 Hz"
    )

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_ylabel(
        "Body Angular Rate [deg/s]"
    )

    ax.grid(
        True,
        alpha=0.20,
    )

    add_command_events(
        ax
    )

    handles, labels = ax.get_legend_handles_labels()

    desired_order = [
        "p desired",
        "p actual",
        "q desired",
        "q actual",
        "r desired",
        "r actual",
    ]

    ordered_handles = []
    ordered_labels = []

    for desired_label in desired_order:
        for handle, label in zip(
            handles,
            labels,
        ):
            if label == desired_label:
                ordered_handles.append(
                    handle
                )
                ordered_labels.append(
                    label
                )

    ax.legend(
        ordered_handles,
        ordered_labels,
        ncol=3,
        frameon=False,
        loc="upper right",
    )

    fig.tight_layout()

    fig.savefig(
        str(output_path),
        format="png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_torque_command_plot(
    result,
    output_path: str | Path,
) -> None:
    """
    Save body-torque commands and torque limits.

    tau_x : roll torque
    tau_y : pitch torque
    tau_z : yaw torque
    """

    output_path = _prepare_output_path(
        output_path
    )

    time = result.time

    torque = result.control[:, 1:4]

    fig, ax = plt.subplots(
        figsize=(10.0, 5.4)
    )

    torque_labels = [
        r"$\tau_x$",
        r"$\tau_y$",
        r"$\tau_z$",
    ]

    for index, label in enumerate(
        torque_labels
    ):
        ax.plot(
            time,
            torque[:, index],
            linewidth=2.0,
            label=label,
        )

    # Roll/pitch torque limits.
    ax.axhline(
        0.35,
        linestyle="--",
        linewidth=1.0,
        alpha=0.35,
    )

    ax.axhline(
        -0.35,
        linestyle="--",
        linewidth=1.0,
        alpha=0.35,
    )

    # Yaw torque limits.
    ax.axhline(
        0.20,
        linestyle=":",
        linewidth=1.0,
        alpha=0.35,
    )

    ax.axhline(
        -0.20,
        linestyle=":",
        linewidth=1.0,
        alpha=0.35,
    )

    ax.set_title(
        "Body-Torque Commands — 1000 Hz"
    )

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_ylabel(
        r"Body Torque [N$\cdot$m]"
    )

    ax.grid(
        True,
        alpha=0.20,
    )

    add_command_events(
        ax
    )

    ax.legend(
        frameon=False,
        loc="upper right",
    )

    fig.tight_layout()

    fig.savefig(
        str(output_path),
        format="png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_attitude_animation(
    result,
    output_path: str | Path,
    *,
    fps: int = 24,
) -> None:
    """
    Save a GitHub-friendly 3D quadrotor attitude animation.

    The controller simulation may run at 1000 Hz.
    The GIF is downsampled to a practical visualization frame rate.
    """

    output_path = _prepare_output_path(
        output_path
    )

    time = result.time

    attitude = result.state[:, 6:9]

    command = result.attitude_command

    simulation_dt = float(
        time[1] - time[0]
    )

    frame_dt = 1.0 / fps

    stride = max(
        1,
        int(
            round(
                frame_dt
                / simulation_dt
            )
        ),
    )

    frame_indices = np.arange(
        0,
        len(time),
        stride,
    )

    fig = plt.figure(
        figsize=(8.5, 7.0)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    arm_length = 0.48

    rotor_points_body = np.array(
        [
            [
                arm_length,
                arm_length,
                0.0,
            ],
            [
                -arm_length,
                -arm_length,
                0.0,
            ],
            [
                arm_length,
                -arm_length,
                0.0,
            ],
            [
                -arm_length,
                arm_length,
                0.0,
            ],
        ],
        dtype=float,
    )

    axis_length = 0.70

    body_axes = np.array(
        [
            [
                axis_length,
                0.0,
                0.0,
            ],
            [
                0.0,
                axis_length,
                0.0,
            ],
            [
                0.0,
                0.0,
                axis_length,
            ],
        ],
        dtype=float,
    )

    def update(
        frame_number: int,
    ):
        index = int(
            frame_indices[
                frame_number
            ]
        )

        ax.cla()

        phi, theta, psi = attitude[
            index
        ]

        rotation = body_to_inertial(
            phi,
            theta,
            psi,
        )

        rotor_points_inertial = (
            rotation
            @ rotor_points_body.T
        ).T

        body_axes_inertial = (
            rotation
            @ body_axes.T
        ).T

        # Quadrotor arms.
        ax.plot(
            [
                rotor_points_inertial[
                    0,
                    0,
                ],
                rotor_points_inertial[
                    1,
                    0,
                ],
            ],
            [
                rotor_points_inertial[
                    0,
                    1,
                ],
                rotor_points_inertial[
                    1,
                    1,
                ],
            ],
            [
                rotor_points_inertial[
                    0,
                    2,
                ],
                rotor_points_inertial[
                    1,
                    2,
                ],
            ],
            linewidth=4.0,
        )

        ax.plot(
            [
                rotor_points_inertial[
                    2,
                    0,
                ],
                rotor_points_inertial[
                    3,
                    0,
                ],
            ],
            [
                rotor_points_inertial[
                    2,
                    1,
                ],
                rotor_points_inertial[
                    3,
                    1,
                ],
            ],
            [
                rotor_points_inertial[
                    2,
                    2,
                ],
                rotor_points_inertial[
                    3,
                    2,
                ],
            ],
            linewidth=4.0,
        )

        # Rotor markers.
        ax.scatter(
            rotor_points_inertial[
                :,
                0,
            ],
            rotor_points_inertial[
                :,
                1,
            ],
            rotor_points_inertial[
                :,
                2,
            ],
            s=80,
        )

        body_axis_labels = [
            "Body +X",
            "Body +Y",
            "Body +Z",
        ]

        for axis_index, label in enumerate(
            body_axis_labels
        ):
            ax.plot(
                [
                    0.0,
                    body_axes_inertial[
                        axis_index,
                        0,
                    ],
                ],
                [
                    0.0,
                    body_axes_inertial[
                        axis_index,
                        1,
                    ],
                ],
                [
                    0.0,
                    body_axes_inertial[
                        axis_index,
                        2,
                    ],
                ],
                linewidth=2.5,
                label=label,
            )

        actual_deg = np.rad2deg(
            attitude[index]
        )

        command_deg = np.rad2deg(
            command[index]
        )

        error_deg = (
            command_deg
            - actual_deg
        )

        ax.set_title(
            "1000 Hz Closed-Loop Quadrotor Attitude Control\n"
            f"t = {time[index]:.2f} s\n"
            f"Command [deg] = "
            f"({command_deg[0]:.1f}, "
            f"{command_deg[1]:.1f}, "
            f"{command_deg[2]:.1f})\n"
            f"Actual [deg] = "
            f"({actual_deg[0]:.1f}, "
            f"{actual_deg[1]:.1f}, "
            f"{actual_deg[2]:.1f})\n"
            f"Error [deg] = "
            f"({error_deg[0]:.2f}, "
            f"{error_deg[1]:.2f}, "
            f"{error_deg[2]:.2f})"
        )

        limit = 0.95

        ax.set_xlim(
            -limit,
            limit,
        )

        ax.set_ylim(
            -limit,
            limit,
        )

        ax.set_zlim(
            -limit,
            limit,
        )

        ax.set_xlabel(
            "Inertial X"
        )

        ax.set_ylabel(
            "Inertial Y"
        )

        ax.set_zlabel(
            "Inertial Z"
        )

        ax.set_box_aspect(
            (
                1.0,
                1.0,
                1.0,
            )
        )

        ax.view_init(
            elev=24.0,
            azim=35.0,
        )

        ax.legend(
            loc="upper left"
        )

        return []

    animation = FuncAnimation(
        fig,
        update,
        frames=len(
            frame_indices
        ),
        interval=1000.0 / fps,
        blit=False,
    )

    animation.save(
        str(output_path),
        writer=PillowWriter(
            fps=fps
        ),
    )

    plt.close(
        fig
    )