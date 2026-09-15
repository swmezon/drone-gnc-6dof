from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import numpy as np

from src.parameters import nominal_parameters
from src.rotations import body_to_inertial
from src.simulation import simulate


# ============================================================
# VIDEO SETTINGS
# ============================================================

FPS = 20

# Simple visual quadrotor arm length [m].
# This is visualization geometry only.
ARM_LENGTH = 0.18


# ============================================================
# DEMONSTRATION TIMING
# ============================================================

HOVER_1_END = 1.5

VERTICAL_START = 1.5
VERTICAL_END = 7.5

HOVER_2_END = 8.5

ROLL_1_START = 8.5
ROLL_1_END = 10.0

ROLL_2_START = 10.0
ROLL_2_END = 11.5

FINAL_TIME = 13.0


# ============================================================
# VERTICAL-MOTION PARAMETERS
# ============================================================

INITIAL_ALTITUDE = 1.0      # [m]
CLIMB_HEIGHT = 1.0         # [m]


# ============================================================
# ATTITUDE-MOTION PARAMETERS
# ============================================================

MAX_ROLL_DEG = 5.0
MAX_ROLL_RAD = np.deg2rad(MAX_ROLL_DEG)


def smooth_vertical_bump(
    t: float,
) -> tuple[float, float, float]:
    """
    Generate a smooth up-and-down vertical trajectory.

    Returns
    -------
    z_offset : float
        Vertical displacement relative to the initial altitude [m].

    z_dot : float
        Desired vertical velocity [m/s].

    z_ddot : float
        Desired vertical acceleration [m/s^2].

    Notes
    -----
    The normalized polynomial

        f(s) = 64 s^3 (1-s)^3

    satisfies

        f(0)   = 0
        f(0.5) = 1
        f(1)   = 0

    and has zero velocity and acceleration at the beginning
    and end of the maneuver.

    Therefore the drone gradually:

        starts at 1 m
        climbs to 2 m
        descends back to 1 m
        stops vertically
    """

    duration = VERTICAL_END - VERTICAL_START

    if t <= VERTICAL_START:
        return 0.0, 0.0, 0.0

    if t >= VERTICAL_END:
        return 0.0, 0.0, 0.0

    # Normalize maneuver time to 0 <= s <= 1.
    s = (t - VERTICAL_START) / duration

    # Position-shape function.
    f = (
        64.0
        * s**3
        * (1.0 - s)**3
    )

    # First derivative with respect to normalized time s.
    df_ds = 64.0 * (
        3.0 * s**2
        - 12.0 * s**3
        + 15.0 * s**4
        - 6.0 * s**5
    )

    # Second derivative with respect to normalized time s.
    d2f_ds2 = 64.0 * (
        6.0 * s
        - 36.0 * s**2
        + 60.0 * s**3
        - 30.0 * s**4
    )

    z_offset = CLIMB_HEIGHT * f

    z_dot = (
        CLIMB_HEIGHT
        * df_ds
        / duration
    )

    z_ddot = (
        CLIMB_HEIGHT
        * d2f_ds2
        / duration**2
    )

    return z_offset, z_dot, z_ddot


def roll_profile(
    t: float,
    start: float,
    end: float,
    sign: float,
) -> tuple[float, float, float]:
    """
    Generate one smooth roll excursion.

    The profile begins at

        phi = 0
        phi_dot = 0

    reaches positive and negative roll angles, and returns to

        phi = 0
        phi_dot = 0

    at the end.

    Parameters
    ----------
    t : float
        Current simulation time [s].

    start : float
        Maneuver start time [s].

    end : float
        Maneuver end time [s].

    sign : float
        +1 for the first excursion.
        -1 for the mirrored excursion.

    Returns
    -------
    phi_des : float
        Desired roll angle [rad].

    phi_dot_des : float
        Desired roll rate [rad/s].

    phi_ddot_des : float
        Desired roll acceleration [rad/s^2].
    """

    if t < start or t > end:
        return 0.0, 0.0, 0.0

    duration = end - start
    local_time = t - start

    omega = (
        2.0
        * np.pi
        / duration
    )

    phase = omega * local_time

    s = np.sin(phase)
    c = np.cos(phase)

    # Smooth angle trajectory.
    phi_des = (
        sign
        * MAX_ROLL_RAD
        * s**3
    )

    # First derivative.
    phi_dot_des = (
        sign
        * MAX_ROLL_RAD
        * 3.0
        * omega
        * s**2
        * c
    )

    # Second derivative.
    phi_ddot_des = (
        sign
        * MAX_ROLL_RAD
        * 3.0
        * omega**2
        * s
        * (
            2.0
            - 3.0 * s**2
        )
    )

    return (
        phi_des,
        phi_dot_des,
        phi_ddot_des,
    )


def continuous_demo_simulation():
    """
    Run one continuous 6-DOF demonstration.

    Sequence
    --------
    1. Hover at z = 1 m.
    2. Smooth climb to z = 2 m.
    3. Smooth descent back to z = 1 m.
    4. Brief hover.
    5. First roll excursion.
    6. Mirrored roll excursion.
    7. Final hover.

    This is intentionally ONE simulation.

    The state is never reset between scenes.
    """

    params = nominal_parameters()

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    x0 = np.zeros(
        12,
        dtype=float,
    )

    # Start one meter above the reference plane so the
    # hover is visually obvious.
    x0[2] = INITIAL_ALTITUDE

    hover_thrust = (
        params.mass
        * params.gravity
    )

    # --------------------------------------------------------
    # CONTROL SCHEDULE
    # --------------------------------------------------------

    def control(
        t: float,
        state: np.ndarray,
    ) -> np.ndarray:

        del state

        thrust = hover_thrust

        tau_x = 0.0
        tau_y = 0.0
        tau_z = 0.0

        # ====================================================
        # PHASE 1:
        # SMOOTH CLIMB + DESCENT
        # ====================================================

        if (
            VERTICAL_START
            <= t
            <= VERTICAL_END
        ):

            _, _, z_ddot_des = (
                smooth_vertical_bump(t)
            )

            # From:
            #
            #     z_ddot = T/m - g
            #
            # solve for T:
            #
            #     T = m(g + z_ddot)
            #
            thrust = (
                params.mass
                * (
                    params.gravity
                    + z_ddot_des
                )
            )

        # ====================================================
        # PHASE 2:
        # FIRST ROLL EXCURSION
        # ====================================================

        elif (
            ROLL_1_START
            <= t
            <= ROLL_1_END
        ):

            (
                phi_des,
                _,
                phi_ddot_des,
            ) = roll_profile(
                t,
                ROLL_1_START,
                ROLL_1_END,
                +1.0,
            )

            # Feedforward roll torque:
            #
            #     tau_x = Ixx * phi_ddot
            #
            # for this single-axis maneuver.
            tau_x = (
                params.Ixx
                * phi_ddot_des
            )

            # When rolled, only T*cos(phi) supports weight.
            #
            # Choose T such that
            #
            #     T*cos(phi) = mg
            #
            # which gives
            #
            #     T = mg/cos(phi)
            #
            # This prevents the drone from losing altitude
            # merely because it is tilted.
            thrust = (
                hover_thrust
                / np.cos(phi_des)
            )

        # ====================================================
        # PHASE 3:
        # MIRRORED ROLL EXCURSION
        # ====================================================

        elif (
            ROLL_2_START
            <= t
            <= ROLL_2_END
        ):

            (
                phi_des,
                _,
                phi_ddot_des,
            ) = roll_profile(
                t,
                ROLL_2_START,
                ROLL_2_END,
                -1.0,
            )

            tau_x = (
                params.Ixx
                * phi_ddot_des
            )

            thrust = (
                hover_thrust
                / np.cos(phi_des)
            )

        return np.array(
            [
                thrust,
                tau_x,
                tau_y,
                tau_z,
            ],
            dtype=float,
        )

    # --------------------------------------------------------
    # RUN ONE CONTINUOUS SIMULATION
    # --------------------------------------------------------

    result = simulate(
        x0,
        control,
        params,
        t_final=FINAL_TIME,
        dt=0.005,
        method="rk4",
    )

    return result


def frame_indices(
    time: np.ndarray,
    fps: int,
) -> np.ndarray:
    """
    Select simulation samples for approximately real-time video.
    """

    frame_times = np.arange(
        time[0],
        time[-1],
        1.0 / fps,
    )

    indices = np.searchsorted(
        time,
        frame_times,
    )

    indices = np.clip(
        indices,
        0,
        len(time) - 1,
    )

    return np.unique(indices)


def phase_name(
    t: float,
) -> str:
    """
    Return a human-readable maneuver name for the video.
    """

    if t < HOVER_1_END:
        return "Hover Equilibrium"

    if t < VERTICAL_END:

        midpoint = (
            VERTICAL_START
            + 0.5
            * (
                VERTICAL_END
                - VERTICAL_START
            )
        )

        if t < midpoint:
            return "Smooth Climb"

        return "Smooth Descent"

    if t < HOVER_2_END:
        return "Hover"

    if t < ROLL_1_END:
        return "Roll Excursion"

    if t < ROLL_2_END:
        return "Return Roll Excursion"

    return "Final Hover"


def main() -> None:

    print(
        "Running continuous 6-DOF demonstration..."
    )

    result = continuous_demo_simulation()

    indices = frame_indices(
        result.time,
        FPS,
    )

    # ========================================================
    # PRINT FINAL CHECKS
    # ========================================================

    final_position = result.state[
        -1,
        0:3,
    ]

    final_velocity = result.state[
        -1,
        3:6,
    ]

    final_attitude_deg = np.rad2deg(
        result.state[
            -1,
            6:9,
        ]
    )

    final_rates_deg_s = np.rad2deg(
        result.state[
            -1,
            9:12,
        ]
    )

    print()
    print("=== Final demonstration state ===")

    print(
        "Position [m]      : "
        f"{final_position}"
    )

    print(
        "Velocity [m/s]    : "
        f"{final_velocity}"
    )

    print(
        "Attitude [deg]    : "
        f"{final_attitude_deg}"
    )

    print(
        "Body rate [deg/s] : "
        f"{final_rates_deg_s}"
    )

    # ========================================================
    # CREATE FIGURE
    # ========================================================

    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    # Fixed field of view.
    #
    # The roll maneuver is deliberately bounded so the drone
    # should remain inside this region.
    ax.set_xlim(
        -0.50,
        0.50,
    )

    ax.set_ylim(
        -0.50,
        0.50,
    )

    ax.set_zlim(
        0.50,
        2.30,
    )

    ax.set_xlabel(
        "x [m]"
    )

    ax.set_ylabel(
        "y [m]"
    )

    ax.set_zlabel(
        "z [m]"
    )

    ax.set_title(
        "Quadrotor GNC â€” 6-DOF Dynamics + Integration"
    )

    # ========================================================
    # TRAJECTORY
    # ========================================================

    trajectory_line, = ax.plot(
        [],
        [],
        [],
        linewidth=2.0,
        label="Trajectory",
    )

    # ========================================================
    # QUADROTOR VISUAL BODY
    # ========================================================

    body_arm_x, = ax.plot(
        [],
        [],
        [],
        linewidth=4.0,
    )

    body_arm_y, = ax.plot(
        [],
        [],
        [],
        linewidth=4.0,
    )

    rotor_points = ax.scatter(
        [],
        [],
        [],
        s=75,
    )

    center_point = ax.scatter(
        [],
        [],
        [],
        s=45,
    )

    # ========================================================
    # STATUS TEXT
    # ========================================================

    status_text = ax.text2D(
        0.03,
        0.95,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
    )

    ax.legend(
        loc="upper right"
    )

    # ========================================================
    # BODY GEOMETRY IN BODY COORDINATES
    # ========================================================

    arm_x_body = np.array(
        [
            [
                -ARM_LENGTH,
                0.0,
                0.0,
            ],
            [
                ARM_LENGTH,
                0.0,
                0.0,
            ],
        ],
        dtype=float,
    )

    arm_y_body = np.array(
        [
            [
                0.0,
                -ARM_LENGTH,
                0.0,
            ],
            [
                0.0,
                ARM_LENGTH,
                0.0,
            ],
        ],
        dtype=float,
    )

    rotor_body = np.array(
        [
            [
                ARM_LENGTH,
                0.0,
                0.0,
            ],
            [
                -ARM_LENGTH,
                0.0,
                0.0,
            ],
            [
                0.0,
                ARM_LENGTH,
                0.0,
            ],
            [
                0.0,
                -ARM_LENGTH,
                0.0,
            ],
        ],
        dtype=float,
    )

    def transform_points(
        body_points: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray,
    ) -> np.ndarray:

        return (
            rotation
            @ body_points.T
        ).T + position

    # ========================================================
    # ANIMATION UPDATE
    # ========================================================

    def update(
        frame_number: int,
    ):

        k = indices[
            frame_number
        ]

        t = result.time[k]

        state = result.state[k]

        position = state[0:3]

        phi = state[6]
        theta = state[7]
        psi = state[8]

        R_BI = body_to_inertial(
            phi,
            theta,
            psi,
        )

        arm_x = transform_points(
            arm_x_body,
            position,
            R_BI,
        )

        arm_y = transform_points(
            arm_y_body,
            position,
            R_BI,
        )

        rotors = transform_points(
            rotor_body,
            position,
            R_BI,
        )

        trajectory = result.state[
            : k + 1,
            0:3,
        ]

        # ----------------------------------------------------
        # TRAJECTORY
        # ----------------------------------------------------

        trajectory_line.set_data(
            trajectory[:, 0],
            trajectory[:, 1],
        )

        trajectory_line.set_3d_properties(
            trajectory[:, 2]
        )

        # ----------------------------------------------------
        # BODY ARM X
        # ----------------------------------------------------

        body_arm_x.set_data(
            arm_x[:, 0],
            arm_x[:, 1],
        )

        body_arm_x.set_3d_properties(
            arm_x[:, 2]
        )

        # ----------------------------------------------------
        # BODY ARM Y
        # ----------------------------------------------------

        body_arm_y.set_data(
            arm_y[:, 0],
            arm_y[:, 1],
        )

        body_arm_y.set_3d_properties(
            arm_y[:, 2]
        )

        # ----------------------------------------------------
        # ROTORS
        # ----------------------------------------------------

        rotor_points._offsets3d = (
            rotors[:, 0],
            rotors[:, 1],
            rotors[:, 2],
        )

        # ----------------------------------------------------
        # CENTER OF MASS
        # ----------------------------------------------------

        center_point._offsets3d = (
            [position[0]],
            [position[1]],
            [position[2]],
        )

        # ----------------------------------------------------
        # INFORMATION DISPLAY
        # ----------------------------------------------------

        status_text.set_text(
            f"{phase_name(t)}\n"
            f"t = {t:.2f} s\n"
            f"z = {position[2]:.2f} m\n"
            f"roll = {np.rad2deg(phi):.2f} deg"
        )

        return (
            trajectory_line,
            body_arm_x,
            body_arm_y,
            rotor_points,
            center_point,
            status_text,
        )

    # ========================================================
    # CREATE ANIMATION
    # ========================================================

    animation = FuncAnimation(
        fig,
        update,
        frames=len(indices),
        interval=1000.0 / FPS,
        blit=False,
    )

    # ========================================================
    # SAVE OUTPUT
    # ========================================================

    output_directory = Path(
        "results/media"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    gif_path = (
        output_directory
        / "quadrotor_gnc_demo.gif"
    )

    mp4_path = (
        output_directory
        / "quadrotor_gnc_demo.mp4"
    )

    print()
    print(
        f"Saving GIF to {gif_path}"
    )

    animation.save(
        gif_path,
        writer=PillowWriter(
            fps=FPS
        ),
        dpi=100,
    )

    if shutil.which(
        "ffmpeg"
    ):

        print(
            f"Saving MP4 to {mp4_path}"
        )

        animation.save(
            mp4_path,
            writer=FFMpegWriter(
                fps=FPS,
                bitrate=2200,
            ),
            dpi=100,
        )

    else:

        print()
        print(
            "FFmpeg not found."
        )

        print(
            "GIF created successfully; "
            "MP4 was skipped."
        )

    plt.close(fig)

    print()
    print(
        "Demo generation complete."
    )

    print(
        f"GIF: {gif_path}"
    )

    if shutil.which(
        "ffmpeg"
    ):

        print(
            f"MP4: {mp4_path}"
        )


if __name__ == "__main__":
    main()
