from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from src.attitude_controller import (
    CascadedAttitudeController,
    nominal_attitude_control_gains,
)
from src.closed_loop_simulation import simulate_closed_loop_attitude
from src.parameters import nominal_parameters
from src.rotations import body_to_inertial


CONTROLLER_HZ = 1000.0
DT = 1.0 / CONTROLLER_HZ
TUNED_GAINS_PATH = Path("results/autotune/tuned_pid_gains.json")
OUTPUT_GIF = Path("results/animations/quadrotor_attitude_control_1000hz.gif")


def commanded_attitude(t: float) -> np.ndarray:
    """
    Validation attitude command sequence.

    0-1 s : level
    1-3 s : +10 deg roll
    3-5 s : +10 deg roll, -7 deg pitch
    5-8 s : +10 deg roll, -7 deg pitch, +20 deg yaw
    """
    if t < 1.0:
        command_deg = [0.0, 0.0, 0.0]
    elif t < 3.0:
        command_deg = [10.0, 0.0, 0.0]
    elif t < 5.0:
        command_deg = [10.0, -7.0, 0.0]
    else:
        command_deg = [10.0, -7.0, 20.0]

    return np.deg2rad(np.array(command_deg, dtype=float))


def load_selected_gains():
    """
    Load the final selected gain set if the tuned-gain JSON exists.

    Falls back to the nominal controller gains when the file is absent.
    """
    gains = nominal_attitude_control_gains()

    if not TUNED_GAINS_PATH.exists():
        return gains

    data = json.loads(
        TUNED_GAINS_PATH.read_text(encoding="utf-8")
    )

    # Update only the gain arrays. Keep limits from the nominal configuration.
    gains = gains.__class__(
        attitude_kp=np.asarray(
            data.get("attitude_kp", gains.attitude_kp),
            dtype=float,
        ),
        rate_kp=np.asarray(
            data.get("rate_kp", gains.rate_kp),
            dtype=float,
        ),
        rate_ki=np.asarray(
            data.get("rate_ki", gains.rate_ki),
            dtype=float,
        ),
        rate_kd=np.asarray(
            data.get("rate_kd", gains.rate_kd),
            dtype=float,
        ),
        max_body_rate=gains.max_body_rate,
        max_torque=gains.max_torque,
        integral_limit=gains.integral_limit,
    )

    return gains


def run_closed_loop():
    params = nominal_parameters()
    gains = load_selected_gains()
    controller = CascadedAttitudeController(gains)

    result = simulate_closed_loop_attitude(
        initial_state=np.zeros(12, dtype=float),
        attitude_reference=commanded_attitude,
        controller=controller,
        params=params,
        t_final=8.0,
        dt=DT,
    )

    return result


def save_animation(result, output_path: Path = OUTPUT_GIF, fps: int = 30):
    """
    Save a GitHub-friendly 3D quadrotor attitude animation.

    The animation displays:
    - the simulated quadrotor orientation;
    - inertial coordinate axes;
    - commanded roll/pitch/yaw;
    - actual roll/pitch/yaw;
    - instantaneous attitude tracking error;
    - 1000 Hz controller update rate.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    time = result.time
    attitude = result.state[:, 6:9]
    command = result.attitude_command

    # Downsample simulation states for video rendering only.
    # The control simulation itself remains at 1000 Hz.
    frame_period = 1.0 / fps
    simulation_dt = float(time[1] - time[0])
    stride = max(1, int(round(frame_period / simulation_dt)))
    frame_indices = np.arange(0, len(time), stride)

    fig = plt.figure(figsize=(9.0, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    arm = 0.46

    # X-configuration quadrotor arms in the body frame.
    rotor_points_b = np.array(
        [
            [ arm,  arm, 0.0],
            [-arm, -arm, 0.0],
            [ arm, -arm, 0.0],
            [-arm,  arm, 0.0],
        ],
        dtype=float,
    )

    body_axis_length = 0.72
    body_axes_b = np.array(
        [
            [body_axis_length, 0.0, 0.0],
            [0.0, body_axis_length, 0.0],
            [0.0, 0.0, body_axis_length],
        ],
        dtype=float,
    )

    def update(frame_number: int):
        k = int(frame_indices[frame_number])
        ax.cla()

        phi, theta, psi = attitude[k]
        rotation = body_to_inertial(phi, theta, psi)

        rotor_points_i = (rotation @ rotor_points_b.T).T
        body_axes_i = (rotation @ body_axes_b.T).T

        # Quadrotor arms.
        ax.plot(
            [rotor_points_i[0, 0], rotor_points_i[1, 0]],
            [rotor_points_i[0, 1], rotor_points_i[1, 1]],
            [rotor_points_i[0, 2], rotor_points_i[1, 2]],
            linewidth=4.0,
        )
        ax.plot(
            [rotor_points_i[2, 0], rotor_points_i[3, 0]],
            [rotor_points_i[2, 1], rotor_points_i[3, 1]],
            [rotor_points_i[2, 2], rotor_points_i[3, 2]],
            linewidth=4.0,
        )

        # Rotor locations.
        ax.scatter(
            rotor_points_i[:, 0],
            rotor_points_i[:, 1],
            rotor_points_i[:, 2],
            s=90,
        )

        # Body axes.
        for i, label in enumerate(("Body +X", "Body +Y", "Body +Z")):
            ax.plot(
                [0.0, body_axes_i[i, 0]],
                [0.0, body_axes_i[i, 1]],
                [0.0, body_axes_i[i, 2]],
                linewidth=2.5,
                label=label,
            )

        command_deg = np.rad2deg(command[k])
        actual_deg = np.rad2deg(attitude[k])
        error_deg = command_deg - actual_deg

        ax.set_title(
            "1000 Hz Closed-Loop Quadrotor Attitude Control\n"
            f"t = {time[k]:.2f} s\n"
            f"Command [deg] = "
            f"({command_deg[0]:.1f}, {command_deg[1]:.1f}, {command_deg[2]:.1f})\n"
            f"Actual [deg] = "
            f"({actual_deg[0]:.1f}, {actual_deg[1]:.1f}, {actual_deg[2]:.1f})\n"
            f"Error [deg] = "
            f"({error_deg[0]:.2f}, {error_deg[1]:.2f}, {error_deg[2]:.2f})"
        )

        limit = 0.95
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)

        ax.set_xlabel("Inertial X")
        ax.set_ylabel("Inertial Y")
        ax.set_zlabel("Inertial Z")

        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.view_init(elev=24.0, azim=35.0)
        ax.legend(loc="upper left")

        return []

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000.0 / fps,
        blit=False,
    )

    animation.save(
        output_path,
        writer=PillowWriter(fps=fps),
    )

    plt.close(fig)

    print(f"Saved animation: {output_path}")


def main():
    print("Running 1000 Hz nonlinear closed-loop simulation...")
    result = run_closed_loop()

    print("Rendering GitHub animation...")
    save_animation(result)

    print("Done.")


if __name__ == "__main__":
    main()
