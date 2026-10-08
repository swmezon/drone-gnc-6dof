import numpy as np


def rmse(err):
    err = np.asarray(
        err,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    err**2,
                    axis=1,
                )
            )
        )
    )


def axis_rmse(err):
    err = np.asarray(
        err,
        dtype=float,
    )

    return np.sqrt(
        np.mean(
            err**2,
            axis=0,
        )
    )


def trajectory_metrics(
    pos,
    refs,
    waypoint_indices=None,
    final_waypoint=None,
    waypoints_reached=None,
):
    """
    Compute time-synchronized trajectory metrics.

    pos[k]:
        Actual position at sample k.

    refs[k]:
        Desired trajectory position at the
        same sample k.
    """

    pos = np.asarray(
        pos,
        dtype=float,
    )

    refs = np.asarray(
        refs,
        dtype=float,
    )

    if pos.shape != refs.shape:
        raise ValueError(
            "pos and refs must "
            "have the same shape; "
            f"got {pos.shape} "
            f"and {refs.shape}"
        )

    if (
        pos.ndim != 2
        or pos.shape[1] != 3
    ):
        raise ValueError(
            "pos and refs must "
            "have shape (N, 3)"
        )

    # ---------------------------------------------
    # TRAJECTORY ERROR
    # ---------------------------------------------

    err = (
        pos
        - refs
    )

    err_norm = (
        np.linalg.norm(
            err,
            axis=1,
        )
    )

    xyz_rmse = (
        axis_rmse(
            err
        )
    )

    trajectory_position_rmse = float(
        np.sqrt(
            np.mean(
                err_norm**2
            )
        )
    )

    result = {

        "trajectory_position_rmse_m":
            trajectory_position_rmse,

        # Kept as an alias for compatibility.
        # It now represents TRUE trajectory RMSE.
        "position_rmse_m":
            trajectory_position_rmse,

        "trajectory_x_rmse_m":
            float(
                xyz_rmse[0]
            ),

        "trajectory_y_rmse_m":
            float(
                xyz_rmse[1]
            ),

        "trajectory_z_rmse_m":
            float(
                xyz_rmse[2]
            ),

        "trajectory_mean_error_m":
            float(
                np.mean(
                    err_norm
                )
            ),

        "trajectory_max_error_m":
            float(
                np.max(
                    err_norm
                )
            ),

        # Compatibility aliases.
        "mean_error_m":
            float(
                np.mean(
                    err_norm
                )
            ),

        "max_error_m":
            float(
                np.max(
                    err_norm
                )
            ),
    }

    # ---------------------------------------------
    # FINAL TARGET ERROR
    # ---------------------------------------------

    if final_waypoint is None:

        final_reference = (
            refs[-1]
        )

    else:

        final_reference = (
            np.asarray(
                final_waypoint,
                dtype=float,
            )
        )

    result[
        "final_error_m"
    ] = float(
        np.linalg.norm(
            pos[-1]
            - final_reference
        )
    )

    # ---------------------------------------------
    # WAYPOINT COUNT
    # ---------------------------------------------

    if waypoints_reached is not None:

        result[
            "waypoints_reached"
        ] = int(
            waypoints_reached
        )

    elif waypoint_indices is not None:

        result[
            "waypoints_reached"
        ] = int(
            np.max(
                np.asarray(
                    waypoint_indices
                )
            )
            + 1
        )

    return result


def estimation_metrics(
    state_true,
    state_est,
):
    """
    Evaluate EKF navigation accuracy.

    State layout:

    [x, y, z,
     u, v, w,
     phi, theta, psi,
     p, q, r]
    """

    state_true = np.asarray(
        state_true,
        dtype=float,
    )

    state_est = np.asarray(
        state_est,
        dtype=float,
    )

    if (
        state_true.shape
        != state_est.shape
    ):
        raise ValueError(
            "state_true and state_est "
            "must have the same shape"
        )

    if (
        state_true.ndim != 2
        or state_true.shape[1] < 9
    ):
        raise ValueError(
            "state histories must contain "
            "at least 9 state elements"
        )

    # Position estimation error.
    pos_err = (
        state_est[:, 0:3]
        - state_true[:, 0:3]
    )

    # Velocity estimation error.
    vel_err = (
        state_est[:, 3:6]
        - state_true[:, 3:6]
    )

    # Attitude estimation error.
    att_err = (
        state_est[:, 6:9]
        - state_true[:, 6:9]
    )

    # Wrap angle errors to [-pi, pi].
    att_err = (
        att_err
        + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi

    att_rmse_deg = (
        np.rad2deg(
            axis_rmse(
                att_err
            )
        )
    )

    return {

        "ekf_position_rmse_m":
            rmse(
                pos_err
            ),

        "ekf_velocity_rmse_mps":
            rmse(
                vel_err
            ),

        "ekf_roll_rmse_deg":
            float(
                att_rmse_deg[0]
            ),

        "ekf_pitch_rmse_deg":
            float(
                att_rmse_deg[1]
            ),

        "ekf_yaw_rmse_deg":
            float(
                att_rmse_deg[2]
            ),
    }


def attitude_tracking_metrics(
    attitude_true,
    attitude_ref,
):
    """
    Compare commanded attitude against
    actual attitude.
    """

    attitude_true = np.asarray(
        attitude_true,
        dtype=float,
    )

    attitude_ref = np.asarray(
        attitude_ref,
        dtype=float,
    )

    err = (
        attitude_ref
        - attitude_true
    )

    err = (
        err
        + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi

    rmse_deg = (
        np.rad2deg(
            axis_rmse(
                err
            )
        )
    )

    return {

        "roll_tracking_rmse_deg":
            float(
                rmse_deg[0]
            ),

        "pitch_tracking_rmse_deg":
            float(
                rmse_deg[1]
            ),

        "yaw_tracking_rmse_deg":
            float(
                rmse_deg[2]
            ),
    }