# Closed-Loop Attitude-Control Milestone

This milestone closes the first rotational feedback loop around the existing
nonlinear quadrotor plant.

## Architecture

```text
Roll/Pitch/Yaw Command
        |
        v
Attitude Error
        |
        v
Outer Attitude P Controller
        |
        v
Desired Body Rates [p_d, q_d, r_d]
        |
        v
Body-Rate Error
        |
        v
Inner PID Rate Controller
        |
        v
Body Torque Command [tau_x, tau_y, tau_z]
        |
        v
Existing Nonlinear 6-DOF Plant
        |
        +------------------> Feedback
```

## Why this structure is used

The existing plant propagates body angular velocity [p, q, r] through Euler's
rigid-body equation. The outer loop therefore converts attitude error into the
body-rate quantities that are natural to the plant, and the inner loop controls
those rates using body torque.

The controller is updated once per 5 ms sample and its torque is held constant
during the RK4 integration interval. This prevents the controller integral
state from being updated multiple times during RK4 sub-evaluations.

## Validation command sequence

- 0-1 s: [0, 0, 0] deg
- 1-3 s: [10, 0, 0] deg
- 3-5 s: [10, -7, 0] deg
- 5-8 s: [10, -7, 20] deg

## Generated artifacts

- attitude commanded-versus-actual plot
- desired-versus-actual body-rate plot
- body-torque command plot
- complete CSV time series
- 3-D attitude animation for GitHub

## Current limitation

Collective thrust is held at the nominal hover value. The rotational loop is
closed, while position is still open-loop. The next milestone is an outer
position/velocity controller that generates thrust and attitude commands.
