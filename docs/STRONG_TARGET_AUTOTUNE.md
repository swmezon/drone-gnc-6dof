# 400 Hz Strong-Target Auto-Tuning

The controller optimizer now runs at 400 Hz:

\[
f_c = 400~\mathrm{Hz}
\]

which corresponds to

\[
\Delta t = \frac{1}{400} = 0.0025~\mathrm{s}.
\]

## Target envelope

The tuner attempts to satisfy all of the following:

- Roll RMSE < 0.5 deg
- Pitch RMSE < 0.5 deg
- Yaw RMSE < 1.0 deg
- steady-state error < 0.1 deg
- overshoot < 5%
- settling time < 1.0 s
- rise time between 0.3 and 0.7 s
- no sustained oscillation
- no sustained torque saturation

## Tuning sequence

Each axis is tuned independently using an isolated step command.

For each axis, gain families are adjusted one at a time:

1. body-rate Kp
2. body-rate Ki
3. body-rate Kd
4. outer attitude Kp

The full nonlinear 6-DOF simulation is re-run after every candidate gain.

A candidate is kept only when it reduces the penalty associated with target
violations.

## Important

The optimizer is designed to *attempt* to meet the target envelope. It does
not force impossible physics. If actuator limits, inertia, or the current
controller structure prevent the targets from being met, the script returns
`Converged: False` and records the best gain set found.
