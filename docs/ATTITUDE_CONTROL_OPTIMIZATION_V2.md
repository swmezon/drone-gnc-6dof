# Attitude-Control Optimization v2

This revision changes the tuner so it no longer tries to minimize the initial
step error as if the vehicle could move instantaneously.

## Primary tracking metric

The optimizer now uses post-transient RMSE, beginning at the first 90% command
crossing.

Targets:

- roll post-transient RMSE < 0.25 deg
- pitch post-transient RMSE < 0.25 deg
- yaw post-transient RMSE < 0.50 deg

Full-step RMSE is still reported, but it is not the primary optimization
target.

## Dynamic-response targets

- steady-state error < 0.10 deg
- overshoot < 5%
- rise time between 0.3 and 0.7 s
- settling time < 1.0 s
- no sustained oscillation
- torque saturation < 2% for roll/pitch
- torque saturation < 5% for yaw

## Controller frequency

The optimization simulation runs at 400 Hz.

## Bidirectional scalar search

The previous optimizer tended to keep increasing gains.

This revision evaluates both:

- a higher candidate gain;
- a lower candidate gain.

Only the lower-penalty candidate is accepted.

This is especially important for the outer attitude proportional gain because
a response that is faster than 0.3 s should be slowed down rather than pushed
more aggressively.

## Tuning order

For each axis:

1. body-rate Kp
2. body-rate Ki
3. body-rate Kd
4. outer attitude Kp

Only one scalar is modified at a time.
