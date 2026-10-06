# Attitude-Control Optimization Update

This update corrects the Python 3.14 dataclass compatibility issue and provides
a 400 Hz sequential gain optimizer for the cascaded quadrotor attitude-control
system.

## Controller rate

\[
f_c = 400~\mathrm{Hz}
\]

\[
\Delta t = 0.0025~\mathrm{s}
\]

## Optimization targets

- Roll RMSE < 0.5 deg
- Pitch RMSE < 0.5 deg
- Yaw RMSE < 1.0 deg
- steady-state error < 0.1 deg
- overshoot < 5%
- settling time < 1.0 s
- rise time between 0.3 and 0.7 s
- no sustained oscillation
- ideally no torque saturation

## Sequential tuning order

For each axis:

1. body-rate Kp
2. body-rate Ki
3. body-rate Kd
4. outer attitude Kp

The optimizer changes one scalar gain at a time and keeps only improvements.

## Compatibility

The module exposes both:

```python
StrongPortfolioTargets
```

and the backward-compatible alias:

```python
AutoTuneTargets
```

so older tests/imports do not fail after this update.

## Run

```powershell
python -m pytest -v
python -m scripts.run_pid_autotune
python -m scripts.run_closed_loop_attitude
```
