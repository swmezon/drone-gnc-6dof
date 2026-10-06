# Tuned Closed-Loop Validation

`run_closed_loop_attitude.py` now performs a reproducible before-versus-after
validation.

## Behavior

When the file below exists:

```text
results/autotune/tuned_pid_gains.json
```

the script:

1. runs the nominal baseline controller;
2. loads the tuned gains from JSON;
3. runs the same nonlinear 6-DOF experiment with the tuned gains;
4. prints baseline versus tuned RMSE for roll, pitch, and yaw;
5. reports the percent RMSE reduction;
6. regenerates the standard tracking plots and GIF using the tuned controller;
7. saves a before-versus-after comparison CSV and PNG.

If the tuned JSON does not exist, the script falls back to the nominal gains.

## Run

```powershell
python -m scripts.run_closed_loop_attitude
```

## Outputs

```text
results/
├── animations/
│   └── closed_loop_attitude_control.gif
├── comparison/
│   ├── before_after_metrics.csv
│   └── before_after_rmse.png
├── data/
│   └── closed_loop_attitude_tracking.csv
└── figures/
    ├── closed_loop_attitude_tracking.png
    ├── closed_loop_body_rate_tracking.png
    └── closed_loop_torque_commands.png
```

The normal GitHub-facing plots and animation always represent the selected
controller, which means the tuned controller whenever the tuned JSON exists.
