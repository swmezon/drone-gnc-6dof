# Sequential PID Auto-Tuning

This module adds an offline auto-tuning routine for the body-rate PID gains.

The tuning order is deliberately sequential:

1. proportional gain `rate_kp`
2. integral gain `rate_ki`
3. derivative gain `rate_kd`

Inside each gain family, the routine tunes one axis at a time:

1. roll
2. pitch
3. yaw

For every candidate gain it:

- runs the nonlinear 6-DOF closed-loop simulation;
- measures roll/pitch/yaw RMSE;
- changes one scalar gain only;
- accepts the gain if the total RMSE score improves;
- rejects and backs off if the score worsens;
- stops when all RMSE targets are met or the search limit is reached.

Run:

```bash
python -m scripts.run_pid_autotune
```

Outputs:

```text
results/autotune/
├── tuned_pid_gains.json
└── pid_autotune_history.csv
```

The JSON file contains the final tuned gains. The CSV contains every gain
candidate that was tested so the tuning process is fully traceable.

This is an offline simulation tuner. Do not apply the same automated search
directly to a physical vehicle without actuator limits, safety constraints,
fault handling, and a bounded experimental protocol.
