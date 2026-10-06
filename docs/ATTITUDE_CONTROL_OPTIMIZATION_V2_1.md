# Attitude-Control Optimization v2.1

This maintenance release fixes compatibility between the new post-transient
RMSE target model and older test modules already present in the repository.

Changes:
- adds `rmse_deg` as a backward-compatible property;
- overwrites the previous `test_pid_autotuner.py`;
- overwrites the previous `test_strong_target_autotuner.py`;
- preserves the 400 Hz optimizer;
- preserves bidirectional gain search;
- preserves post-transient RMSE targets;
- preserves rise-time, overshoot, settling-time, oscillation, and saturation constraints.
