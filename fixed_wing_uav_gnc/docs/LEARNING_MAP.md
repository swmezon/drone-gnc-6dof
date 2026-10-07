# Learning map

1. `src/dynamics.py`: rigid-body 6-DOF equations and aerodynamic forces/moments.
2. `src/trim.py`: solve for straight-and-level equilibrium.
3. `src/linearize.py`: numerical Jacobians A and B.
4. `src/autopilot.py`: outer attitude / inner rate loops plus actuator limits.
5. `src/sensors.py`: noisy IMU and GPS.
6. `src/ekf.py`: 15-state navigation EKF.
7. `src/stability.py`: gain/phase margin calculations.
8. `src/experiment.py`: closed-loop mission.
9. `scripts/run_all.py`: reproducible validation and Monte Carlo.
