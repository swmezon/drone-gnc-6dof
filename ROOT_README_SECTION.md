## Fixed-Wing UAV GNC Simulation

A fixed-wing GNC proof of concept covering nonlinear 6-DOF dynamics, straight-and-level trim, numerical linearization, cascaded attitude/rate control, waypoint guidance, simulated IMU/GPS navigation with a 15-state EKF, local stability margins, actuator limits, and dispersed Monte Carlo verification.

**Current development validation:** 0.76 m nominal 3D position RMSE, 4.02 s nominal position convergence, 20.5 dB minimum local gain margin, 81.4° minimum local phase margin, and 3.94–4.77 rad/s roll/pitch gain crossover. Final Monte Carlo résumé statistics are generated directly by the reproducible validation script.

See [`fixed_wing_uav_gnc/README.md`](fixed_wing_uav_gnc/README.md) for architecture, requirements, plots, run commands, and the interview study guide.
