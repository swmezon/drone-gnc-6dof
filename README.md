# Quadrotor Guidance, Navigation, and Control Simulation

6-DOF quadrotor GNC simulation with nonlinear rigid-body dynamics, waypoint guidance, cascaded flight control, IMU/position sensor modeling, 15-state Extended Kalman Filter state estimation, and Monte Carlo validation.

## Closed-Loop GNC

![Closed-loop waypoint tracking](results/animations/optimized_2cm_full_gnc_3d.gif)

## Performance

| Metric | Nominal | Monte Carlo P95 |
|---|---:|---:|
| Trajectory RMSE | **0.0223 m** | **0.0304 m** |
| Cross-track RMSE | **0.0180 m** | **0.0242 m** |
| Final waypoint error | **0.0263 m** | **0.0411 m** |
| EKF position RMSE | **0.0176 m** | **0.0195 m** |

**Mission completion:** 100 / 100 trials  
**Tracking-envelope pass rate:** 99 / 100 trials

![Monte Carlo validation](results/robustness/monte_carlo_validation_summary.png)

## GNC Architecture

- Nonlinear 6-DOF quadrotor dynamics
- Waypoint guidance and trajectory generation
- Cascaded position, attitude, and angular-rate control
- IMU and absolute-position sensor simulation
- 15-state error-state Extended Kalman Filter
- 100 Hz control and IMU propagation
- 10 Hz position measurement updates
- 100-run Monte Carlo validation

## State Estimation

The navigation filter estimates position, velocity, attitude, accelerometer bias, and gyroscope bias using high-rate IMU propagation and lower-rate absolute-position corrections.

```text
δx = [δp, δv, δθ, δba, δbg]