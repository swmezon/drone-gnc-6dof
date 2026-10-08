# Navigation and Estimation Notes

The repository includes a compact 15-state error-state EKF structure for demonstrating IMU/GPS fusion concepts.

Error-state ordering:

`[dp(3), dv(3), dtheta(3), dba(3), dbg(3)]`

- `dp`: position error
- `dv`: velocity error
- `dtheta`: small attitude error
- `dba`: accelerometer-bias error
- `dbg`: gyroscope-bias error

The IMU performs high-rate propagation. GPS supplies lower-rate absolute position and velocity corrections. `P` stores current state uncertainty, `Q` describes process uncertainty, `R` describes GPS measurement uncertainty, and `H` maps state errors into the GPS measurement space.

This portfolio implementation is intentionally compact. It demonstrates the estimator architecture and covariance machinery; a production inertial-navigation system would use a fuller continuous-time error Jacobian, quaternion nominal attitude, gravity/earth models as required, sensor timing logic, observability checks, and consistency testing.
