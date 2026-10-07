import numpy as np
from src.rotations import R_body_to_ned, euler_rate_matrix

# Navigation state:
# [pn,pe,pd, vn,ve,vd, phi,theta,psi, bgx,bgy,bgz, bax,bay,baz]
class NavigationEKF:
    def __init__(self, x0=None):
        self.x = np.zeros(15) if x0 is None else np.asarray(x0, float).copy()
        self.P = np.diag(
            [25, 25, 25, 4, 4, 4, 0.05, 0.05, 0.1, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05]
        )
        self.Q = np.diag([1e-5] * 3 + [3e-3] * 3 + [2e-4] * 3 + [1e-7] * 3 + [1e-6] * 3)
        self.R = np.diag([1.5**2] * 3 + [0.15**2] * 3)

    def f(self, x, gyro, accel):
        out = np.zeros(15)
        phi, theta, psi = x[6:9]
        bg = x[9:12]
        ba = x[12:15]
        omega = gyro - bg
        a_n = R_body_to_ned(phi, theta, psi) @ (accel - ba) + np.array([0, 0, 9.80665])
        out[0:3] = x[3:6]
        out[3:6] = a_n
        out[6:9] = euler_rate_matrix(phi, theta) @ omega
        return out

    def _continuous_jacobian(self, x, gyro, accel):
        """Build a lightweight continuous-time Jacobian for covariance propagation.

        The obvious blocks are analytic: position depends on velocity, acceleration
        depends on accelerometer bias, and attitude rate depends on gyro bias.
        Only the three attitude columns are finite-differenced because the rotation
        matrix makes those derivatives nonlinear. This is much faster than
        finite-differencing all 15 states at every IMU step.
        """
        A = np.zeros((15, 15))
        A[0:3, 3:6] = np.eye(3)

        phi, theta, psi = x[6:9]
        R = R_body_to_ned(phi, theta, psi)
        W = euler_rate_matrix(phi, theta)
        A[3:6, 12:15] = -R
        A[6:9, 9:12] = -W

        # Couple attitude uncertainty into acceleration and attitude-rate uncertainty.
        for j in range(3):
            i = 6 + j
            h = 1e-6
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            fp = self.f(xp, gyro, accel)
            fm = self.f(xm, gyro, accel)
            A[3:9, i] = (fp[3:9] - fm[3:9]) / (2.0 * h)
        return A

    def predict(self, gyro, accel, dt):
        x0 = self.x.copy()
        f0 = self.f(x0, gyro, accel)
        self.x = x0 + dt * f0
        A = self._continuous_jacobian(x0, gyro, accel)
        F = np.eye(15) + A * dt
        self.P = F @ self.P @ F.T + self.Q * dt

    def update_gps(self, z):
        H = np.zeros((6, 15))
        H[0:6, 0:6] = np.eye(6)
        y = np.asarray(z) - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(15)
        # Joseph form preserves symmetry/positive-semidefiniteness better numerically.
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T
        return y, S
