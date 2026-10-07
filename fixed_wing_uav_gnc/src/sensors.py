from dataclasses import dataclass
import numpy as np
from src.rotations import R_body_to_ned

@dataclass(frozen=True)
class SensorConfig:
    gyro_noise_std: float=np.deg2rad(0.08)
    accel_noise_std: float=0.06
    gps_pos_std: float=1.5
    gps_vel_std: float=0.15
    gyro_bias_std: float=np.deg2rad(0.20)
    accel_bias_std: float=0.08


def draw_biases(rng,cfg):
    return rng.normal(0,cfg.gyro_bias_std,3), rng.normal(0,cfg.accel_bias_std,3)

def imu_measurement(x,dx,bg,ba,rng,cfg,g=9.80665):
    phi,theta,psi=x[6:9]
    omega=x[9:12]
    Rbn=R_body_to_ned(phi,theta,psi)
    a_n=dx[3:6]  # temporary body acceleration derivative is not inertial
    # reconstruct specific force using body translational equation: f = vdot + omega x v - R^T g_n
    vel=x[3:6]
    specific_b=dx[3:6]+np.cross(omega,vel)-Rbn.T@np.array([0,0,g])
    gyro=omega+bg+rng.normal(0,cfg.gyro_noise_std,3)
    accel=specific_b+ba+rng.normal(0,cfg.accel_noise_std,3)
    return gyro,accel

def gps_measurement(x,rng,cfg):
    pos=x[:3]+rng.normal(0,cfg.gps_pos_std,3)
    vel_n=R_body_to_ned(*x[6:9])@x[3:6]
    vel=vel_n+rng.normal(0,cfg.gps_vel_std,3)
    return np.concatenate([pos,vel])
