from dataclasses import dataclass
import numpy as np
from drone_gnc.vehicles.quadrotor import rotation_matrix

@dataclass
class SensorConfig:
    accel_noise_std: float=0.05
    gyro_noise_std: float=0.003
    gps_pos_std: float=0.45
    gps_vel_std: float=0.08
    accel_bias_rw_std: float=0.002
    gyro_bias_rw_std: float=0.0002

class ImuGpsSensorSuite:
    def __init__(self, config=None, seed=42):
        self.cfg=config or SensorConfig(); self.rng=np.random.default_rng(seed)
        self.ba=np.zeros(3); self.bg=np.zeros(3)
    def imu(self,state,acc_i,dt):
        c=self.cfg; self.ba += self.rng.normal(0,c.accel_bias_rw_std*np.sqrt(dt),3); self.bg += self.rng.normal(0,c.gyro_bias_rw_std*np.sqrt(dt),3)
        R=rotation_matrix(*state[6:9]); specific_b=R.T@(acc_i-np.array([0,0,-9.81]))
        accel=specific_b+self.ba+self.rng.normal(0,c.accel_noise_std,3)
        gyro=state[9:12]+self.bg+self.rng.normal(0,c.gyro_noise_std,3)
        return accel,gyro
    def gps(self,state):
        c=self.cfg
        return state[0:3]+self.rng.normal(0,c.gps_pos_std,3), state[3:6]+self.rng.normal(0,c.gps_vel_std,3)
