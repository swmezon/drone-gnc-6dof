import numpy as np
from drone_gnc.vehicles.quadrotor import rotation_matrix

class ErrorStateEKF15:
    """Compact 15-state error-state EKF for educational/portfolio simulation.

    Nominal state: position, velocity, Euler attitude, accel bias, gyro bias.
    Error covariance: [dp,dv,dtheta,dba,dbg].
    """
    def __init__(self, dt, gps_pos_std=0.45, gps_vel_std=0.08):
        self.dt=dt; self.p=np.zeros(3); self.v=np.zeros(3); self.att=np.zeros(3); self.ba=np.zeros(3); self.bg=np.zeros(3)
        sig=np.r_[np.ones(3)*1.0,np.ones(3)*0.5,np.ones(3)*np.deg2rad(5),np.ones(3)*0.1,np.ones(3)*0.01]
        self.P=np.diag(sig**2)
        self.Q=np.diag(np.r_[np.ones(3)*1e-5,np.ones(3)*3e-3,np.ones(3)*3e-5,np.ones(3)*1e-6,np.ones(3)*1e-8])
        self.R=np.diag(np.r_[np.ones(3)*gps_pos_std**2,np.ones(3)*gps_vel_std**2])
        self.H=np.zeros((6,15)); self.H[0:3,0:3]=np.eye(3); self.H[3:6,3:6]=np.eye(3)
    def initialize(self,p,v,att): self.p=p.copy(); self.v=v.copy(); self.att=att.copy()
    def predict(self,accel_m,gyro_m):
        dt=self.dt; omega=gyro_m-self.bg; self.att += omega*dt
        Rbi=rotation_matrix(*self.att); acc_i=Rbi@(accel_m-self.ba)+np.array([0,0,-9.81])
        self.p += self.v*dt+0.5*acc_i*dt*dt; self.v += acc_i*dt
        F=np.eye(15); F[0:3,3:6]=np.eye(3)*dt
        self.P=F@self.P@F.T+self.Q
    def update_gps(self,p_gps,v_gps):
        z=np.r_[p_gps,v_gps]; h=np.r_[self.p,self.v]; r=z-h
        S=self.H@self.P@self.H.T+self.R; K=self.P@self.H.T@np.linalg.inv(S); dx=K@r
        self.p+=dx[0:3]; self.v+=dx[3:6]; self.att+=dx[6:9]; self.ba+=dx[9:12]; self.bg+=dx[12:15]
        I=np.eye(15); self.P=(I-K@self.H)@self.P@(I-K@self.H).T+K@self.R@K.T
        return r,K
    def state12(self): return np.r_[self.p,self.v,self.att,np.zeros(3)]
