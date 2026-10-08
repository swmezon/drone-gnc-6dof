import numpy as np
from drone_gnc.simulation.integrators import rk4_step


def run_waypoint_mission(vehicle, controller, guidance, sensor_suite=None, ekf=None, duration=32.0, dt=0.01, gps_hz=10.0, wind=None):
    n=int(duration/dt)+1; t=np.arange(n)*dt; x=np.zeros(12); x[2]=0.0; controller.reset()
    if ekf is not None: ekf.initialize(x[:3],x[3:6],x[6:9])
    logs={k:[] for k in ['t','state','control','ref','vel_ref','wp_index','distance','estimate','gps_pos']}
    gps_period=max(1,int(round(1/(gps_hz*dt))))
    for k,tk in enumerate(t):
        feedback=x if ekf is None else np.r_[ekf.p,ekf.v,x[6:9],x[9:12]]
        pref,vref,yaw,idx,dist=guidance.update(feedback[:3]); u,_=controller.command(feedback,pref,vref,yaw,dt)
        xdot=vehicle.derivatives(x,u); acc_i=xdot[3:6]
        if wind is not None: xdot[3:6]+=wind.acceleration()
        x=rk4_step(lambda xx,uu: vehicle.derivatives(xx,uu)+(np.r_[np.zeros(3),wind.acceleration(),np.zeros(6)] if wind is not None else 0),x,u,dt) if k<n-1 else x
        gps_p=np.full(3,np.nan)
        if sensor_suite is not None and ekf is not None:
            accel_m,gyro_m=sensor_suite.imu(x,acc_i,dt); ekf.predict(accel_m,gyro_m)
            if k%gps_period==0:
                gps_p,gps_v=sensor_suite.gps(x); ekf.update_gps(gps_p,gps_v)
        logs['t'].append(tk); logs['state'].append(x.copy()); logs['control'].append(u.copy()); logs['ref'].append(pref); logs['vel_ref'].append(vref); logs['wp_index'].append(idx); logs['distance'].append(dist); logs['estimate'].append(x.copy() if ekf is None else np.r_[ekf.p,ekf.v,ekf.att,x[9:12]]); logs['gps_pos'].append(gps_p)
    for k in logs: logs[k]=np.asarray(logs[k])
    return logs
