import json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from drone_gnc.vehicles import QuadrotorModel
from drone_gnc.control.cascaded import CascadedController
from drone_gnc.guidance.waypoints import WaypointGuidance
from drone_gnc.estimation.sensors import ImuGpsSensorSuite
from drone_gnc.estimation.error_state_ekf import ErrorStateEKF15
from drone_gnc.simulation.mission import run_waypoint_mission
from drone_gnc.evaluation.metrics import trajectory_metrics

for d in ['results/animations','results/guidance','results/estimation','results/data']:(ROOT/d).mkdir(parents=True,exist_ok=True)
waypoints=np.array([[0,0,1.5],[2,0,1.5],[2,2,1.5],[0,2,1.5],[0,0,1.5]])
vehicle=QuadrotorModel(); ctrl=CascadedController(vehicle.params.mass,vehicle.params.gravity)
guidance=WaypointGuidance(waypoints,acceptance_radius=0.25,cruise_speed=0.60)
sensors=ImuGpsSensorSuite(seed=24); ekf=ErrorStateEKF15(dt=0.01)
logs=run_waypoint_mission(vehicle,ctrl,guidance,sensors,ekf,duration=40.0,dt=0.01,gps_hz=10.0)
metrics=trajectory_metrics(logs['state'][:,0:3],logs['ref'],logs['wp_index'])
est_err=logs['estimate'][:,0:3]-logs['state'][:,0:3]
metrics['ekf_position_rmse_m']=float(np.sqrt(np.mean(np.sum(est_err**2,axis=1))))
# Cross-track error to the piecewise-linear waypoint path.
def point_segment_distance(p,a,b):
    ab=b-a; den=float(ab@ab)
    if den<1e-12: return float(np.linalg.norm(p-a))
    q=np.clip(((p-a)@ab)/den,0.0,1.0)
    return float(np.linalg.norm(p-(a+q*ab)))
path_err=[]
for pnt in logs['state'][:,0:3]:
    path_err.append(min(point_segment_distance(pnt,waypoints[j],waypoints[j+1]) for j in range(len(waypoints)-1)))
path_err=np.asarray(path_err)
metrics['cross_track_rmse_m']=float(np.sqrt(np.mean(path_err**2)))
metrics['cross_track_max_m']=float(np.max(path_err))
metrics['max_thrust_N']=float(np.max(logs['control'][:,0])); metrics['max_torque_Nm']=float(np.max(np.abs(logs['control'][:,1:4])))
metrics['torque_saturation_pct']=float(100*np.mean(np.any(np.isclose(np.abs(logs['control'][:,1:4]),1.2,atol=1e-9),axis=1)))
metrics['duration_s']=float(logs['t'][-1]); metrics['sample_rate_hz']=100.0; metrics['gps_rate_hz']=10.0
(ROOT/'results/data/waypoint_metrics.json').write_text(json.dumps(metrics,indent=2))
np.savetxt(ROOT/'results/data/waypoint_log.csv',np.c_[logs['t'],logs['state'],logs['estimate'],logs['ref'],logs['control']],delimiter=',',header='t,x,y,z,u,v,w,phi,theta,psi,p,q,r,est_x,est_y,est_z,est_u,est_v,est_w,est_phi,est_theta,est_psi,est_p,est_q,est_r,ref_x,ref_y,ref_z,T,tau_x,tau_y,tau_z',comments='')

fig=plt.figure(figsize=(8,6)); ax=fig.add_subplot(111,projection='3d'); ax.plot(logs['state'][:,0],logs['state'][:,1],logs['state'][:,2],label='Actual'); ax.plot(waypoints[:,0],waypoints[:,1],waypoints[:,2],'o--',label='Waypoints'); ax.set_xlabel('X [m]');ax.set_ylabel('Y [m]');ax.set_zlabel('Z [m]');ax.set_title('Autonomous Waypoint Tracking');ax.legend();fig.tight_layout();fig.savefig(ROOT/'results/guidance/waypoint_tracking_3d.png',dpi=180);plt.close(fig)

fig,axs=plt.subplots(3,1,figsize=(9,7),sharex=True)
for i,lbl in enumerate(['X','Y','Z']):
    axs[i].plot(logs['t'],logs['state'][:,i],label='True'); axs[i].plot(logs['t'],logs['estimate'][:,i],label='EKF',alpha=.8); axs[i].set_ylabel(f'{lbl} [m]'); axs[i].grid(True,alpha=.3)
axs[0].legend(); axs[-1].set_xlabel('Time [s]'); fig.suptitle('GPS/IMU State Estimation');fig.tight_layout();fig.savefig(ROOT/'results/estimation/ekf_position_tracking.png',dpi=180);plt.close(fig)

# Recruiter-facing GIF: actual vehicle path following waypoints.
stride=12; pts=logs['state'][::stride,0:3]
fig=plt.figure(figsize=(7,5)); ax=fig.add_subplot(111,projection='3d'); ax.plot(waypoints[:,0],waypoints[:,1],waypoints[:,2],'o--',label='Waypoint path'); trail,=ax.plot([],[],[],lw=2,label='Vehicle path'); marker,=ax.plot([],[],[],'o',markersize=7,label='Quadrotor'); ax.set_xlim(-.5,2.5);ax.set_ylim(-.5,2.5);ax.set_zlim(0,2.2);ax.set_xlabel('X [m]');ax.set_ylabel('Y [m]');ax.set_zlabel('Z [m]');ax.set_title('Closed-Loop Waypoint Tracking');ax.legend(loc='upper right')
def update(i):
    q=pts[:i+1]; trail.set_data(q[:,0],q[:,1]); trail.set_3d_properties(q[:,2]); marker.set_data([q[-1,0]],[q[-1,1]]); marker.set_3d_properties([q[-1,2]]); return trail,marker
ani=FuncAnimation(fig,update,frames=len(pts),interval=45,blit=False);ani.save(ROOT/'results/animations/waypoint_tracking.gif',writer=PillowWriter(fps=20));plt.close(fig)
print(json.dumps(metrics,indent=2))
