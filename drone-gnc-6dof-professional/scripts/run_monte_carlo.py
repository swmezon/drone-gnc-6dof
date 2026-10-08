import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from drone_gnc.vehicles.quadrotor import QuadrotorModel,QuadrotorParams
from drone_gnc.control.cascaded import CascadedController
from drone_gnc.guidance.waypoints import WaypointGuidance
from drone_gnc.simulation.mission import run_waypoint_mission
from drone_gnc.evaluation.metrics import trajectory_metrics
rng=np.random.default_rng(2026);N=30;rows=[]
wp=np.array([[0,0,1.5],[2,0,1.5],[2,2,1.5],[0,2,1.5],[0,0,1.5]])
for i in range(N):
    p=QuadrotorParams(mass=1.5*rng.uniform(.9,1.1),inertia_x=.03*rng.uniform(.85,1.15),inertia_y=.03*rng.uniform(.85,1.15),inertia_z=.05*rng.uniform(.85,1.15))
    v=QuadrotorModel(p); c=CascadedController(p.mass,p.gravity); g=WaypointGuidance(wp.copy(),.32,.80)
    log=run_waypoint_mission(v,c,g,duration=36,dt=.04)
    m=trajectory_metrics(log['state'][:,:3],log['ref'],log['wp_index']); success=(m['waypoints_reached']==len(wp) and m['final_error_m']<.45)
    rows.append([m['position_rmse_m'],m['max_error_m'],m['final_error_m'],int(success)])
a=np.array(rows);summary={'cases':N,'success_rate_pct':float(100*a[:,3].mean()),'median_position_rmse_m':float(np.median(a[:,0])),'p95_position_rmse_m':float(np.percentile(a[:,0],95)),'worst_final_error_m':float(np.max(a[:,2]))}
(ROOT/'results/data/monte_carlo_summary.json').write_text(json.dumps(summary,indent=2));np.savetxt(ROOT/'results/data/monte_carlo.csv',a,delimiter=',',header='position_rmse_m,max_error_m,final_error_m,success',comments='');print(json.dumps(summary,indent=2))
