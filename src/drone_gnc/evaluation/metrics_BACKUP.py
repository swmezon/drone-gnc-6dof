import numpy as np

def rmse(err): return float(np.sqrt(np.mean(np.sum(np.asarray(err)**2,axis=1))))
def axis_rmse(err): return np.sqrt(np.mean(np.asarray(err)**2,axis=0))

def trajectory_metrics(pos, refs, waypoint_indices):
    e=np.asarray(pos)-np.asarray(refs)
    d=np.linalg.norm(e,axis=1)
    return {"position_rmse_m":float(np.sqrt(np.mean(d*d))),"mean_error_m":float(np.mean(d)),"max_error_m":float(np.max(d)),"final_error_m":float(d[-1]),"waypoints_reached":int(np.max(waypoint_indices)+1)}
