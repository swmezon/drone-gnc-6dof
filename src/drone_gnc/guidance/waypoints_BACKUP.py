from dataclasses import dataclass
import numpy as np

@dataclass
class WaypointGuidance:
    waypoints: np.ndarray
    acceptance_radius: float=0.30
    cruise_speed: float=1.0
    index: int=0

    def update(self, position: np.ndarray):
        target=self.waypoints[self.index]
        delta=target-position; dist=float(np.linalg.norm(delta))
        if dist < self.acceptance_radius and self.index < len(self.waypoints)-1:
            self.index += 1; target=self.waypoints[self.index]; delta=target-position; dist=float(np.linalg.norm(delta))
        vel_ref=np.zeros(3) if dist < 1e-9 else self.cruise_speed*delta/dist
        if self.index==len(self.waypoints)-1 and dist < self.acceptance_radius: vel_ref*=0.0
        yaw_ref=0.0
        return target.copy(),vel_ref,yaw_ref,self.index,dist
