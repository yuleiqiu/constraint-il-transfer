"""Transport-only, oracle-geometry local planner for absolute EEF actions."""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from robomimic.utils.ellipsoid_guidance_utils import fit_robosuite_geom_ellipsoid
from geometry import Ellipsoid, Geometry


def fitted(raw, ids, **kwargs):
    e = fit_robosuite_geom_ellipsoid(raw, ids, tolerance=1e-3, max_iterations=3000, padding=.002, **kwargs)
    return Ellipsoid(e.center, e.rotation, e.semi_axes)


class LocalPlanner:
    def __init__(self, raw, obstacle_ids):
        self.raw = raw
        self.obstacle = fitted(raw, obstacle_ids)
        # Static obstacle envelope includes the support via obstacle_ids.
        self.gripper = None
        self.route = None
        self.index = 1
        self.mode = 'source'
        self.fixed_rotation = None
        self.triggers = 0
        self.target = fitted(raw, [raw.sim.model.geom_name2id(n) for n in raw.can.contact_geoms],
                             reference_position=raw.sim.data.body_xpos[raw.can_body_id],
                             reference_rotation=np.asarray(raw.sim.data.body_xmat[raw.can_body_id]).reshape(3, 3))

    def pose(self):
        arm = self.raw.robots[0].arms[0]
        site = self.raw.robots[0].eef_site_id[arm]
        return self.raw.sim.data.site_xpos[site].copy(), self.raw.sim.data.site_xmat[site].reshape(3, 3).copy()

    def geometry(self, p, r):
        if self.gripper is None:
            model = self.raw.sim.model
            ids = [i for i in range(model.ngeom) if (model.geom_id2name(i) or '').startswith('gripper')
                   and (model.geom_contype[i] or model.geom_conaffinity[i])]
            self.gripper = fitted(self.raw, ids, reference_position=p, reference_rotation=r)
        bpos = self.raw.sim.data.body_xpos[self.raw.can_body_id]
        brot = self.raw.sim.data.body_xmat[self.raw.can_body_id].reshape(3, 3)
        target_world = self.target.at(bpos, brot)
        target_local = Ellipsoid(r.T @ (target_world.center-p), r.T @ target_world.rotation, target_world.axes)
        return Geometry(self.obstacle, [self.gripper, target_local], margin=.01)

    def control(self, source):
        started = time.monotonic()
        p, r = self.pose()
        grasp = bool(self.raw._check_grasp(self.raw.robots[0].gripper, self.raw.can))
        lifted = self.raw.sim.data.body_xpos[self.raw.can_body_id, 2] > .88
        info = {'mode_before': self.mode, 'grasp': grasp, 'lifted': bool(lifted), 'trigger': False}
        def finish(output, reason):
            info.update(mode=self.mode, reason=reason, planning_seconds=time.monotonic()-started,
                        changed=bool(not np.array_equal(output, source)))
            return output, info
        if not grasp:
            if self.mode in ('bypass', 'blocked'):
                self.mode = 'lost_grasp'
            return finish(source, 'no_grasp_source')
        if self.mode == 'lost_grasp' or (self.mode == 'source' and not lifted):
            return finish(source, 'not_transport')
        geometry = self.geometry(p, r)
        clearance = geometry.chunk_clearance(p, r, source)
        info['source_clearance_lower_m'] = clearance
        if self.mode in ('source', 'handed_back'):
            if clearance >= 0:
                return finish(source, 'source_chunk_clear')
            route = geometry.bypass(p, r, self.raw.goal_xy, self.raw.table_full_size[:2]/2-.025)
            self.triggers += 1
            info['trigger'] = True
            self.fixed_rotation = r.copy()
            self.hold_position = p.copy()
            if route is None:
                self.mode = 'blocked'
            else:
                self.mode = 'bypass'; self.route = route; self.index = 1
                info['new_route'] = route
        hold = np.tile(np.r_[p, Rotation.from_matrix(r).as_rotvec(), 1.], (len(source), 1))
        if self.mode == 'blocked':
            # Explicit infeasibility outcome; do not execute a known risky chunk.
            return finish(hold, 'no_feasible_route_hold')
        points = self.route['points']
        while self.index < len(points) and np.linalg.norm(p-points[self.index]) < .008:
            self.index += 1
        info.update(route_index=self.index, route_side=self.route['side'])
        if self.index == len(points):
            progress = float(np.mean((source[:, :3]-p) @ self.route['forward']))
            info['source_forward_m'] = progress
            # A short-horizon handoff gate, not a learned recoverability claim.
            if clearance >= 0 and progress >= -.005:
                self.mode = 'handed_back'
                return finish(source, 'route_complete_source_clear')
            return finish(hold, 'route_complete_wait_for_source')
        waypoint = points[self.index]
        delta = waypoint-p
        length = np.linalg.norm(delta)
        output = np.tile(np.r_[p, Rotation.from_matrix(self.fixed_rotation).as_rotvec(), 1.], (len(source), 1))
        for i in range(len(output)):
            output[i, :3] = p + delta*min(1., .006*(i+1)/max(length, 1e-12))
        checked = geometry.chunk_clearance(p, r, output)
        info['output_clearance_lower_m'] = checked
        if checked < 0:
            return finish(hold, 'tracking_or_geometry_invalidated_route')
        return finish(output, 'follow_fixed_bypass')
