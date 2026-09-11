"""MVEE collision envelopes and conservative interpolated-pose checks.

Certificates concern the supplied ellipsoids and interpolated Cartesian poses,
not unobserved obstacle surfaces, full-arm kinematics, or controller tracking.
"""
from dataclasses import dataclass
import heapq
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class Ellipsoid:
    center: np.ndarray
    rotation: np.ndarray
    axes: np.ndarray

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        self.rotation = np.asarray(self.rotation, dtype=float)
        self.axes = np.asarray(self.axes, dtype=float)
        if self.center.shape != (3,) or self.rotation.shape != (3, 3) or self.axes.shape != (3,):
            raise ValueError('Invalid ellipsoid dimensions')
        if not all(np.isfinite(x).all() for x in (self.center, self.rotation, self.axes)) or np.any(self.axes <= 0):
            raise ValueError('Nonfinite or nonpositive ellipsoid')
        if not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-6) or np.linalg.det(self.rotation) < 0:
            raise ValueError('Invalid rotation')

    def radius(self, direction):
        return np.linalg.norm((np.asarray(direction) @ self.rotation) * self.axes, axis=-1)

    def at(self, position, rotation):
        return Ellipsoid(position + rotation @ self.center, rotation @ self.rotation, self.axes)


class Geometry:
    def __init__(self, obstacle, actors, margin=.01):
        if not actors or margin < 0:
            raise ValueError('Actors and nonnegative margin required')
        self.obstacle, self.actors, self.margin = obstacle, actors, margin

    def segment_clearance(self, p0, r0, p1, r1):
        """Lower bound using midpoint separating planes and rigid-motion bounds.

        Bound movement over each half subinterval by translation plus rotation
        angle times the furthest actor point's distance from the EEF origin.
        A positive bound certifies the whole interpolated segment. Negative is
        inconclusive; it is not an exact penetration depth.
        """
        p0, p1 = np.asarray(p0), np.asarray(p1)
        distance = np.linalg.norm(p1-p0)
        angle = (Rotation.from_matrix(r1) * Rotation.from_matrix(r0).inv()).magnitude()
        n = max(1, int(np.ceil(distance/.005)), int(np.ceil(angle/.04)))
        ts = (np.arange(n)+.5)/n
        rotations = Slerp([0., 1.], Rotation.from_matrix(np.stack([r0, r1])))(ts).as_matrix()
        positions = p0 + ts[:, None]*(p1-p0)
        lower = np.inf
        for actor in self.actors:
            centers = positions + np.einsum('tij,j->ti', rotations, actor.center)
            rots = rotations @ actor.rotation
            delta = centers-self.obstacle.center
            lengths = np.linalg.norm(delta, axis=1)
            directions = delta / np.maximum(lengths[:, None], 1e-12)
            directions[lengths < 1e-12] = [1., 0., 0.]
            radii = np.linalg.norm(np.einsum('ti,tij->tj', directions, rots)*actor.axes, axis=1)
            movement = distance/(2*n) + (np.linalg.norm(actor.center)+max(actor.axes))*angle/(2*n)
            gaps = lengths-radii-self.obstacle.radius(directions)-self.margin-movement
            lower = min(lower, float(gaps.min()))
        return lower

    def chunk_clearance(self, position, rotation, actions):
        lower = np.inf
        for action in actions:
            next_rotation = Rotation.from_rotvec(action[3:6]).as_matrix()
            lower = min(lower, self.segment_clearance(position, rotation, action[:3], next_rotation))
            position, rotation = action[:3], next_rotation
        return lower

    def bypass(self, position, rotation, goal_xy, workspace_half):
        """Shortest visible route around an inflated rectangle in the goal frame.

        Four rectangle corners are candidate waypoints, not a tuned margin
        sweep. Edges must pass the 3-D interpolated ellipsoid check. The fixed
        route commits to its chosen side across subsequent action chunks.
        """
        position = np.asarray(position)
        forward = np.r_[np.asarray(goal_xy)-position[:2], 0.]
        if np.linalg.norm(forward) < 1e-6:
            return None
        forward /= np.linalg.norm(forward)
        lateral = np.cross([0., 0., 1.], forward)
        basis = np.stack([forward, lateral], axis=1)
        obstacle_center = self.obstacle.center @ basis
        obstacle_radius = self.obstacle.radius(basis.T)
        lows, highs = [], []
        for a in self.actors:
            offset = rotation @ a.center
            world = a.at(np.zeros(3), rotation)
            radius = world.radius(basis.T)
            lows.append(offset @ basis-radius)
            highs.append(offset @ basis+radius)
        # Extra 15 mm gives graph corners space for the conservative checker.
        low = obstacle_center-obstacle_radius-np.max(highs, axis=0)-self.margin-.015
        high = obstacle_center+obstacle_radius-np.min(lows, axis=0)+self.margin+.015
        start_uv = position @ basis
        goal_uv = np.r_[goal_xy, position[2]] @ basis
        end_u = high[0]+.03
        # Do not plan beyond the task goal, or start a second bypass after it.
        if end_u >= goal_uv[0] or start_uv[0] >= end_u:
            return None
        end = position + forward*(end_u-start_uv[0])
        nodes = [position.copy(), end]
        for u, v in ((low[0], low[1]), (high[0], low[1]), (low[0], high[1]), (high[0], high[1])):
            q = basis @ [u, v]
            q[2] = position[2]
            nodes.append(q)
        valid = [bool(np.all(np.abs(q[:2]) <= workspace_half)) for q in nodes]
        if not valid[0] or not valid[1]:
            return None
        graph = [[] for _ in nodes]
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                if valid[i] and valid[j] and self.segment_clearance(nodes[i], rotation, nodes[j], rotation) >= 0:
                    length = float(np.linalg.norm(nodes[j]-nodes[i]))
                    graph[i].append((j, length)); graph[j].append((i, length))
        queue = [(0., 0, [0])]; best = {}
        while queue:
            cost, i, path = heapq.heappop(queue)
            if i in best:
                continue
            best[i] = cost
            if i == 1:
                route = np.asarray([nodes[k] for k in path])
                signed = (route-self.obstacle.center) @ lateral
                side = int(np.sign(signed[np.argmax(np.abs(signed))]))
                return {'points': route, 'side': side, 'length': cost, 'forward': forward}
            for j, weight in graph[i]:
                if j not in best:
                    heapq.heappush(queue, (cost+weight, j, path+[j]))
        return None
