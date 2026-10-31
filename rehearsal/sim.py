"""Execute high-level actions on the Panda in MuJoCo and record what happened.

Grasping is a suction weld (an explicit assumption); everything else — reachability via IK,
arm/object collisions, object motion after release — comes from the physics engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .scene import footprint_radius, object_height

HOME_Q = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
TCP_OFFSET = 0.1034          # hand origin -> fingertip center, along the hand z axis
SPEED = 0.35                 # m/s for cartesian moves
GAP = 0.004                  # suction cup stand-off above the object's top face


class IKFailure(Exception):
    """Raised when a motion cannot be executed. `info` carries machine-readable evidence."""

    def __init__(self, msg: str, info: dict | None = None):
        super().__init__(msg)
        self.info = info or {}


# Human-readable names for the Panda's links, so a language model can reason about which part
# of the arm touched something (the held object hangs below the tool tip; the arm is above/behind).
LINK_ROLE = {"link0": "base", "link1": "shoulder", "link2": "shoulder", "link3": "upper arm",
             "link4": "elbow", "link5": "forearm", "link6": "wrist", "link7": "wrist flange",
             "hand": "gripper body", "left_finger": "suction tip/finger", "right_finger": "suction tip/finger"}


@dataclass
class Event:
    t: float
    kind: str
    detail: dict = field(default_factory=dict)

    def to_dict(self):
        return {"t": round(self.t, 3), "kind": self.kind, **self.detail}


class Sim:
    def __init__(self, model: mujoco.MjModel, scene: dict, render: bool = False):
        self.m, self.scene = model, scene
        self.d = mujoco.MjData(model)
        self.ik_d = mujoco.MjData(model)
        self.objs = {o["id"]: o for o in scene["objects"]}
        self.hand = model.body("hand").id
        self.link0 = model.body("link0").id
        self.table = model.body("table").id
        self.body_of = {oid: model.body(oid).id for oid in self.objs}
        self.eq_of = {oid: model.equality(f"suction_{oid}").id for oid in self.objs}
        self.jlo, self.jhi = model.jnt_range[:7, 0], model.jnt_range[:7, 1]

        self.d.qpos[:7] = HOME_Q
        self.d.ctrl[:7] = HOME_Q
        self.d.ctrl[7] = 0.0  # fingers closed: the hand acts as a suction tool
        mujoco.mj_forward(model, self.d)
        self.down_quat = self.d.xquat[self.hand].copy()  # top-down tool orientation
        for _ in range(300):  # let objects settle on the table
            mujoco.mj_step(model, self.d)
        self.start_pos = {oid: self.d.xpos[b].copy() for oid, b in self.body_of.items()}
        self.held: str | None = None
        self.settle_motion: dict[str, float] = {}
        self.perception_error: dict[str, np.ndarray] = {}
        self.placed_by_robot: set[str] = set()
        self.events: list[Event] = []
        self.phase = "settle"
        self._geom_root = [self._root(g) for g in range(model.ngeom)]
        self._geom_link = [model.body(model.geom_bodyid[g]).name if r == "robot_arm" else None
                           for g, r in enumerate(self._geom_root)]
        self.expect_contact: str | None = None
        self._seen_contacts: set = {k for k, _ in self._contact_pairs()}
        self.frames: list[np.ndarray] = []
        self.traj: list[list[float]] = []   # recorded poses, see start_recording
        self._record_bodies: list[int] = []
        self._record_every = 0
        self._record_pos_only = False
        self.renderer = mujoco.Renderer(model, 360, 480) if render else None
        self._cam = mujoco.MjvCamera()
        self._cam.lookat[:] = [0.45, 0.0, 0.08]
        self._cam.distance, self._cam.azimuth, self._cam.elevation = 1.55, 205, -32

    # ---------------------------------------------------------------- kinematics
    def tcp(self, d=None) -> np.ndarray:
        d = self.d if d is None else d
        return d.xpos[self.hand] + d.xmat[self.hand].reshape(3, 3)[:, 2] * TCP_OFFSET

    def ik(self, target: np.ndarray, q0: np.ndarray, tol=2e-3) -> np.ndarray:
        """Damped least squares IK for TCP position + top-down orientation."""
        m, d = self.m, self.ik_d
        best, best_err = None, np.inf
        seeds = [q0, HOME_Q] + [HOME_Q + np.random.default_rng(i).normal(0, 0.4, 7) for i in range(4)]
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for seed in seeds:
            d.qpos[:] = self.d.qpos
            q = np.clip(seed.copy(), self.jlo, self.jhi)
            for _ in range(150):
                d.qpos[:7] = q
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                p = self.tcp(d)
                ep = target - p
                qerr = np.zeros(4)
                mujoco.mju_mulQuat(qerr, self.down_quat, _conj(d.xquat[self.hand]))
                er = np.zeros(3)
                mujoco.mju_quat2Vel(er, qerr, 1.0)
                err = np.linalg.norm(ep) + 0.05 * np.linalg.norm(er)
                if err < tol:
                    break
                mujoco.mj_jac(m, d, jacp, jacr, p, self.hand)
                J = np.vstack([jacp[:, :7], 0.5 * jacr[:, :7]])
                e = np.concatenate([ep, 0.5 * er])
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
                q = np.clip(q + dq, self.jlo, self.jhi)
            if err < best_err:
                best, best_err = q.copy(), err
            if err < tol:
                return q
        raise IKFailure(f"IK residual {best_err*100:.1f} cm at {np.round(target, 3).tolist()}",
                        {"target_xyz": [round(float(v), 3) for v in target],
                         "residual_cm": round(float(best_err) * 100, 1)})

    def max_reachable_z(self, xy, z_hi: float) -> float | None:
        """Highest tool-tip height (<= z_hi) reachable top-down above xy, or None if none is.
        Only called after an IK failure, to tell the planner what height would have worked."""
        q0 = self.d.qpos[:7].copy()

        def ok(z):
            try:
                self.ik(np.array([xy[0], xy[1], z]), q0)
                return True
            except IKFailure:
                return False
        lo, hi = 0.03, z_hi
        if not ok(lo):
            return None
        for _ in range(5):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        return round(lo, 2)

    def reachable(self, oid: str) -> bool:
        top = self.d.xpos[self.body_of[oid]].copy()
        top[2] += object_height(self.objs[oid]) / 2 + GAP
        try:
            self.ik(top, self.d.qpos[:7].copy())
            return True
        except IKFailure:
            return False

    # ---------------------------------------------------------------- motion
    def move_to(self, target, label: str):
        """Straight-line TCP move, IK every ~2 cm, joint-space servo in between."""
        self.phase = label
        start = self.tcp()
        dist = np.linalg.norm(target - start)
        n = max(1, int(dist / 0.02))
        q = self.d.ctrl[:7].copy()
        for k in range(1, n + 1):
            wp = start + (target - start) * k / n
            try:
                q_next = self.ik(wp, q)
            except IKFailure as e:
                xy = wp[:2]
                zmax = self.max_reachable_z(xy, float(wp[2]))
                e.info.update({"phase": label, "segment_from": [round(float(v), 3) for v in start],
                               "segment_to": [round(float(v), 3) for v in target],
                               "progress": round(k / n, 2),
                               "horizontal_dist_from_base": round(float(np.linalg.norm(xy)), 3),
                               "max_reachable_tool_height_here": zmax})
                hint = (f"no top-down pose reaches xy {np.round(xy, 3).tolist()} at any height "
                        f"({np.linalg.norm(xy):.2f} m from the robot base)" if zmax is None else
                        f"at xy {np.round(xy, 3).tolist()} the tool tip can reach at most "
                        f"z={zmax:.2f} m, the move asked for z={wp[2]:.2f} m")
                e.args = (f"IK failed during '{label}': {hint}",)
                self.events.append(Event(self.d.time, "ik_failure", {"msg": str(e), **e.info}))
                raise
            steps = max(1, int((dist / n) / SPEED / self.m.opt.timestep))
            for s in range(1, steps + 1):
                self.d.ctrl[:7] = q + (q_next - q) * s / steps
                self._step()
            q = q_next
        self.wait(0.15)

    def wait(self, seconds: float):
        for _ in range(int(seconds / self.m.opt.timestep)):
            self._step()

    def grasp(self, oid: str):
        tcp = self.tcp()
        pos = self.d.xpos[self.body_of[oid]]
        top = pos[2] + object_height(self.objs[oid]) / 2
        off = float(np.linalg.norm(tcp[:2] - pos[:2]))
        if abs(tcp[2] - GAP - top) > 0.02 or off > footprint_radius(self.objs[oid]) * 0.8:
            info = {"obj": oid, "xy_offset_cm": round(off * 100, 1),
                    "z_offset_cm": round(float(tcp[2] - GAP - top) * 100, 1)}
            self.events.append(Event(self.d.time, "grasp_miss", info))
            raise IKFailure(f"suction cup missed {oid}'s top face (xy offset {off*100:.1f} cm, "
                            f"height offset {info['z_offset_cm']} cm)", {"phase": f"grasp {oid}", **info})
        eq = self.eq_of[oid]
        hb, ob = self.hand, self.body_of[oid]
        # relpose = pose of the object expressed in the hand frame
        rel_p = self.d.xmat[hb].reshape(3, 3).T @ (self.d.xpos[ob] - self.d.xpos[hb])
        rel_q = np.zeros(4)
        mujoco.mju_mulQuat(rel_q, _conj(self.d.xquat[hb]), self.d.xquat[ob])
        self.m.eq_data[eq, 0:3] = 0
        self.m.eq_data[eq, 3:6] = rel_p
        self.m.eq_data[eq, 6:10] = rel_q
        self.m.eq_data[eq, 10] = 1.0
        self.d.eq_active[eq] = 1
        self.held = oid
        self.events.append(Event(self.d.time, "grasp", {"obj": oid}))

    def release(self):
        if self.held:
            self.d.eq_active[self.eq_of[self.held]] = 0
            self.events.append(Event(self.d.time, "release", {"obj": self.held}))
            self.held = None

    # ---------------------------------------------------------------- logging
    def _root(self, geom: int) -> str:
        b = self.m.body_rootid[self.m.geom_bodyid[geom]]
        if b == self.link0:
            return "robot_arm"
        if b == self.table:
            return "table"
        if b == 0:
            return "floor"
        return self.m.body(b).name

    def _contact_pairs(self):
        """Yields (sorted root pair, robot link name or None, contact index)."""
        roots, links = self._geom_root, self._geom_link
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g1, g2 = c.geom1, c.geom2
            ra, rb = roots[g1], roots[g2]
            if ra == rb:
                continue
            link = links[g1] or links[g2]
            a, b = (ra, rb) if ra < rb else (rb, ra)
            yield (a, b, link), i

    def _step(self):
        mujoco.mj_step(self.m, self.d)
        if self.d.ncon:
            self._log_contacts()
        step = int(round(self.d.time / self.m.opt.timestep))
        if self.renderer is not None and step % 25 == 0:
            self.renderer.update_scene(self.d, self._cam)
            self.frames.append(self.renderer.render().copy())
        if self._record_every and step % self._record_every == 0:
            self._record()

    def _log_contacts(self):
        for key, i in self._contact_pairs():
            if key in self._seen_contacts:
                continue
            self._seen_contacts.add(key)
            a, b, link = key
            pair = (a, b)
            if link is not None and self.held in pair:
                continue  # the tool touching what it holds is expected
            if link is not None and self.expect_contact in pair and link in _TOOL_LINKS:
                continue  # the suction tip touching the object it is about to pick
            if pair == ("floor", "table"):
                continue
            detail = {"between": list(pair), "while_holding": self.held,
                      "tcp_z": round(float(self.tcp()[2]), 3), "phase": self.phase,
                      "at_xyz": [round(float(v), 3) for v in self.d.contact[i].pos]}
            if link is not None:
                detail["link"] = link
                detail["link_role"] = LINK_ROLE.get(link, link)
            self.events.append(Event(self.d.time, "contact", detail))

    # ---------------------------------------------------------------- recording
    def start_recording(self, mode: str):
        """'full': every body's pose at 50 Hz (3D replay). 'objects': object positions at 10 Hz (trails)."""
        if mode == "full":
            self._record_bodies = list(range(1, self.m.nbody))
            self._record_every, self._record_pos_only = 10, False
        else:
            self._record_bodies = [self.body_of[o] for o in self.objs]
            self._record_every, self._record_pos_only = 50, True
        self._record()

    def _record(self):
        row = [round(float(self.d.time), 3)]
        for b in self._record_bodies:
            row += [round(float(v), 4) for v in self.d.xpos[b]]
            if not self._record_pos_only:
                row += [round(float(v), 4) for v in self.d.xquat[b]]
        self.traj.append(row)

    # ---------------------------------------------------------------- state
    def obj_state(self, oid: str) -> dict:
        b = self.body_of[oid]
        return {"pos": self.d.xpos[b].copy(), "zaxis": self.d.xmat[b].reshape(3, 3)[:, 2].copy()}


_TOOL_LINKS = {"hand", "left_finger", "right_finger"}


def _conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])
