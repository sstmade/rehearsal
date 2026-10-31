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
    pass


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
        self._seen_contacts: set = set(self._contact_pairs())
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
        raise IKFailure(f"IK residual {best_err*100:.1f} cm at {np.round(target, 3).tolist()}")

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
        start = self.tcp()
        dist = np.linalg.norm(target - start)
        n = max(1, int(dist / 0.02))
        q = self.d.ctrl[:7].copy()
        for k in range(1, n + 1):
            wp = start + (target - start) * k / n
            try:
                q_next = self.ik(wp, q)
            except IKFailure as e:
                self.events.append(Event(self.d.time, "ik_failure", {"phase": label, "msg": str(e)}))
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
            self.events.append(Event(self.d.time, "grasp_miss", {
                "obj": oid, "xy_offset_cm": round(off * 100, 1)}))
            raise IKFailure(f"suction cup missed {oid}'s top face (xy offset {off*100:.1f} cm)")
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
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            a, b = sorted((self._root(c.geom1), self._root(c.geom2)))
            if a != b:
                yield (a, b)

    def _step(self):
        mujoco.mj_step(self.m, self.d)
        for pair in self._contact_pairs():
            if pair in self._seen_contacts:
                continue
            self._seen_contacts.add(pair)
            if self.held in pair and "robot_arm" in pair:
                continue  # the tool touching what it holds is expected
            if pair == ("floor", "table"):
                continue
            self.events.append(Event(self.d.time, "contact", {
                "between": list(pair), "while_holding": self.held,
                "tcp_z": round(float(self.tcp()[2]), 3)}))
        step = int(round(self.d.time / self.m.opt.timestep))
        if self.renderer is not None and step % 25 == 0:
            self.renderer.update_scene(self.d, self._cam)
            self.frames.append(self.renderer.render().copy())
        if self._record_every and step % self._record_every == 0:
            self._record()

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


def _conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])
