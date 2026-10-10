"""Closed-loop waypoint tours: the command comes from where the robot is, not from a clock.

``WaypointTour`` turns a list of waypoints (relative to the pose the tour starts from) into
velocity commands each policy step, the way a teleoperation layer or a benchmark harness
does: a proportional controller on the position and heading error in the body frame, with
hysteresis on "reached", then clamped to the policy's command limits. It records the error
at the end of each waypoint's time slot.

``bench_waypoints``, ``bench_arm_walk`` and ``bench_punches`` build the tour, the arm motion
and the punches of rhoyn's teleop-walking-benchmark (main.cpp): 18 waypoints, 5 s each, in a 0.75 m disc around
(0.3, 0); arms walking at random around STANCE, kept out of the way of the body and the
head; a punch on a random link every 5 s whose force ceiling ramps from 1/3 to 1 of 500 N
over 60 s. They draw from the benchmark's own random streams (``BenchRandom``: its
std::mt19937 and seed_seq, reproduced), so seed n here is the benchmark's run id n: the same
waypoints, the same arm targets and the same punches, and a run here pairs with that run there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .runner import External, Push, Start


def _yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


@dataclass
class WaypointTour:
    waypoints: list[tuple[float, float, float]]  # (x, y, yaw) relative to the start pose
    point_s: float = 5.0
    start_s: float = 0.0
    kp_pos: float = 1.5
    kp_yaw: float = 1.5
    vx: tuple[float, float] = (-0.5, 1.0)
    vy_abs: float = 0.3
    wz_abs: float = 0.2
    speed_norm: float = 0.0
    pos_enter: float = 0.10
    pos_exit: float = 0.20
    yaw_enter: float = 0.05
    yaw_exit: float = 0.12
    _anchor: tuple[float, float, float] | None = field(default=None, repr=False)
    _index: int = field(default=-1, repr=False)
    _pos_ok: bool = field(default=False, repr=False)
    _yaw_ok: bool = field(default=False, repr=False)
    _last: tuple[float, float] | None = field(default=None, repr=False)
    _ends: list[tuple[float, float]] = field(default_factory=list, repr=False)
    # The harness's task for the step just commanded (main.cpp h_task): distance to the
    # waypoint, yaw error, and the waypoint in the body frame; zeros with no waypoint.
    task: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def __call__(self, time: float, base: np.ndarray) -> np.ndarray:
        """``base`` is the free joint's qpos: position, then quaternion wxyz."""
        elapsed = time - self.start_s
        self.task = (0.0, 0.0, 0.0, 0.0)
        if elapsed < 0:
            return np.zeros(3)
        x, y, yaw = float(base[0]), float(base[1]), _yaw(base[3:7])
        if self._anchor is None:
            self._anchor = (x, y, yaw)
        index = int(elapsed / self.point_s)
        if index != self._index:
            if self._index >= 0 and self._last is not None:
                self._ends.append(self._last)
            self._index, self._pos_ok, self._yaw_ok = index, False, False
        if index >= len(self.waypoints):
            return np.zeros(3)
        ax, ay, ayaw = self._anchor
        wx, wy, wyaw = self.waypoints[index]
        tx = ax + wx * math.cos(ayaw) - wy * math.sin(ayaw)
        ty = ay + wx * math.sin(ayaw) + wy * math.cos(ayaw)
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        yaw_err = math.remainder(ayaw + wyaw - yaw, 2 * math.pi)
        self._last = (dist, yaw_err)
        self._pos_ok = dist <= self.pos_exit if self._pos_ok else dist < self.pos_enter
        self._yaw_ok = (
            abs(yaw_err) <= self.yaw_exit if self._yaw_ok else abs(yaw_err) < self.yaw_enter
        )
        c, s = math.cos(yaw), math.sin(yaw)
        self.task = (dist, yaw_err, c * dx + s * dy, -s * dx + c * dy)
        cmd = np.zeros(3)
        if not self._pos_ok:
            cmd[0] = self.kp_pos * (c * dx + s * dy)
            cmd[1] = self.kp_pos * (-s * dx + c * dy)
        if not self._yaw_ok:
            cmd[2] = self.kp_yaw * yaw_err
        norm = math.hypot(cmd[0], cmd[1])
        if self.speed_norm > 0 and norm > self.speed_norm:
            cmd[:2] *= self.speed_norm / norm
        cmd[0] = min(max(cmd[0], self.vx[0]), self.vx[1])
        cmd[1] = min(max(cmd[1], -self.vy_abs), self.vy_abs)
        cmd[2] = min(max(cmd[2], -self.wz_abs), self.wz_abs)
        return cmd

    def report(self, fell_at: float | None) -> dict[str, Any]:
        """Error at the end of each waypoint slot that finished before a fall, as the
        benchmark scores it: centimetres and degrees, averaged over finished slots."""
        ends = list(self._ends)
        finished = len(self.waypoints) if fell_at is None else len(ends)
        if fell_at is None and self._last is not None and len(ends) < len(self.waypoints):
            ends.append(self._last)
        ends = ends[:finished]
        pos = [d * 100 for d, _ in ends]
        yaw = [abs(e) * 180 / math.pi for _, e in ends]
        return {
            "outcome": "complete" if fell_at is None else "fell",
            "survival_s": None if fell_at is None else fell_at,
            "targets": len(ends),
            "pos_err_cm": float(np.mean(pos)) if pos else math.nan,
            "yaw_err_deg": float(np.mean(yaw)) if yaw else math.nan,
            "reached": sum(1 for d, e in ends if d < self.pos_enter and abs(e) < self.yaw_enter),
        }


# -- rhoyn/teleop-walking-benchmark ----------------------------------------------------------

BENCH_WAYPOINTS = 18
BENCH_POINT_S = 5.0
BENCH_CENTRE = (0.3, 0.0)
BENCH_RADIUS_M = 0.75
BENCH_FORCE_MAX_N = 500.0
BENCH_FORCE_SCALE_MIN = 0.5
BENCH_RAMP_FLOOR = 1.0 / 3.0
BENCH_RAMP_S = 60.0
BENCH_PUNCH_DELAY_S = 0.1
BENCH_PUNCH_S = 0.08
BENCH_ARM_STEP_RAD = 0.06
BENCH_ARM_WINDOW_S = 0.5
BENCH_ARM_DRAWS = 8
BENCH_ARM_MIRROR = (1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
BENCH_ARM_LEFT = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
BENCH_ARM_RIGHT = tuple(n.replace("left_", "right_") for n in BENCH_ARM_LEFT)
# The harness's starting pose for the arms (STANCE, MuJoCo motor order 15..28).
BENCH_ARM_STANCE_LEFT = (0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0)
# The benchmark's motors, in its order (mjwarp_capture.py JOINTS): a punch picks one by index.
BENCH_MOTORS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    *BENCH_ARM_LEFT,
    *BENCH_ARM_RIGHT,
)
# The harness's whole-body STANCE, in BENCH_MOTORS order: the crane holds it before release.
BENCH_STANCE = (
    *(-0.312, 0.0, 0.0, 0.669, -0.363, 0.0) * 2,
    0.0,
    0.0,
    0.0,
    *BENCH_ARM_STANCE_LEFT,
    *(m * v for m, v in zip(BENCH_ARM_MIRROR, BENCH_ARM_STANCE_LEFT)),
)
BENCH_CRANE_HEIGHT_M = 0.793
BENCH_FALL_PELVIS_Z = 0.20  # the harness ends a run when the pelvis is below this
# The harness clock: the walk starts after 3 s, physics steps 2 ms (a float32 in the MuJoCo
# Warp capture), and the controller runs on every 10th step. Control step k is at
# (_BENCH_S0 + 10 k) * dt - 3 s, which the arm walk's windows are counted from.
BENCH_INIT_S = 3.0
BENCH_PERIOD_S = 0.02
_BENCH_DT = float(np.float32(0.002))
_BENCH_PER_CONTROL = 10
_BENCH_S0 = 1500

BENCH_STREAM_TOUR = 2
BENCH_STREAM_PUNCH = 3
BENCH_STREAM_ARM = 4
_U32 = 0xFFFFFFFF


def _seed_seq(values: list[int], n: int = 624) -> list[int]:
    """std::seed_seq{values...}.generate() into n 32-bit words ([rand.util.seedseq])."""
    b = [0x8B8B8B8B] * n
    s = len(values)
    t = 11 if n >= 623 else 7 if n >= 68 else 5 if n >= 39 else 3 if n >= 7 else (n - 1) // 2
    p = (n - t) // 2
    q = p + t
    m = max(s + 1, n)

    def mix(x: int) -> int:
        return (x ^ (x >> 27)) & _U32

    for k in range(m):
        r1 = (1664525 * mix(b[k % n] ^ b[(k + p) % n] ^ b[(k - 1) % n])) & _U32
        if k == 0:
            r2 = (r1 + s) & _U32
        elif k <= s:
            r2 = (r1 + k % n + values[k - 1]) & _U32
        else:
            r2 = (r1 + k % n) & _U32
        b[(k + p) % n] = (b[(k + p) % n] + r1) & _U32
        b[(k + q) % n] = (b[(k + q) % n] + r2) & _U32
        b[k % n] = r2
    for k in range(m, m + n):
        r3 = (1566083941 * mix((b[k % n] + b[(k + p) % n] + b[(k - 1) % n]) & _U32)) & _U32
        r4 = (r3 - k % n) & _U32
        b[(k + p) % n] ^= r3
        b[(k + q) % n] ^= r4
        b[k % n] = r4
    return b


class BenchRandom:
    """One of the benchmark's random streams (main.cpp random_for): a std::mt19937 seeded by
    std::seed_seq{run_id, stream}, with the benchmark's draws on it (random_unit,
    random_between, random_below). The same run id and stream give the same numbers as the
    benchmark's C++."""

    def __init__(self, run_id: int, stream: int):
        key = _seed_seq([int(run_id) & _U32, int(stream) & _U32])
        if not key[0] & 0x80000000 and not any(key[1:]):
            key[0] = 0x80000000  # std::mt19937::seed's guard against an all-zero state
        self._bg = np.random.MT19937()
        self._bg.state = {
            "bit_generator": "MT19937",
            "state": {"key": np.array(key, dtype=np.uint32), "pos": 624},
        }

    def raw(self) -> int:
        return int(self._bg.random_raw())

    def unit(self) -> float:
        return (self.raw() >> 8) * (1.0 / 16777216.0)

    def units(self, n: int) -> np.ndarray:
        """``n`` unit draws in order, as an array."""
        return (np.asarray(self._bg.random_raw(n), dtype=np.uint64) >> 8).astype(float) * (
            1.0 / 16777216.0
        )

    def between(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.unit()

    def below(self, bound: int) -> int:
        span = 0x100000000 // bound
        limit = span * bound
        while True:
            draw = self.raw()
            if draw < limit:
                return draw // span


def bench_waypoints(seed: int) -> list[tuple[float, float, float]]:
    """The benchmark's tour for run id ``seed`` (main.cpp tour_make)."""
    rng = BenchRandom(seed, BENCH_STREAM_TOUR)
    out = []
    for _ in range(BENCH_WAYPOINTS):
        r = BENCH_RADIUS_M * math.sqrt(rng.unit())
        b = rng.between(0.0, 2.0 * math.pi)
        yaw = rng.between(0.0, 2.0 * math.pi)
        out.append(
            (
                BENCH_CENTRE[0] + r * math.cos(b),
                BENCH_CENTRE[1] + r * math.sin(b),
                math.remainder(yaw, 2.0 * math.pi),
            )
        )
    return out


def bench_motor_links(m: Any) -> list[tuple[str, str, tuple[float, float, float]]]:
    """For each of the benchmark's motors, in its order: the joint's child link, the link it
    hangs from, and the joint's anchor in the child link's frame."""
    import mujoco

    out = []
    for name in BENCH_MOTORS:
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise KeyError(f"benchmark punches: joint {name!r} not in the model")
        b = int(m.jnt_bodyid[j])
        out.append(
            (
                m.body(b).name,
                m.body(int(m.body_parentid[b])).name,
                tuple(float(x) for x in m.jnt_pos[j]),
            )
        )
    return out


def bench_punches(
    seed: int, links: list[tuple[str, str, tuple[float, float, float]]]
) -> list[Push]:
    """The benchmark's punches for run id ``seed`` (main.cpp schedule_make, applied as
    mjwarp_capture.py k_punch_apply does): one per waypoint slot, on a random motor's joint.
    The force pushes the joint's child link at the joint's anchor, so it also twists the link
    about its centre of mass; its direction, uniform on the sphere, is fixed in the child
    link's frame or in its parent's, at random, and turns with that link while the punch
    lasts. ``links`` are ``bench_motor_links``: (child, parent, anchor) per motor."""
    if len(links) != len(BENCH_MOTORS):
        raise ValueError(
            f"benchmark punches: {len(links)} links, the benchmark has {len(BENCH_MOTORS)}"
        )
    rng = BenchRandom(seed, BENCH_STREAM_PUNCH)
    out = []
    for i in range(BENCH_WAYPOINTS):
        t = BENCH_PUNCH_DELAY_S + BENCH_POINT_S * i
        joint = rng.below(len(BENCH_MOTORS))
        frame_child = rng.below(2) != 0
        z = rng.between(-1.0, 1.0)
        phi = rng.between(0.0, 2.0 * math.pi)
        r = math.sqrt(max(0.0, 1.0 - z * z))
        d = (r * math.cos(phi), r * math.sin(phi), z)
        climbed = min((t - BENCH_PUNCH_DELAY_S) / BENCH_RAMP_S, 1.0)
        ceiling = BENCH_FORCE_MAX_N * (BENCH_RAMP_FLOOR + (1 - BENCH_RAMP_FLOOR) * climbed)
        f = ceiling * rng.between(BENCH_FORCE_SCALE_MIN, 1.0)
        child, parent, anchor = links[joint]
        out.append(
            Push(
                t,
                "force",
                tuple(x * f for x in d),
                child,
                BENCH_PUNCH_S,
                tuple(float(x) for x in anchor),
                child if frame_child else parent,
            )
        )
    return out


def _stl_box(path: Path) -> np.ndarray:
    """The 8 corners of a binary STL's bounding box, as the benchmark reads them (main.cpp
    mesh_corners: the file's own coordinates, corner c takes hi on axis k when bit k is set)."""
    raw = path.read_bytes()
    tris = int.from_bytes(raw[80:84], "little")
    if len(raw) != 84 + 50 * tris:
        raise ValueError(f"{path}: not a binary STL")
    rec = np.frombuffer(raw, dtype=np.uint8, offset=84).reshape(tris, 50)[:, 12:48]
    v = np.ascontiguousarray(rec).view("<f4").reshape(-1, 3).astype(float)
    lo, hi = v.min(axis=0), v.max(axis=0)
    return np.array([[hi[k] if c >> k & 1 else lo[k] for k in range(3)] for c in range(8)])


class BenchHands:
    """The benchmark's check on a drawn arm pose (main.cpp hand_box, arm_bounds_make). With
    the left arm at the pose and the right arm mirroring it, each hand's box (its rubber hand
    mesh's bounding box, in the wrist yaw link's frame, placed in the torso's frame) may not
    reach further back than the hands do at STANCE, nor above the bottom of the head mesh."""

    def __init__(self, m: Any, torso: int, hands: tuple[int, int], corners: list, head_m: float):
        import mujoco

        self.m, self.d = m, mujoco.MjData(m)
        self.d.qpos[:] = m.qpos0
        self.torso, self.hands, self.corners, self.head_m = torso, hands, corners, head_m
        jq = {n: int(m.joint(n).qposadr[0]) for n in BENCH_ARM_LEFT + BENCH_ARM_RIGHT}
        self._left = np.array([jq[n] for n in BENCH_ARM_LEFT])
        self._right = np.array([jq[n] for n in BENCH_ARM_RIGHT])
        self._mirror = np.array(BENCH_ARM_MIRROR)
        lo, _ = self.boxes(np.array(BENCH_ARM_STANCE_LEFT))
        self.front_m = float(min(lo[0][0], lo[1][0]))

    @classmethod
    def from_mjcf(cls, m: Any, mjcf: str | Path) -> BenchHands | None:
        """None when the model lacks what the check needs (the torso, the wrist yaw links,
        and the rubber hand and head meshes, as files)."""
        import mujoco

        try:
            spec = mujoco.MjSpec.from_file(str(mjcf))
            root = Path(mjcf).resolve().parent
            meshdir = root / (spec.meshdir or "")
            files = {me.name: meshdir / me.file for me in spec.meshes}
            body = {
                n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
                for n in ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
            }
            if min(body.values()) < 0:
                return None
            corners = [_stl_box(files[f"{s}_rubber_hand"]) for s in ("left", "right")]
            head = _stl_box(files["head_link"])
            geoms = [g for g in spec.body("torso_link").geoms if g.meshname == "head_link"]
            dz = float(geoms[0].pos[2]) if geoms else 0.0
        except (KeyError, OSError, ValueError, AttributeError):
            return None
        return cls(
            m,
            body["torso_link"],
            (body["left_wrist_yaw_link"], body["right_wrist_yaw_link"]),
            corners,
            float(np.min(head[:, 2] + dz)),
        )

    def boxes(self, left: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        import mujoco

        self.d.qpos[self._left] = left
        self.d.qpos[self._right] = self._mirror * left
        mujoco.mj_kinematics(self.m, self.d)
        rt, pt = self.d.xmat[self.torso].reshape(3, 3), self.d.xpos[self.torso]
        lo, hi = [], []
        for h, c in zip(self.hands, self.corners):
            r = rt.T @ self.d.xmat[h].reshape(3, 3)
            v = rt.T @ (self.d.xpos[h] - pt) + c @ r.T
            lo.append(v.min(axis=0))
            hi.append(v.max(axis=0))
        return lo, hi

    def __call__(self, left: np.ndarray) -> bool:
        lo, hi = self.boxes(left)
        return all(lo_[0] >= self.front_m and hi_[2] <= self.head_m for lo_, hi_ in zip(lo, hi))


def bench_arm_walk(
    seed: int,
    seconds: float,
    limits: dict[str, tuple[float, float]],
    hands: BenchHands | None = None,
) -> dict[str, list[list[float]]]:
    """The benchmark's arm targets for run id ``seed``, as {joint: [[t, value], ...]} at each
    control step. Each 0.5 s window draws a new left arm target 25 steps of up to 0.06 rad per
    joint away, inside the joint limits, and moves to it over the window; the right arm
    mirrors the left. A target ``hands`` rejects is drawn again, up to 8 times, then the arm
    stays. Without ``hands`` every first draw is taken, and the draws part from the
    benchmark's at its first rejection."""
    rng = BenchRandom(seed, BENCH_STREAM_ARM)
    ticks = int(BENCH_ARM_WINDOW_S / BENCH_PERIOD_S + 0.5)
    lo = np.array([limits[n][0] for n in BENCH_ARM_LEFT])
    hi = np.array([limits[n][1] for n in BENCH_ARM_LEFT])
    left = np.array(BENCH_ARM_STANCE_LEFT, float)
    start, to = left.copy(), left.copy()
    window = -1
    steps = int(round(seconds / BENCH_PERIOD_S))
    times, poses = [], []
    for k in range(steps):
        # main.cpp's control loop: the walk starts on the second control step after release
        elapsed = float(_BENCH_S0 + _BENCH_PER_CONTROL * k) * _BENCH_DT - BENCH_INIT_S
        if k > 0:
            w = int(elapsed / BENCH_ARM_WINDOW_S)
            if w != window:
                window = w
                start, to = left.copy(), left.copy()
                for _ in range(BENCH_ARM_DRAWS):
                    u = rng.units(ticks * len(left)).reshape(ticks, len(left))
                    cand = start.copy()
                    for row in -1.0 + 2.0 * u:
                        cand = np.minimum(np.maximum(cand + BENCH_ARM_STEP_RAD * row, lo), hi)
                    if hands is None or hands(cand):
                        to = cand
                        break
            phase = min(
                max(
                    (elapsed - window * BENCH_ARM_WINDOW_S + BENCH_PERIOD_S) / BENCH_ARM_WINDOW_S,
                    0.0,
                ),
                1.0,
            )
            left = start + (to - start) * phase
        times.append(k * BENCH_PERIOD_S)
        poses.append(left.copy())
    out: dict[str, list[list[float]]] = {}
    for i, n in enumerate(BENCH_ARM_LEFT):
        out[n] = [[tt, float(p[i])] for tt, p in zip(times, poses)]
    for i, n in enumerate(BENCH_ARM_RIGHT):
        out[n] = [[tt, float(BENCH_ARM_MIRROR[i] * p[i])] for tt, p in zip(times, poses)]
    return out


def bench_arms_external(
    seed: int,
    seconds: float,
    limits: dict[str, tuple[float, float]],
    kp: dict[str, float],
    kd: dict[str, float],
    obs: str = "real",
    hands: BenchHands | None = None,
) -> External:
    walk = bench_arm_walk(seed, seconds, limits, hands)
    return External(list(walk), drive="trajectory", pose=walk, kp=kp, kd=kd, obs=obs)


# -- running a tour ------------------------------------------------------------------------

BENCH_ARMS = BENCH_ARM_LEFT + BENCH_ARM_RIGHT
_WN = 10 * 2.0 * math.pi  # the benchmark's holding servo: 10 Hz, damping ratio 2
_ZETA = 2.0


def armature_gains(m: Any, joints: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    """The benchmark's gains for joints it holds: kp = armature * wn^2, kd = 2 zeta armature wn."""
    import mujoco

    kp, kd = {}, {}
    for j in joints:
        arm = float(
            m.dof_armature[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]]
        )
        kp[j], kd[j] = arm * _WN * _WN, 2 * _ZETA * arm * _WN
    return kp, kd


@dataclass
class TourOptions:
    waypoints: list[tuple[float, float, float]] | None = None  # None: the benchmark's draws
    point_s: float = BENCH_POINT_S
    arms: str = "policy"  # "policy", "hold" (at STANCE) or "walk" (the benchmark's random walk)
    arms_obs: str = (
        "real"  # how the policy sees arms it does not drive; "contract": as its externals say
    )
    hold_gains: str = "armature"  # "armature" (the benchmark's) or "policy" (the contract's)
    punches: str = "none"  # "none" or "benchmark"
    backend: str | None = None
    # "crane": as the benchmark's harness starts a run, 3 s held at STANCE with the base
    # pinned at 0.793 m and its own gains, then let go; "default": at the policy's default
    # pose, the policy in control from the first step
    start: str = "crane"
    # "benchmark": a run ends when the pelvis is below 0.2 m, as the benchmark's does;
    # "tilt": gaitkeeper's usual (low, or tilted past 1 rad)
    fall: str = "benchmark"


def _tour_job(job: tuple) -> dict[str, Any]:
    import mujoco

    from .contract import Contract
    from .policy import load_policy
    from .runner import RunConfig, Runner

    contract_d, mjcf, policy_path, seed, o = job
    c = Contract.from_dict(contract_d)
    r = Runner(c, mjcf, load_policy(c, policy_path))
    m = mujoco.MjModel.from_xml_path(str(mjcf))
    lim = c.get("policy_io.commands.base_velocity.limit", None) or {}
    wps = o.waypoints if o.waypoints is not None else bench_waypoints(seed)
    tour = WaypointTour(
        list(wps),
        point_s=o.point_s,
        vx=tuple(lim.get("vx", (-0.5, 1.0))),
        vy_abs=max(abs(x) for x in lim.get("vy", (-0.3, 0.3))),
        wz_abs=max(abs(x) for x in lim.get("wz", (-0.2, 0.2))),
    )
    seconds = len(wps) * o.point_s
    ext = None
    un_traj = None
    if o.arms != "policy":
        arms = [j for j in r.names if j in BENCH_ARMS]
        # The contract's own externals (a port's held waist, say) stay; the arms are replaced.
        base = [e for e in r.externals(RunConfig())]
        contract_obs = {j: e.obs for e in base for j in e.joints}
        ext = []
        for e in base:
            rest = [j for j in e.joints if j not in BENCH_ARMS]
            if rest:
                ext.append(External(rest, e.drive, e.pose, e.kp, e.kd, e.obs))
        if o.hold_gains == "armature":
            kp, kd = armature_gains(m, arms)
        else:
            kp = {j: float(r.kp[r.names.index(j)]) for j in arms}
            kd = {j: float(r.kd[r.names.index(j)]) for j in arms}
        stance = dict(zip(BENCH_ARM_LEFT, BENCH_ARM_STANCE_LEFT))
        stance.update(
            {
                rj: BENCH_ARM_MIRROR[i] * BENCH_ARM_STANCE_LEFT[i]
                for i, rj in enumerate(BENCH_ARM_RIGHT)
            }
        )
        groups: dict[str, list[str]] = {}
        for j in arms:
            obs = contract_obs.get(j, "real") if o.arms_obs == "contract" else o.arms_obs
            groups.setdefault(obs, []).append(j)
        walk = None
        if o.arms == "walk":
            limits = {
                j: tuple(m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)])
                for j in BENCH_ARMS
            }
            walk = bench_arm_walk(seed, seconds, limits, BenchHands.from_mjcf(m, mjcf))
            # Arms the policy does not list (a legs-only policy) are the harness's already:
            # they follow the walk through control.unlisted.
            unlisted = c.get("control.unlisted", None)
            if unlisted:
                un_traj = {j: walk[j] for j in BENCH_ARMS if j not in r.names}
        for obs, js in groups.items():
            if walk is not None:
                ext.append(External(js, "trajectory", {j: walk[j] for j in js}, kp, kd, obs))
            else:
                ext.append(External(js, "hold", {j: stance[j] for j in js}, kp, kd, obs))
    pushes = []
    if o.punches == "benchmark":
        pushes = bench_punches(seed, bench_motor_links(m))
    start = None
    if o.start == "crane":
        kp, kd = armature_gains(m, list(BENCH_MOTORS))
        start = Start(
            dict(zip(BENCH_MOTORS, BENCH_STANCE)), kp, kd, BENCH_CRANE_HEIGHT_M, BENCH_INIT_S
        )
    elif o.start != "default":
        raise ValueError(f"tour start: {o.start!r}; known: crane, default")
    if o.fall not in ("benchmark", "tilt"):
        raise ValueError(f"tour fall: {o.fall!r}; known: benchmark, tilt")
    res = r.run(
        RunConfig(
            backend=o.backend,
            seconds=seconds,
            command_source=tour,
            external=ext,
            pushes=pushes,
            seed=seed,
            unlisted_trajectory=un_traj,
            start=start,
            fall_z=BENCH_FALL_PELVIS_Z if o.fall == "benchmark" else None,
        )
    )
    return {"seed": seed, **(res.task or {})}


def run_tour(
    contract: Any,
    mjcf: str,
    policy_path: str,
    seeds: list[int],
    options: TourOptions,
    workers: int | None = None,
) -> dict[str, Any]:
    """Run the tour once per seed. Returns per-seed reports and their summary."""
    from concurrent.futures import ProcessPoolExecutor

    from .runner import pool_context

    jobs = [(contract.to_dict(), str(mjcf), str(policy_path), s, options) for s in seeds]
    if workers and workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(workers, mp_context=pool_context()) as ex:
            runs = list(ex.map(_tour_job, jobs))
    else:
        runs = [_tour_job(j) for j in jobs]
    n_wp = len(options.waypoints) if options.waypoints is not None else BENCH_WAYPOINTS
    full = n_wp * options.point_s
    surv = [r["survival_s"] if r.get("survival_s") is not None else full for r in runs]
    pos = [r["pos_err_cm"] for r in runs if not math.isnan(r.get("pos_err_cm", math.nan))]
    yaw = [r["yaw_err_deg"] for r in runs if not math.isnan(r.get("yaw_err_deg", math.nan))]
    return {
        "runs": runs,
        "seconds": full,
        "mean_survival_s": float(np.mean(surv)),
        "complete": sum(r.get("outcome") == "complete" for r in runs),
        "pos_err_cm": float(np.mean(pos)) if pos else math.nan,
        "yaw_err_deg": float(np.mean(yaw)) if yaw else math.nan,
        "options": {k: v for k, v in options.__dict__.items() if k != "waypoints"},
    }
