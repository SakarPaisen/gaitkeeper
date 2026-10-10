import math

import numpy as np
import pytest

from gaitkeeper.tour import (
    BENCH_ARM_LEFT,
    BENCH_ARM_RIGHT,
    BENCH_ARM_STANCE_LEFT,
    BENCH_CENTRE,
    BENCH_MOTORS,
    BENCH_RADIUS_M,
    BenchHands,
    BenchRandom,
    WaypointTour,
    bench_arm_walk,
    bench_punches,
    bench_waypoints,
)


def _base(x, y, yaw):
    return np.array([x, y, 0.8, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])


def test_tour_commands_toward_the_waypoint_in_the_body_frame_and_clamps():
    t = WaypointTour([(1.0, 0.0, 0.0)], vx=(-0.5, 1.0), vy_abs=0.3, wz_abs=0.2)
    cmd = t(0.0, _base(0, 0, 0))
    np.testing.assert_allclose(cmd, [1.0, 0.0, 0.0])  # 1.5 * 1 m, clamped to vx max
    # Turned 90 degrees left: the same target is now to the robot's right.
    t2 = WaypointTour([(0.0, -0.1, math.pi / 2)], vy_abs=0.3, wz_abs=0.2)
    cmd = t2(0.0, _base(0, 0, 0))
    assert cmd[1] == pytest.approx(-0.15) and cmd[2] == pytest.approx(0.2)


def test_tour_reached_has_hysteresis_and_report_scores_slot_ends():
    t = WaypointTour([(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)], point_s=1.0)
    assert np.allclose(t(0.0, _base(0, 0, 0)), 0)  # at the target: reached, no command
    assert np.allclose(t(0.5, _base(0.15, 0, 0)), 0)  # inside the exit radius: still reached
    assert t(0.6, _base(0.25, 0, 0))[0] < 0  # left it: drives back
    t(1.0, _base(0.45, 0, 0))  # slot 2 starts
    rep = t.report(None)
    assert rep["outcome"] == "complete" and rep["targets"] == 2
    assert rep["pos_err_cm"] == pytest.approx((25 + 5) / 2)
    assert t.report(1.5)["outcome"] == "fell" and t.report(1.5)["targets"] == 1


def _links():
    return [(f"link{i}", f"parent{i}", (0.0, 0.0, 0.01 * i)) for i in range(len(BENCH_MOTORS))]


def test_benchmark_draws_stay_in_their_ranges():
    for x, y, yaw in bench_waypoints(3):
        assert math.hypot(x - BENCH_CENTRE[0], y - BENCH_CENTRE[1]) <= BENCH_RADIUS_M + 1e-9
        assert -math.pi <= yaw <= math.pi
    p = bench_punches(3, _links())
    forces = [np.linalg.norm(q.vector) for q in p]
    assert len(p) == 18 and p[0].t == pytest.approx(0.1) and p[1].t == pytest.approx(5.1)
    assert 500 / 6 - 1e-6 <= forces[0] <= 500 / 3 + 1e-6  # floor 1/3 of 500, scale 0.5..1
    assert max(forces) <= 500 + 1e-6
    lims = {n: (-1.0, 1.0) for n in BENCH_ARM_LEFT}
    walk = bench_arm_walk(3, 5.0, lims)
    for left, right, sign in zip(BENCH_ARM_LEFT, BENCH_ARM_RIGHT, (1, -1, -1, 1, -1, 1, -1)):
        lv = np.array(walk[left])[:, 1]
        assert np.all(np.abs(lv) <= 1.0)
        np.testing.assert_allclose(np.array(walk[right])[:, 1], sign * lv)


# Printed by the benchmark's own C++ (main.cpp random_for, tour_make, schedule_make and its
# arm walk, compiled from its source): seed n here must draw what run id n draws there.


def test_bench_random_is_the_benchmarks_stream():
    rng = BenchRandom(0, 2)
    assert [rng.raw() for _ in range(4)] == [3825457058, 706492411, 2134599173, 1489500560]


def test_bench_tour_is_the_benchmarks_run():
    assert bench_waypoints(0)[0] == (0.66224871621850956, 0.60809981725150375, 3.122743714855837)
    assert bench_waypoints(7)[0] == (0.35520893719601315, -0.49940458455993647, 2.1794076984728643)


def test_bench_punches_are_the_benchmarks_run():
    links = _links()
    # run 0: motor 23 (right shoulder roll), fixed in the child's frame; then motor 18, the
    # parent's frame. Run 7: motor 2, then 8.
    for run, want in (
        (
            0,
            [
                (
                    23,
                    True,
                    (0.29424279654533969, -0.5422523712258781, -0.78700923919677734),
                    163.27714920043945,
                ),
                (
                    18,
                    False,
                    (-0.85249265709124655, -0.20278835080669047, -0.48180198669433594),
                    129.42918803956775,
                ),
            ],
        ),
        (
            7,
            [
                (
                    2,
                    True,
                    (-0.076878020301946376, -0.86373292632548959, -0.49805140495300293),
                    163.92889122168222,
                ),
                (
                    8,
                    False,
                    (-0.81650488892334294, 0.53521156585691021, 0.2164909839630127),
                    183.18902949492139,
                ),
            ],
        ),
    ):
        got = bench_punches(run, links)
        for p, (j, child, d, f) in zip(got, want):
            assert p.body == f"link{j}" and p.point == links[j][2]
            assert p.frame == (f"link{j}" if child else f"parent{j}")
            np.testing.assert_allclose(p.vector, np.array(d) * f, rtol=0, atol=1e-11)
            assert p.duration == pytest.approx(0.08)
    with pytest.raises(ValueError):
        bench_punches(0, links[:5])


def test_bench_arm_walk_is_the_benchmarks_run():
    """Run 1's first draw is not rejected until window 6, so without the hand check the walk is
    the benchmark's up to there: stance on the release step, then moving each step."""
    lims = {
        "left_shoulder_pitch_joint": (-3.0892, 2.6704),
        "left_shoulder_roll_joint": (-1.5882, 2.2515),
        "left_shoulder_yaw_joint": (-2.618, 2.618),
        "left_elbow_joint": (-1.0472, 2.0944),
        "left_wrist_roll_joint": (-1.972222054, 1.972222054),
        "left_wrist_pitch_joint": (-1.614429558, 1.614429558),
        "left_wrist_yaw_joint": (-1.614429558, 1.614429558),
    }
    walk = bench_arm_walk(1, 3.0, lims)
    left = np.array([[v for _, v in walk[n]] for n in BENCH_ARM_LEFT]).T
    assert left.shape == (150, 7)
    np.testing.assert_array_equal(left[0], BENCH_ARM_STANCE_LEFT)
    want = {
        1: [
            0.17680742911372899,
            0.21164282930358638,
            -0.013165388779666005,
            0.59930335042994509,
            0.016728879456146949,
            -0.0018503663614486575,
            0.012764806936972638,
        ],
        24: [
            -0.089906096458435003,
            0.34553484439849863,
            -0.16456676959991456,
            0.59129191160202033,
            0.20911024332046507,
            -0.023129496574401852,
            0.15955951452255249,
        ],
        149: [
            0.011332309246063349,
            -0.053177785873413014,
            0.0063109874725341825,
            0.69530797004699696,
            -0.20600523233413709,
            0.055466237068176091,
            -0.99386242389678936,
        ],
    }
    for k, v in want.items():
        np.testing.assert_allclose(left[k], v, rtol=0, atol=1e-9)
    assert walk[BENCH_ARM_LEFT[0]][1][0] == pytest.approx(0.02)


def test_bench_arm_walk_draws_again_when_the_hands_check_rejects():
    lims = {n: (-3.0, 3.0) for n in BENCH_ARM_LEFT}
    seen = []

    def no(left):
        seen.append(left.copy())
        return False

    walk = bench_arm_walk(4, 1.0, lims, no)
    # every draw rejected: 8 draws a window, and the arm stays where it is
    assert len(seen) == 16
    for n, v in zip(BENCH_ARM_LEFT, BENCH_ARM_STANCE_LEFT):
        assert {x for _, x in walk[n]} == {v}
    calls = []

    def second(left):
        calls.append(left.copy())
        return len(calls) % 8 == 2

    walk = bench_arm_walk(4, 0.5, lims, second)
    end = np.array([walk[n][-1][1] for n in BENCH_ARM_LEFT])
    np.testing.assert_allclose(end, calls[1])  # the second draw, reached at the window's end
    assert not np.allclose(calls[0], calls[1])


def _stl(path, lo, hi):
    """A binary STL of the box lo..hi."""
    import struct

    c = [
        (hi[0] if i & 1 else lo[0], hi[1] if i & 2 else lo[1], hi[2] if i & 4 else lo[2])
        for i in range(8)
    ]
    faces = [(0, 4, 6), (0, 6, 2), (1, 3, 7), (1, 7, 5), (0, 1, 5), (0, 5, 4)]
    faces += [(2, 6, 7), (2, 7, 3), (0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6)]
    out = bytearray(80) + struct.pack("<I", len(faces))
    for t in faces:
        out += struct.pack("<3f", 0, 0, 0)
        for v in t:
            out += struct.pack("<3f", *c[v])
        out += b"\0\0"
    path.write_bytes(bytes(out))


def test_bench_hands_keeps_the_hands_behind_the_front_and_under_the_head(tmp_path):
    import mujoco

    (tmp_path / "m").mkdir()
    _stl(tmp_path / "m" / "hand.stl", (0.0, -0.02, -0.04), (0.1, 0.02, 0.0))
    _stl(tmp_path / "m" / "head.stl", (-0.05, -0.05, 0.0), (0.05, 0.05, 0.1))
    axes = ["0 1 0", "1 0 0", "0 0 1", "0 1 0", "1 0 0", "0 1 0", "0 0 1"]

    def arm(side, y):
        # a 0.3 m upper arm from the shoulder at (0, y, 0.2); the hand at the elbow
        pos = [f"0 {y} 0.2", "0 0 0", "0 0 0", "0 0 -0.3", "0 0 0", "0 0 0", "0 0 0"]
        names = [n.replace("left_", f"{side}_") for n in BENCH_ARM_LEFT]
        chain = "".join(
            f'<body name="{n.replace("_joint", "_link")}" pos="{pos[i]}">'
            f'<joint name="{n}" axis="{axes[i]}" range="-3 3"/><geom size="0.01" mass="0.1"/>'
            for i, n in enumerate(names)
        )
        hand = f'<geom type="mesh" mesh="{side}_rubber_hand" density="0"/>'
        return chain + hand + "</body>" * len(names)

    xml = f"""<mujoco><compiler meshdir="m"/>
      <asset><mesh name="left_rubber_hand" file="hand.stl"/><mesh name="right_rubber_hand" file="hand.stl"/>
        <mesh name="head_link" file="head.stl"/></asset>
      <worldbody><body name="pelvis"><freejoint/><geom size="0.05"/>
        <body name="torso_link"><geom type="mesh" mesh="head_link" pos="0 0 0.35" density="0"/>
          <geom size="0.05" mass="5"/>{arm("left", 0.15)}{arm("right", -0.15)}</body></body></worldbody></mujoco>"""
    f = tmp_path / "robot.xml"
    f.write_text(xml)
    m = mujoco.MjModel.from_xml_path(str(f))
    h = BenchHands.from_mjcf(m, f)
    assert h is not None and h.head_m == pytest.approx(0.35)
    stance = np.array(BENCH_ARM_STANCE_LEFT)
    assert h(stance)  # the pose the bounds are taken at passes
    up = stance.copy()
    up[0] = -3.0  # shoulder pitched up: the hand above the head's bottom
    assert not h(up)
    back = stance.copy()
    back[0] = 0.8  # shoulder pitched back: the hand behind where it is at stance
    assert not h(back)
    # a model without the meshes has no check
    plain = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><freejoint/><geom size='.1'/></body></worldbody></mujoco>"
    )
    assert BenchHands.from_mjcf(plain, f) is None


def test_runner_takes_a_closed_loop_command_and_reports_the_tour():
    from assets import UMJ_G1, URL_G1, need

    need(UMJ_G1, URL_G1 / "deploy.yaml")
    from gaitkeeper.policy import OnnxPolicy
    from gaitkeeper.readers.unitree_deploy import read_unitree_deploy
    from gaitkeeper.runner import RunConfig, Runner

    c, _ = read_unitree_deploy(URL_G1 / "deploy.yaml", URL_G1 / "policy.onnx")
    r = Runner(c, UMJ_G1, OnnxPolicy(URL_G1 / "policy.onnx"))
    tour = WaypointTour([(1.0, 0.0, 0.0)], point_s=3.0, vx=(-0.5, 1.0))
    res = r.run(RunConfig(seconds=3.0, command_source=tour))
    assert res.survived and res.dx > 0.5
    assert res.task["outcome"] == "complete" and res.task["targets"] == 1


def test_tour_command_runs_waypoints_from_a_file_and_takes_the_arms(tmp_path, capsys):
    from assets import UMJ_G1, URL_G1, need

    need(UMJ_G1, URL_G1 / "deploy.yaml")
    from gaitkeeper.cli import main

    wp = tmp_path / "wp.yaml"
    wp.write_text("waypoints:\n  - [0.3, 0.0, 0.0]\n  - [0.3, 0.2, 0.5]\n")
    out = tmp_path / "t.json"
    argv = ["tour", "--deploy", str(URL_G1 / "deploy.yaml"), "--onnx", str(URL_G1 / "policy.onnx")]
    argv += ["--mjcf", str(UMJ_G1), "--waypoints", str(wp), "--point-s", "2", "--seeds", "1"]
    argv += ["--arms", "walk", "--json", str(out)]
    code = main(argv)
    text = capsys.readouterr().out
    assert "arms walk (gains armature" in text and "Mean survival" in text
    import json

    d = json.loads(out.read_text())
    assert d["command"] == "tour" and len(d["runs"]) == 1 and d["seconds"] == 4.0
    assert code in (0, 5)
