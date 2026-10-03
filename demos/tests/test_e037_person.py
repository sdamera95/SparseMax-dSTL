"""Tests of E037's person-scene extension (experiments/e037_person.py). CPU only.

- VISIT at 0.45 m is E039's scene visit_h0.45_L10.0; the hand point on the visit's plateau lies `place` metres from the
  zone centre in the table plane, on the person's side, 0.10 m above the table.
- choose: the rule of RULE-hand-place.txt on hand-built rows.
- setup: per-run sphere centres equal the visit at the instance's place and the run's wait; per-run targets are the
  start's.
"""
import numpy as np

from experiments import e034_until_demo as U
from experiments import e037_person as Q
from experiments import e039_person_zone as P39

E039_SCENE = {"name": "visit_h0.45_L10.0", "kind": "visit", "phi": 0.0, "stand": 0.9, "depth": -0.45, "reach_z": 0.1, "t_move": 0.2, "lead": 0.1,
              "t_rest": 2.1, "length": 10.0, "far": 0.6, "hover": 0.6, "hover_z": -0.15}


def test_visit_is_e039_scene():
    assert Q.visit(0.45) == {k: v for k, v in E039_SCENE.items() if k != "name"}


def test_hand_place_on_plateau():
    for place in (0.40, 0.45, 0.60):  # three places
        for w in (2.0, 7.22):  # two waits
            P = Q.person(w, place)
            t = np.arange(len(P["human_centres"])) * U.HS
            v0, v1 = P["visit"]
            m = (t >= v0 + 1e-9) & (t <= v1 - 1e-9)
            h = P["human_centres"][m, P39.HAND]
            zone = np.asarray(U.scenario(w).zone)
            assert np.allclose(h[:, 0] - zone[0], place, atol=1e-9) and np.allclose(h[:, 1], 0.0, atol=1e-9)
            assert np.allclose(h[:, 2], 0.1, atol=1e-9)


def rows_for(inst, place, sep, slow, root, gap):
    return [{"instance": inst, "place": place, "depth": d, "wait": w, "sep_min": sep, "slow_min": slow, "root_plain_0.2": root, "hand_gap_plateau": gap,
             "deciding": "order", "root_exact": -d, "plain_separation": 0.5, "plain_slowdown": 0.7,
             **{"gap_" + p: 1.0 for p in P39.PHASES}, **{"hand_gap_" + p: 1.0 for p in P39.PHASES}} for d in U.DEPTHS for w in U.WAITS]


def test_choose_rule(tmp_path):
    rows = []
    places = sorted(set(Q.PLACES + (Q.E039_PLACE,)))
    for p in places:  # instance 0: passes at 0.45; instance 1: fails at 0.45, first good place 0.50; instance 2: nowhere
        rows += rows_for(0, p, 0.5, 0.9, 0.01, 0.29)
        rows += rows_for(1, p, 0.5 if p >= 0.48 else -0.1, 0.9, 0.01 if p >= 0.48 else -0.01, 0.29 if p <= 0.50 else 0.33)
        rows += rows_for(2, p, -0.1, 0.9, 0.01, 0.29)
    reg = tmp_path / "regime.csv"
    U._csv(str(reg), ["instance", "depth", "wait", "eps", "share_entry_deciding", "support_hold_deciding"],
           [[i, d, w, e, 0.15 if i == 0 else 0.3, 0] for i in range(3) for d in U.DEPTHS for w in U.WAITS for e in U.EPS])
    choice, pairs = Q.choose(rows, str(reg))
    assert [(c["place"], c["how"]) for c in choice] == [(0.45, "e039"), (0.48, "rule"), (0.45, "no place")]
    assert [p["pass"] for p in pairs] == [True, True, True, True, False, False]
    assert [p["until_inside_0.2"] for p in pairs] == [True, True, False, False, False, False]


def test_setup(tmp_path):
    rng = np.random.default_rng(41)
    Iv, Dv, Wv = (x.ravel() for x in np.meshgrid(np.arange(3), np.arange(2), np.arange(4), indexing="ij"))
    n = len(Iv)
    pick, hand = rng.standard_normal((3, 3))[Iv], rng.standard_normal((3, 3))[Iv]
    np.savez(tmp_path / "starts.npz", V0=rng.standard_normal((n, 500, 7)).astype(np.float32), x0=np.zeros(14), wait=np.asarray(U.WAITS)[Wv],
             depth=np.asarray(U.DEPTHS)[Dv], instance=Iv, pick=pick, handover=hand)
    U._csv(str(tmp_path / "choice.csv"), ["instance", "place"], [[0, 0.45], [1, 0.5], [2, 0.45]])
    out = str(tmp_path / "inst.npz")
    Q.setup_main(str(tmp_path), str(tmp_path / "choice.csv"), out, (1, 2), ("sparsemax", "lse"))
    z = np.load(out)
    assert len(z["run_method"]) == 4 * 2 * 2 * 2
    for r in (0, 5, 17, 31):  # sampled runs
        place = 0.5 if z["run_instance"][r] == 1 else 0.45
        assert np.array_equal(z["hc"][r], Q.person(float(z["run_wait"][r]), place)["human_centres"].astype(np.float32))
        assert np.array_equal(z["pick"][r], pick[z["plan"][r]]) and Iv[z["plan"][r]] == z["run_instance"][r]
    assert z["run_person"][0] == "visit_0.5"
