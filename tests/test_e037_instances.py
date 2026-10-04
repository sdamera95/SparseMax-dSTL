"""Tests of the instance generator and the run bookkeeping of examples.e037_instances, on the CPU."""
import numpy as np
import pytest

from examples import e034_until_demo as U
from examples import e037_instances as X
from sparsemax_dstl.tasks import workspace as W


# ------------------------------------------------------------------
# the generator

def test_draws_deterministic_and_in_range():
    a, b, c = X.draws(), X.draws(), X.draws(n=7)
    for k in a:
        assert np.array_equal(a[k], b[k])
        assert np.array_equal(a[k][:7], c[k])
    for k, (lo, hi) in X.RANGES.items():
        assert np.all((a[k] >= lo) & (a[k] <= hi)), k
    assert a["via_noise"].shape == (X.N_DRAWS, 7)
    assert not np.array_equal(X.draws(seed=X.SEED + 1)["az"], a["az"])


def test_targets_reproduce_e034():
    sc = U.scenario(U.WAITS[0])
    one = lambda v: np.array([v])  # noqa: E731
    d = {"az": one(np.pi), "el": one(U.ELEV), "hd_az": one(0.0), "hd_el": one(U.ELEV), "hd_r": one(U.HANDOVER_RADIUS), "via_out": one(U.VIA_OUT),
         "via_up": one(U.VIA_UP), "via_side": one(0.0)}
    n, pick, hand, via = X.targets(d, sc)
    g, g_hand = W.until_targets(sc, U.DIRECTION, U.HANDOVER_RADIUS)
    n34 = np.asarray(U.DIRECTION) / np.linalg.norm(U.DIRECTION)
    assert np.max(np.abs(n[0] - n34)) <= 1e-12
    assert np.max(np.abs(pick[0] - g)) <= 1e-12
    assert np.max(np.abs(hand[0] - g_hand)) <= 1e-12
    assert np.max(np.abs(via[0] - (g + U.VIA_OUT * n34 + np.array([0.0, 0.0, U.VIA_UP])))) <= 1e-12


def test_targets_geometry():
    sc = U.scenario(U.WAITS[0])
    d = X.draws()
    n, pick, hand, via = X.targets(d, sc)
    z = np.asarray(sc.zone)
    assert np.max(np.abs(np.linalg.norm(n, axis=1) - 1)) <= 1e-12
    assert np.max(np.abs(np.linalg.norm(pick - z, axis=1) - (sc.zone_radius + sc.until_hold))) <= 1e-12
    assert np.max(np.abs(np.linalg.norm(hand - z, axis=1) - d["hd_r"])) <= 1e-12
    assert np.all(d["hd_r"] < sc.zone_radius) and np.all(hand[:, 2] >= W.HANDOVER_MIN_Z)
    assert np.all(np.linalg.norm(via - z, axis=1) > sc.zone_radius + sc.until_hold)
    assert np.all(np.sum((via - pick) * n, 1) > 0)


# ------------------------------------------------------------------
# the regime check

def synthetic(B, w, rng):
    """Zone and pick margins and scores (B, T) with an entry of 8 to 18 samples on [1.0, 2.0] s to a
    depth of 0.03 to 0.12, a hold 0.025 outside the zone from 2.0 s to w + 1, and noise."""
    T = U.H * 50 + 1
    t = np.arange(T) * U.HS
    depth = rng.uniform(0.03, 0.12, B)
    k = rng.integers(8, 19, B)
    c = 1.5
    dip = np.clip(1 - ((t[None] - c) / (k[:, None] * U.HS / 2)) ** 2, 0, None)
    approach = np.clip(0.5 * (1.0 - t[None]), 0, None)
    zone_m = 0.025 + approach - (0.025 + depth[:, None]) * dip + 0.002 * rng.standard_normal((B, T)) * (t[None] > 2.0)
    zone_m = np.where(t[None] > w + 1, 0.025 - (t[None] - w - 1), zone_m)
    pick_m = np.where((t[None] >= 1.9) & (t[None] <= w + 1.01), 0.99, -1.0) + 0.001 * rng.standard_normal((B, T))
    zone_s = zone_m - 0.001 * rng.random((B, T))
    pick_s = pick_m - 0.001 * rng.random((B, T))
    entry = (zone_m < 0) & (t[None] <= U.T_HOLD + 1e-9)
    return zone_s, pick_s, zone_m, pick_m, entry


@pytest.mark.parametrize("w,eps", [(2.0, 0.4), (7.22, 0.4), (4.0, 0.2)])
def test_until_batch_equals_until_rows(w, eps):
    rng = np.random.default_rng(int(w * 100) + int(eps * 10))
    B = 5
    zs, ps, zm, pm, entry = synthetic(B, w, rng)
    got = X.until_batch(zs, ps, zm, pm, w, eps, entry, U.T_HOLD)
    for b in range(B):  # brute-force oracle: one start at a time through until_rows
        ref = U.until_rows(zs[b], ps[b], zm[b], pm[b], w, eps, entry[b], U.T_HOLD)
        assert set(ref) == set(got), b
        for key, v in ref.items():
            g = got[key][b]
            if isinstance(v, (bool, int, np.integer)):
                assert int(g) == int(v), (b, key, g, v)
            else:
                assert np.isclose(g, v, rtol=1e-12, atol=1e-12, equal_nan=True), (b, key, g, v)
    assert np.any(got["support_hold_deciding"] == 0)


# ------------------------------------------------------------------
# bookkeeping

def test_run_outcomes_against_e034():
    rng = np.random.default_rng(31)
    n, K = 40, 15
    rs = rng.standard_normal((n, K)) + np.linspace(-1, 1, K)
    ex = rng.standard_normal((n, K)) + np.linspace(-1, 1.5, K)
    ex[:3] = -1.0
    rs[3:6] = -1.0
    o = X.run_outcomes(rs, ex)
    r = U.outcomes(rs, ex, rng.standard_normal((n, K, 4)))
    assert np.array_equal(o["first_claim"], r["claim"])
    assert np.array_equal(o["first_safe"], r["safe"])
    assert np.array_equal(o["false_claim_end"], r["false_safe_end"])
    for i in range(n):  # brute-force oracle: runs
        cl = rs[i] >= 0
        assert o["false_claim_iterates"][i] == sum(cl[k] and ex[i, k] < 0 for k in range(K))
        if cl.any():
            assert o["deepest_while_claiming"][i] == min(ex[i, k] for k in range(K) if cl[k])
        else:
            assert np.isnan(o["deepest_while_claiming"][i])
        fs = o["first_safe"][i]
        assert o["stays_safe"][i] == (fs >= 0 and all(ex[i, k] >= 0 for k in range(fs, K)))


def test_paired_oracle():
    rng = np.random.default_rng(32)
    K = 101
    for trial in range(20):  # brute-force oracle: random cases
        n = int(rng.integers(1, 17))
        fa, fb = rng.integers(-1, 30, n), rng.integers(-1, 30, n)
        q = X.paired(fa, fb, K, np.random.default_rng(5))
        d = [(K if a < 0 else a) - (K if b < 0 else b) for a, b in zip(fa, fb)]  # brute-force oracle: instances
        assert (q["earlier"], q["equal"], q["later"]) == (sum(x < 0 for x in d), sum(x == 0 for x in d), sum(x > 0 for x in d))
        assert q["n"] == n and q["median"] == float(np.median(d))
        assert q["lo"] <= q["median"] <= q["hi"]
        assert min(d) <= q["lo"] and q["hi"] <= max(d)
    q1, q2 = X.paired(fa, fb, K, np.random.default_rng(5)), X.paired(fa, fb, K, np.random.default_rng(5))
    assert q1 == q2


def test_label_rows():
    rows = []
    for inst, share, hold in ((0, 0.15, 0), (1, 0.25, 0), (2, 0.1, 3)):
        for w in U.WAITS:
            for eps in U.EPS:
                rows.append({"instance": inst, "depth": 0.05, "draw": inst + 10, "eps": eps, "wait": w,
                             "share_entry_deciding": share if w == max(U.WAITS) else 0.6, "support_hold_deciding": hold if (w == 4.0 and eps == 0.4) else 0,
                             "support_hold_any": 0, "k_entry": 17, "gap": 0.075, "gamma_over_k": 0.047, "root_lse_plain": 0.02, "deciding": "order"})
    lab = X.label_rows(rows)
    assert [r["instance"] for r in lab] == [0, 1, 2]
    assert [r["inside"] for r in lab] == [True, False, False]
    assert [r["share_ok"] for r in lab] == [True, False, True]
    assert [r["support_ok"] for r in lab] == [True, True, False]


def test_setup_and_load(tmp_path):
    rng = np.random.default_rng(33)
    Iv, Dv, Wv = (x.ravel() for x in np.meshgrid(np.arange(3), np.arange(2), np.arange(4), indexing="ij"))
    n = len(Iv)
    pick = rng.standard_normal((3, 3))[Iv]
    hand = rng.standard_normal((3, 3))[Iv]
    V0 = rng.standard_normal((n, 500, 7)).astype(np.float32)
    np.savez(tmp_path / "starts.npz", V0=V0, x0=np.zeros(14), wait=np.asarray(U.WAITS)[Wv], depth=np.asarray(U.DEPTHS)[Dv], instance=Iv,
             pick=pick, handover=hand)
    out = str(tmp_path / "inst.npz")
    X.setup_main(str(tmp_path), out, 0.4, (2, 0), ("lse", "sparsemax"))
    z = np.load(out)
    assert len(z["run_method"]) == 4 * 2 * 2 * 2
    p = z["plan"]
    assert np.array_equal(z["V0"], V0[p])
    assert np.array_equal(z["pick"], pick[p]) and np.array_equal(z["handover"], hand[p])
    assert np.array_equal(Iv[p], z["run_instance"]) and np.allclose(np.asarray(U.WAITS)[Wv][p], z["run_wait"])
    assert np.allclose(np.asarray(U.DEPTHS)[Dv][p], z["run_depth"])
    # order: wait, arm, instance, depth (depth fastest)
    assert z["run_instance"][:4].tolist() == [2, 2, 0, 0] and z["run_depth"][:2].tolist() == [0.05, 0.1]
    assert np.all(z["run_person"] == "out")
    assert np.array_equal(z["hc"][-1], U.instance(z["run_wait"][-1], "out")["human_centres"].astype(np.float32))
    I = X.load(out)
    assert len(I["groups"]) == 8 and all(g[3] - g[2] == 4 for g in I["groups"])
    assert [(g[0], g[4]) for g in I["groups"][:2]] == [("lse", 2.0), ("sparsemax", 2.0)]
    J = X.load(out, "5,6")
    assert np.array_equal(J["pick"], z["pick"][[5, 6]])
