"""The Warp backend in a process where JAX, MJX and the optional packages of the examples cannot be imported, against the JAX results of the test process."""
import json
import pkgutil
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

import sparsemax_dstl
from examples import e034_until_demo, e040_conj
from sparsemax_dstl.jax import methods, robustness
from sparsemax_dstl.stl import Atom, Until, compile_formula
from sparsemax_dstl.tasks import planar_disk
from sparsemax_dstl.tasks import workspace as W
from sparsemax_dstl.tasks import workspace_mjx as Wm
from sparsemax_dstl.tasks import workspace_program
from sparsemax_dstl.tasks.planar_al_jax import JaxChain

ROOT = Path(__file__).resolve().parents[1]
ABSENT = ("jax", "mujoco.mjx", "optax", "scipy", "matplotlib")
NO_JAX = """
import importlib, sys
for name in ABSENT:
    sys.modules[name] = None
for name in ABSENT:
    try:
        importlib.import_module(name)
    except ImportError:
        continue
    raise SystemExit(name + " imported")
""".replace("ABSENT", repr(ABSENT))
# the scripts whose Warp stages optimize the manipulator and take the torque gradient
SOLVER_SCRIPTS = ["examples.e034_until_demo", "examples.e037_instances", "examples.e037_person", "examples.e038_horizon",
                  "examples.e040_conj", "examples.e042_instances", "examples.e045_two_properties"]
BETA, GAMMA, EPS = 10.0, 0.1, 0.2


def without_jax(code, *args, absent=True):
    """Standard output of the code run from the repository root by a new interpreter in which importing any of ABSENT
    raises; with absent=False the interpreter is an ordinary one."""
    head = NO_JAX if absent else "import importlib, sys\n"
    done = subprocess.run([sys.executable, "-c", head + textwrap.dedent(code), *args], capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_only_the_jax_modules_need_jax():
    """Importing the package loads neither Warp nor JAX, and every module of the package imports except those of
    sparsemax_dstl.jax and the task modules named *_jax and *_mjx."""
    names = sorted(m.name for m in pkgutil.walk_packages(sparsemax_dstl.__path__, "sparsemax_dstl."))
    out = without_jax("""
        import json
        import sparsemax_dstl
        print("warp" in sys.modules)
        failed = []
        for n in json.loads(sys.argv[1]):
            try:
                importlib.import_module(n)
            except ImportError:
                failed.append(n)
        print(json.dumps(failed))
    """, json.dumps(names)).split("\n")
    assert out[0] == "False"
    assert json.loads(out[1]) == [n for n in names if n.startswith("sparsemax_dstl.jax") or n.endswith(("_jax", "_mjx"))]
    assert {"sparsemax_dstl.warp.evaluator", "sparsemax_dstl.warp.plant", "sparsemax_dstl.warp.predicates", "sparsemax_dstl.warp.solver",
            "sparsemax_dstl.warp.solver_conjuncts", "sparsemax_dstl.tasks.planar_warp", "sparsemax_dstl.tasks.workspace_program"} <= set(names)


def readme_example():
    """Stay out of the zone (predicate 0) until the pick (predicate 1), 41 samples: the program and the predicate values."""
    t = np.arange(41)
    scores = np.stack([np.where((t >= 10) & (t < 14), -0.05, 0.2), np.where(t >= 30, 0.1, -1.0)], -1)
    return compile_formula(Until((0, 40), Atom(0), Atom(1)), T=41), scores


def test_evaluator_gives_the_jax_values():
    """Robustness and gradient of the README's example under every measure. gm_exp takes the error per node in the Warp
    evaluator, as methods.SEMANTICS["gm_exp"] does in JAX."""
    program, scores = readme_example()
    jax_measure = {"exact": ("exact", None), "lse_plain": ("lse_plain", BETA), "lse": ("lse", BETA), "gm_pm01": ("gm_pm01", None),
                   "gm_pm10": ("gm_pm10", None), "gm_exp": (methods.SEMANTICS["gm_exp"], EPS), "sparsemax": ("sparsemax", GAMMA)}
    warp_param = {"exact": None, "lse_plain": BETA, "lse": BETA, "gm_pm01": None, "gm_pm10": None, "gm_exp": EPS, "sparsemax": GAMMA}
    out = json.loads(without_jax("""
        import json
        import numpy as np
        import warp as wp
        from sparsemax_dstl.stl import Atom, Until, compile_formula
        from sparsemax_dstl.warp import Evaluator

        wp.config.log_level = wp.LOG_WARNING
        program = compile_formula(Until((0, 40), Atom(0), Atom(1)), T=41)
        t = np.arange(41)
        scores = np.stack([np.where((t >= 10) & (t < 14), -0.05, 0.2), np.where(t >= 30, 0.1, -1.0)], -1)[None]
        x = wp.array(scores, dtype=wp.float64, device="cpu")
        out = {}
        for measure, param in json.loads(sys.argv[1]).items():
            rho, grad = Evaluator(program, measure, param, B=1, dtype=wp.float64, device="cpu").gradient(x)
            out[measure] = [rho.numpy()[0, 0].item(), grad.numpy()[0].tolist()]
        print(json.dumps(out))
    """, json.dumps(warp_param)))
    assert sorted(out) == sorted(jax_measure)
    with jax.enable_x64(True):
        z = jnp.asarray(scores, jnp.float64)
        for measure, (semantics, param) in jax_measure.items():
            value, grad = jax.value_and_grad(lambda s: robustness(program, s, semantics, param)[0])(z)
            assert abs(out[measure][0] - float(value)) <= 1e-12, measure
            assert np.abs(np.asarray(out[measure][1]) - np.asarray(grad)).max() <= 1e-12, measure


def sizes(programs):
    """Number of steps and of rows of each program."""
    return [[len(p.steps), sum(st.length for st in p.steps)] for p in programs]


def test_manipulator_scripts_import_and_build_the_specification():
    """The scripts of the manipulator's Warp stages import, and their scene, person and pruned programs (the whole
    specification and its four conjuncts) are those of the test process."""
    out = without_jax("""
        import json
        for n in json.loads(sys.argv[1]):
            importlib.import_module(n)
        from examples import e034_until_demo, e040_conj
        from sparsemax_dstl.tasks import panda, workspace_program
        sc = e034_until_demo.scenario(7.22)
        n_h = len(e034_until_demo.instance(7.22, "zone")["human_radii"])
        programs = (workspace_program.core_program(sc, n_h),) + e040_conj.conj_programs(sc, n_h)
        sizes = [[len(p.steps), sum(st.length for st in p.steps)] for p in programs]
        print(json.dumps([workspace_program.N_R, n_h, sizes, panda.torque_limit().tolist()]))
    """, json.dumps(SOLVER_SCRIPTS))
    n_r, n_h, got, torque = json.loads(out)
    sc = e034_until_demo.scenario(7.22)
    assert n_r == workspace_program.N_R and n_h == len(e034_until_demo.instance(7.22, "zone")["human_radii"])
    assert got == sizes((workspace_program.core_program(sc, n_h),) + e040_conj.conj_programs(sc, n_h))
    assert len(got) == 5 and min(rows for _, rows in got) > 1
    assert torque == [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]


def test_predicates_give_the_mjx_values(tmp_path):
    """The predicate values of two worlds of five samples from the Warp kernels, against workspace_mjx.scores."""
    plant, sc = W.Plant(), W.Scenario()
    rng = np.random.default_rng(5)
    m = plant.model
    X = np.concatenate([rng.uniform(m.jnt_range[:, 0], m.jnt_range[:, 1], (2, 5, 7)), rng.standard_normal((2, 5, 7))], -1)
    hc = rng.uniform([-0.2, -0.5, 0.0], [0.8, 0.5, 1.0], (2, 5, 5, 3))
    hr = rng.uniform(0.04, 0.1, 5)
    goals = rng.uniform([0.2, -0.5, 0.1], [0.7, 0.5, 0.6], (2, 2, 3))
    np.savez(tmp_path / "in.npz", X=X, hc=hc, hr=hr, goals=goals)
    without_jax("""
        import numpy as np
        import warp as wp
        from sparsemax_dstl.tasks import workspace as W
        from sparsemax_dstl.warp.predicates import Predicates

        wp.config.log_level = wp.LOG_WARNING
        d = np.load(sys.argv[1] + "/in.npz")
        pr = Predicates(W.Plant(), W.Scenario(), 5, nworld=2, dtype=wp.float64, device="cpu")
        pr.set_instance(d["goals"][0], d["goals"][1], d["hc"], d["hr"])
        q = wp.array(d["X"][..., :7].reshape(10, 7), dtype=wp.float64, device="cpu")
        v = wp.array(d["X"][..., 7:].reshape(10, 7), dtype=wp.float64, device="cpu")
        np.save(sys.argv[1] + "/Z.npy", pr.scores(q, v).numpy())
    """, str(tmp_path))
    Z = np.load(tmp_path / "Z.npy")
    with jax.enable_x64(True):
        mx = mjx.put_model(m, impl="jax")

        def ref(X, hc, pick, hand):
            return Wm.scores(mx, plant, sc, {"pick": pick, "handover": hand, "human_centres": hc, "human_radii": jnp.asarray(hr)}, X)

        Zr = np.asarray(jax.vmap(ref)(jnp.asarray(X), jnp.asarray(hc), jnp.asarray(goals[0]), jnp.asarray(goals[1])))
    assert Z.shape == Zr.shape
    assert np.abs(Z - Zr).max() <= 1e-12


def test_unicycle_warp_chain_and_solver(tmp_path):
    """The unicycle's Warp chain and ten updates of the solver without JAX: the conjuncts' values against the JAX chain
    within 1e-9, and every output equal, bit for bit, to the same run in an interpreter that has JAX."""
    T, eps, z0 = 120, 0.1, (1.5, 1.0, np.pi / 2)
    regions = planar_disk.make_regions((3.0, 4.5, 1.4), (5.3, 7.3, 1.0), (8.3, 2.5, 1.0), (8.8, 6.8, 0.5))
    s = np.arange(T - 1) / (T - 1)
    V0 = np.stack([np.full((2, T - 1), 0.9), 0.4 * np.sin(2 * np.pi * s + np.array([[0.0], [1.0]]))], -1)
    np.savez(tmp_path / "in.npz", V0=V0, regions=regions, z0=np.asarray(z0))
    code = """
        import numpy as np
        import warp as wp
        from sparsemax_dstl.tasks import planar, planar_al, planar_disk, planar_oracle, planar_oracle_sound
        from sparsemax_dstl.tasks.planar_warp import WarpChain

        wp.config.log_level = wp.LOG_WARNING
        d = np.load(sys.argv[1] + "/in.npz")
        conjuncts = planar_disk.specification(40, 60, 80, 95, 120)[1]
        chain = WarpChain(conjuncts, 120, "sparsemax", 0.1, d["z0"], d["regions"], 2)
        r = chain.forward(d["V0"])
        V, rs, Ls = planar_al.solve(chain, d["V0"], 10, 0.01, 0.0, 0.002)
        np.savez(sys.argv[1] + "/" + sys.argv[2], r=r, C=chain.C, V=V, rs=rs, Ls=Ls)
    """
    # both runs in new interpreters: in this process an earlier test may have built the MuJoCo Warp plant, which rebinds
    # wp.sqrt (the same derivative, rounded differently) for every kernel built afterwards
    without_jax(code, str(tmp_path), "absent.npz")
    without_jax(code, str(tmp_path), "present.npz", absent=False)
    out, ref = np.load(tmp_path / "absent.npz"), np.load(tmp_path / "present.npz")
    conjuncts = planar_disk.specification(40, 60, 80, 95, T)[1]
    with jax.enable_x64(True):
        r_jax = JaxChain(conjuncts, T, "sparsemax", eps, z0, regions, 2).forward(V0)
    assert out["r"].shape == (2, 4) and np.abs(out["r"] - r_jax).max() < 1e-9
    assert sorted(out.files) == ["C", "Ls", "V", "r", "rs"]
    for k in out.files:
        assert out[k].tobytes() == ref[k].tobytes(), k
    assert np.abs(out["V"] - V0).max() > 0 and np.abs(out["C"]).max() > 1e-3
