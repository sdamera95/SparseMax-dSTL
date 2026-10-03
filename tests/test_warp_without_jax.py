"""The Warp side in a process where JAX, MJX and the optional packages of the examples cannot be imported: imports,
the evaluator and the predicates, CPU, float64.

The reference values come from the JAX evaluator and the MJX predicates in the test process."""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from sparsemax_dstl.jax import methods, robustness
from sparsemax_dstl.stl import Atom, Until, compile_formula
from sparsemax_dstl.tasks import workspace as W
from sparsemax_dstl.tasks import workspace_mjx as Wm

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
WARP_MODULES = ["sparsemax_dstl.warp.evaluator", "sparsemax_dstl.warp.plant", "sparsemax_dstl.warp.predicates",
                "sparsemax_dstl.warp.solver", "sparsemax_dstl.warp.solver_conjuncts"]
# the modules the optimization of the manipulator loads before it builds the specification
SOLVER_MODULES = ["examples.e034_until_demo", "examples.e037_instances", "examples.e037_person", "examples.e038_horizon",
                  "examples.e040_conj", "examples.e042_instances", "sparsemax_dstl.plants", "sparsemax_dstl.stl",
                  "sparsemax_dstl.tasks.human", "sparsemax_dstl.tasks.panda", "sparsemax_dstl.tasks.workspace"] + WARP_MODULES
BETA, GAMMA, EPS = 10.0, 0.1, 0.2


def without_jax(code, *args):
    """Standard output of the code run from the repository root by a new interpreter in which importing any of ABSENT raises."""
    done = subprocess.run([sys.executable, "-c", NO_JAX + textwrap.dedent(code), *args], capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_package_and_warp_modules_import():
    """Importing the package loads neither Warp nor JAX, and every module of sparsemax_dstl.warp imports."""
    out = without_jax("""
        import importlib, pkgutil
        import sparsemax_dstl
        print("warp" in sys.modules)
        import sparsemax_dstl.warp
        names = sorted(m.name for m in pkgutil.walk_packages(sparsemax_dstl.warp.__path__, "sparsemax_dstl.warp."))
        for n in names:
            importlib.import_module(n)
        print(" ".join(names))
    """)
    assert out.split() == ["False"] + WARP_MODULES


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
    warp_param = {"exact": None, "lse_plain": BETA, "lse": BETA, "gm_pm01": 1.0, "gm_pm10": 1.0, "gm_exp": EPS, "sparsemax": GAMMA}
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


def test_solver_modules_import_and_the_scene_is_built():
    """The scene, the person and the four conjuncts of the specification, built without JAX. The scripts prune the compiled
    specification with functions of examples/e022_regime.py, which imports JAX, MJX and optax and is not loaded here."""
    out = without_jax("""
        import importlib, json
        for n in json.loads(sys.argv[1]):
            importlib.import_module(n)
        from examples import e034_until_demo as U
        from sparsemax_dstl import stl
        from sparsemax_dstl.tasks import panda, workspace as W
        sc = U.scenario(7.22)
        plant = W.Plant()
        n_r = len(W.robot_spheres(plant, sc.robot_spacing)["body"])
        inst = U.instance(7.22, "zone")
        names, rows, spec = W.specs(sc, n_r, len(inst["human_radii"]))
        steps = [len(stl.compile_formula(r, sc.samples).steps) for r in rows]
        print(json.dumps([n_r, len(inst["human_radii"]), list(names), steps, panda.torque_limit().tolist()]))
    """, json.dumps(SOLVER_MODULES))
    sc = W.Scenario()
    n_r, n_h, names, steps, torque = json.loads(out)
    assert n_r == len(W.robot_spheres(W.Plant(), sc.robot_spacing)["body"]) and n_h > 0
    assert names == ["order", "handover", "separation", "slowdown"] and len(steps) == 4 and min(steps) > 1
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
