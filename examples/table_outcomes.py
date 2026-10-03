# The outcome table of the manipulator optimization (Table III of the paper), from the run records of
# examples.e042_instances run (four constraints, 100 updates).
#     python examples/table_outcomes.py <out_dir> <grid_dir>
# <grid_dir> holds t1_b0..t1_b3.npz (10 s), h20_b0, h20_b1.npz (20 s) and h40_a0..h40_a2.npz (40 s).
# Per run (one smoothing, one pick and handover location, one initial trajectory): the exact robustness of every
# iterate (float64 replay), the four smoothed constraint values of every iterate, the effort of every iterate.
import sys, json, csv
import numpy as np

out, G = sys.argv[1], sys.argv[2]
files = {10: ["t1_b0", "t1_b1", "t1_b2", "t1_b3"], 20: ["h20_b0", "h20_b1"], 40: ["h40_a0", "h40_a1", "h40_a2"]}
late = {10: 7.22, 20: 17.22, 40: 37.22}
names = {"lse_plain": "plain LSE", "lse": "sound LSE", "gm_pm01": "GMR (0,1)", "gm_pm10": "GMR (-10,10)", "sparsemax": "sparsemax"}

def first(mask):
    # first index along axis 1 at which mask holds; -1 where it never holds
    return np.where(mask.any(1), mask.argmax(1), -1)

def q(x):
    return np.percentile(x, [50, 25, 75]) if len(x) else np.full(3, np.nan)

rows, check = [], []
for H, fl in files.items():
    for f in fl:
        z = np.load(G + "/" + f + ".npz", allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        keys = meta["trace_keys"]
        exact = z["referee64"]                                   # (runs, 101)
        V = z["V"]                                               # (runs, 101, T, 7) normalized torques
        effort = np.mean(np.sum(V.astype(np.float64) ** 2, -1), -1)   # (runs, 101)
        check.append(float(np.max(np.abs(effort[:, :100] - z["trace"][:, :, keys.index("effort")]) / effort[:, :100])))
        r = z["conj_r"]                                          # (runs, 100, 4) smoothed values of iterates 0..99
        i_sat = first(exact >= 0)
        i_acc = first(r.min(-1) >= 0)
        acc_exact = np.where(i_acc >= 0, np.take_along_axis(exact, np.maximum(i_acc, 0)[:, None], 1)[:, 0], np.nan)
        sat_effort = np.where(i_sat >= 0, np.take_along_axis(effort, np.maximum(i_sat, 0)[:, None], 1)[:, 0], np.nan)
        acc_viol = np.sum((r.min(-1) >= 0) & (exact[:, :100] < 0), 1)   # iterations the solver accepts while the float64 replay violates
        worst_acc = np.where((r.min(-1) >= 0), exact[:, :100], np.inf).min(1)  # smallest exact robustness among accepted iterations
        after = np.where(np.arange(101)[None, :] >= np.maximum(i_sat, 0)[:, None], exact, np.inf).min(1)
        for j in range(exact.shape[0]):                          # table assembly, one row per run
            rows.append(dict(H=H, w=float(z["run_wait"][j]), method=str(z["run_method"][j]), depth=float(z["run_depth"][j]),
                             location=int(z["run_instance"][j]), eps=float(z["run_eps"][j]), rho0=exact[j, 0], rho100=exact[j, 100],
                             i_sat=int(i_sat[j]), i_acc=int(i_acc[j]), rho_at_acc=acc_exact[j], rho_min_after_sat=after[j] if i_sat[j] >= 0 else np.nan,
                             E0=effort[j, 0], E100=effort[j, 100], E_at_sat=sat_effort[j],
                             accept_while_violating=int(acc_viol[j]), smallest_rho_among_accepted=worst_acc[j]))
print("largest relative difference between the effort recomputed from the torques and the trace's effort:", max(check))
with open(out + "/runs_single_shooting.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

A = {k: np.array([r[k] for r in rows]) for k in rows[0]}
lines = ["| horizon (s) | initial rho | smoothing | locations | i_sat median [q25, q75] (not reached) | i_acc median [q25, q75] (not reached) | rho at i_acc median [q25, q75] (min) | rho_100 median [q25, q75] (min) | E_100/E_0 median [q25, q75] | iterations accepted while the replay violates (total, runs, smallest exact robustness among accepted iterations) |", "|---|---|---|---|---|---|---|---|---|---|"]
def cell(x, miss=None, mn=False, nd=3):
    x = x[~np.isnan(x)]
    m = q(x)
    s = f"{round(float(m[0]), nd)} [{round(float(m[1]), nd)}, {round(float(m[2]), nd)}]" if len(x) else "none"
    if miss is not None: s += f" ({miss})"
    if mn and len(x): s += f" ({round(float(x.min()), nd)})"
    return s
for H in (10, 20, 40):
    for d in (0.05, 0.10):
        for m in names:
            sel = (A["H"] == H) & (np.abs(A["w"] - late[H]) < 1e-6) & (np.abs(A["depth"] - d) < 1e-9) & (A["method"] == m) & (np.abs(A["eps"] - 0.2) < 1e-9)
            n = int(sel.sum())
            isat, iacc = A["i_sat"][sel].astype(float), A["i_acc"][sel].astype(float)
            lines.append(f"| {H} | {-d} | {names[m]} | {n} | " + cell(np.where(isat >= 0, isat, np.nan), int((isat < 0).sum()), nd=1) + " | " + cell(np.where(iacc >= 0, iacc, np.nan), int((iacc < 0).sum()), nd=1)
                         + " | " + cell(A["rho_at_acc"][sel], mn=True) + " | " + cell(A["rho100"][sel], mn=True) + " | " + cell(A["E100"][sel] / A["E0"][sel], nd=2) + f" | {int(A['accept_while_violating'][sel].sum())} in {int((A['accept_while_violating'][sel] > 0).sum())} runs, smallest rho {round(float(A['smallest_rho_among_accepted'][sel].min()), 4)} |")
open(out + "/table_outcomes.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
# paired comparison per location: sparsemax against the sound LSE
print()
for H in (10, 20, 40):
    for d in (0.05, 0.10):
        base = (A["H"] == H) & (np.abs(A["w"] - late[H]) < 1e-6) & (np.abs(A["depth"] - d) < 1e-9) & (np.abs(A["eps"] - 0.2) < 1e-9)
        a, b = base & (A["method"] == "sparsemax"), base & (A["method"] == "lse")
        oa, ob = np.argsort(A["location"][a]), np.argsort(A["location"][b])
        da = A["i_sat"][a][oa] - A["i_sat"][b][ob]
        dc = A["i_acc"][a][oa] - A["i_acc"][b][ob]
        print(f"H {H} rho0 {-d}: i_sat sparsemax minus sound LSE: earlier/equal/later {int((da<0).sum())}/{int((da==0).sum())}/{int((da>0).sum())}, median {np.median(da)}; i_acc: earlier/equal/later {int((dc<0).sum())}/{int((dc==0).sum())}/{int((dc>0).sum())}, median {np.median(dc)} (both reached in {int(((A['i_acc'][a][oa]>=0)&(A['i_acc'][b][ob]>=0)).sum())})")
