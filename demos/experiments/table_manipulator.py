"""The table of the until conjunct on the manipulator's violating initial trajectories (Table II of the paper), from
the files that experiments.e045_two_properties writes.

    python experiments/table_manipulator.py <gpu.csv|-> <prefix> ...

<prefix>_cpu.csv (mode cpu, one prefix per horizon): the value of the until conjunct and the sum of the leaf weights
on the samples inside the zone. <gpu.csv> (mode gpu; - leaves the columns out): the norm of the part of the torque
gradient that passes through those samples. Medians over the initial trajectories with exact robustness -0.05, at
eps 0.2 for the measures that have one; value at the shortest horizon, weight and gradient at the shortest and the
longest.
"""
import csv
import sys

import numpy as np

names = {"lse_plain": "plain LSE", "lse": "sound LSE", "gm_pm01": "GMR (0,1)", "gm_pm10": "GMR (-10,10)", "sparsemax": "sparsemax"}
no_eps = ("gm_pm01", "gm_pm10")


def read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def median(rows, col, H, arm):
    eps = "nan" if arm in no_eps else "0.2"
    return float(np.median([float(r[col]) for r in rows
                            if int(float(r["horizon"])) == H and r["arm"] == arm and r["eps"] == eps and abs(float(r["depth"]) - 0.05) < 1e-9]))


cpu = [r for prefix in sys.argv[2:] for r in read(prefix + "_cpu.csv")]
gpu = read(sys.argv[1]) if sys.argv[1] != "-" else None
Hs = sorted({int(float(r["horizon"])) for r in cpu})
lo, hi = Hs[0], Hs[-1]
head = ["measure", "value, " + str(lo) + " s", "weight, " + str(lo) + " s", "weight, " + str(hi) + " s"]
if gpu:
    head += ["torque gradient, " + str(lo) + " s", "torque gradient, " + str(hi) + " s"]
lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head),
         "| exact | " + str(round(median(cpu, "exact", lo, "sparsemax"), 4)) + " |" + " |" * (len(head) - 2)]
for arm in names:  # table assembly, one row per measure
    cells = [round(median(cpu, "value", lo, arm), 4), round(median(cpu, "w_viol", lo, arm), 4), round(median(cpu, "w_viol", hi, arm), 4)]
    if gpu:
        cells += [round(median(gpu, "grad_norm_viol", lo, arm), 3), round(median(gpu, "grad_norm_viol", hi, arm), 3)]
    lines.append("| " + names[arm] + " | " + " | ".join(str(c) for c in cells) + " |")
print("\n".join(lines))
