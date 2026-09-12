"""Score simulator/dataflow.py's `recompute` lever against a real GPU profile
from `profile_recompute.py`.

Runs locally, no GPU. Same `gpu_specs.py` mapping (and the same GPU-realism
knobs) as the other two anchors -- directly comparable, not a separate
calibration.

    python validate/silicon/compare_recompute.py recompute_profile.json
"""

import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simulator.dataflow import Layer, training_step  # noqa: E402
from validate.silicon import gpu_specs  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_MODES = ["stored", "recompute"]


def ci95(d, iters):
    return 1.96 * d.get("std_ms", 0.0) / math.sqrt(max(1, iters))


def predict(prof, spec):
    mac_bytes = 4 if prof.get("dtype") == "float32" else 2
    # NOTE: this anchor's shape mix (2x 768x768 + 768x3072 + 3072x768) was
    # tried with `chained=True` (the depth-isolated corrections calibrated
    # on a uniform 768x768 chain, docs/silicon_anchor.md) and it barely
    # moved this anchor's own backward error (106.1% -> 105.2%) -- that
    # calibration doesn't transfer to this shape mix, so left off here
    # rather than presented as a fix it isn't.
    hw = spec.hardware(mac_bytes=mac_bytes)
    mem = spec.memory()
    layers = [Layer(f"l{i}", M=prof["tokens"], K=k, N=n)
             for i, (k, n) in enumerate(prof["chain_dims"])]
    out = {}
    for mode, recompute in (("stored", False), ("recompute", True)):
        fwd = sum(training_step(l, hw, mem, recompute=recompute).forward_s
                  for l in layers) * 1e3
        bwd = sum(training_step(l, hw, mem, recompute=recompute).backward_s
                  for l in layers) * 1e3
        out[mode] = {"forward": fwd, "backward": bwd, "total": fwd + bwd}
    return hw, mem, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", help="recompute_profile.json from profile_recompute.py")
    ap.add_argument("--gpu", help="force a spec name from gpu_specs.catalog()")
    ap.add_argument("--tflops", type=float)
    ap.add_argument("--l2-mb", type=float)
    ap.add_argument("--hbm-gbs", type=float)
    ap.add_argument("--clock-ghz", type=float)
    args = ap.parse_args()

    with open(args.profile) as f:
        prof = json.load(f)

    if prof.get("synthetic"):
        print("\n  !!  SYNTHETIC SAMPLE -- numbers below are illustrative only.")
        print("      Replace with a real profile_recompute.py run on a GPU.\n")

    overrides = [args.tflops, args.l2_mb, args.hbm_gbs, args.clock_ghz]
    if any(v is not None for v in overrides):
        if not all(v is not None for v in overrides):
            sys.exit("pass all four of --tflops --l2-mb --hbm-gbs --clock-ghz")
        spec = gpu_specs.from_overrides(prof.get("gpu"), *overrides)
    elif args.gpu:
        spec = next((s for s in gpu_specs.catalog() if s.name == args.gpu), None)
        if spec is None:
            sys.exit(f"unknown --gpu {args.gpu!r}; see gpu_specs.catalog()")
    else:
        spec = gpu_specs.lookup(prof.get("gpu", ""))
        if spec is None:
            sys.exit(f"no spec match for {prof.get('gpu')!r}; pass --gpu or "
                     f"--tflops/--l2-mb/--hbm-gbs/--clock-ghz")

    hw, mem, pred = predict(prof, spec)

    print(f"profile : {prof.get('gpu')}   dtype={prof.get('dtype')}   "
          f"tokens={prof['tokens']}   chain={prof['chain_dims']}")
    print(f"spec    : {spec.name}  array {hw.rows}x{hw.cols}  "
          f"eff {hw.compute_efficiency}  L2 {spec.l2_mb:.0f} MB  HBM {spec.hbm_gbs:.0f} GB/s")
    print()

    iters = prof.get("iters", 50)
    rows = []
    print(f"  {'mode':10s} {'phase':9s} {'measured':>10s} {'95% CI':>9s} "
          f"{'predicted':>10s} {'ratio':>7s} {'abs %err':>9s}")
    for mode in _MODES:
        for ph in ("forward", "backward"):
            d = prof[mode][ph]
            m, p = d["mean_ms"], pred[mode][ph]
            ci = ci95(d, iters)
            ratio = p / m if m else float("nan")
            err = abs(p - m) / m * 100 if m else float("nan")
            rows.append((mode, ph, m, ci, p, ratio, err))
            print(f"  {mode:10s} {ph:9s} {m:10.3f} {'±' + f'{ci:.3f}':>9s} "
                  f"{p:10.3f} {ratio:7.2f} {err:8.1f}%")
        tot_m = prof[mode]["forward"]["mean_ms"] + prof[mode]["backward"]["mean_ms"]
        tot_p = pred[mode]["total"]
        print(f"  {mode:10s} {'total':9s} {tot_m:10.3f} {'':>9s} {tot_p:10.3f} "
              f"{tot_p / tot_m:7.2f} {abs(tot_p - tot_m) / tot_m * 100:8.1f}%")
        rows.append((mode, "total", tot_m, float("nan"), tot_p, tot_p / tot_m,
                     abs(tot_p - tot_m) / tot_m * 100))

    stored_tot_m = prof["stored"]["forward"]["mean_ms"] + prof["stored"]["backward"]["mean_ms"]
    recompute_tot_m = prof["recompute"]["forward"]["mean_ms"] + prof["recompute"]["backward"]["mean_ms"]
    meas_delta = (recompute_tot_m - stored_tot_m) / stored_tot_m * 100
    pred_delta = (pred["recompute"]["total"] - pred["stored"]["total"]) / pred["stored"]["total"] * 100
    print(f"\n  measured recompute delta: {meas_delta:+.1f}%    "
          f"predicted recompute delta: {pred_delta:+.1f}%")
    print(f"  (positive = recompute is slower, i.e. this config is compute-bound)")

    csv_path = os.path.join(_ROOT, "results", "validation", "recompute_silicon.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gpu", "spec", "dtype", "mode", "phase", "measured_ms",
                    "ci95_ms", "predicted_ms", "ratio", "abs_pct_err"])
        for mode, ph, m, ci, p, ratio, err in rows:
            w.writerow([prof.get("gpu"), spec.name, prof.get("dtype"), mode, ph,
                        f"{m:.4f}", "" if math.isnan(ci) else f"{ci:.4f}",
                        f"{p:.4f}", f"{ratio:.4f}", f"{err:.2f}"])
    print(f"\nwrote {csv_path}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["stored", "recompute"]
    meas_vals = [stored_tot_m, recompute_tot_m]
    pred_vals = [pred["stored"]["total"], pred["recompute"]["total"]]
    x = range(len(labels))
    w = 0.35
    ax.bar([i - w / 2 for i in x], meas_vals, w, label="measured (GPU)", color="#333")
    ax.bar([i + w / 2 for i in x], pred_vals, w, label="predicted (accel-sim)", color="#1f77b4")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("forward+backward latency (ms)")
    title = (f"Recompute silicon anchor: {prof.get('gpu')}  "
            f"(measured {meas_delta:+.0f}%, predicted {pred_delta:+.0f}%)")
    if prof.get("synthetic"):
        title = "SYNTHETIC SAMPLE -- " + title
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    plot_path = os.path.join(_ROOT, "results", "plots", "recompute_silicon.png")
    fig.savefig(plot_path, dpi=130)
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
