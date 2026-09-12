"""Score simulator/attention.py's naive and fused cost models against a real
GPU profile from `profile_attention.py`.

Runs locally, no GPU. Uses the same `gpu_specs.py` mapping (and the same
GPU-realism knobs -- compute_efficiency, util_tiles, library_gemm,
kernel_launch_s) as `compare_silicon.py`, so this is directly comparable to
the GEMM silicon anchor rather than a separate calibration.

    python validate/silicon/compare_attention.py attention_profile.json
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

from simulator.attention import Attention, attention_step  # noqa: E402
from validate.silicon import gpu_specs  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_MODES = ["naive", "fused"]
_PHASES = ["forward", "backward"]


def ci95(d, iters):
    """95% CI half-width (ms) for one {mean_ms, std_ms} measurement."""
    return 1.96 * d.get("std_ms", 0.0) / math.sqrt(max(1, iters))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", help="attention_profile.json from profile_attention.py")
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
        print("      Replace with a real profile_attention.py run on a GPU.\n")

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

    mac_bytes = 4 if prof.get("dtype") == "float32" else 2
    hw = spec.hardware(mac_bytes=mac_bytes)
    mem = spec.memory()
    attn = Attention("attention", batch=prof["batch"], seq=prof["seq"],
                     n_heads=prof["n_heads"], d_head=prof["d_head"])

    print(f"profile : {prof.get('gpu')}   dtype={prof.get('dtype')}   "
          f"B={prof['batch']} S={prof['seq']} H={prof['n_heads']} Dh={prof['d_head']}   "
          f"torch={prof.get('torch')}")
    print(f"spec    : {spec.name}  array {hw.rows}x{hw.cols}  "
          f"eff {hw.compute_efficiency}  util_tiles {hw.util_tiles}  "
          f"L2 {spec.l2_mb:.0f} MB  HBM {spec.hbm_gbs:.0f} GB/s")
    print()

    iters = prof.get("iters", 50)
    rows = []
    print(f"  {'mode':7s} {'phase':9s} {'measured':>10s} {'95% CI':>9s} "
          f"{'predicted':>10s} {'ratio':>7s} {'abs %err':>9s}")
    for mode in _MODES:
        pred = attention_step(attn, hw, mem, fused=(mode == "fused"))
        pred_by_phase = {"forward": pred.forward_s * 1e3, "backward": pred.backward_s * 1e3}
        for ph in _PHASES:
            d = prof[mode][ph]
            m, p = d["mean_ms"], pred_by_phase[ph]
            ci = ci95(d, iters)
            ratio = p / m if m else float("nan")
            err = abs(p - m) / m * 100 if m else float("nan")
            rows.append((mode, ph, m, ci, p, ratio, err))
            print(f"  {mode:7s} {ph:9s} {m:10.3f} {'±' + f'{ci:.3f}':>9s} "
                  f"{p:10.3f} {ratio:7.2f} {err:8.1f}%")
        tot_m = prof[mode]["forward"]["mean_ms"] + prof[mode]["backward"]["mean_ms"]
        tot_p = pred_by_phase["forward"] + pred_by_phase["backward"]
        print(f"  {mode:7s} {'total':9s} {tot_m:10.3f} {'':>9s} {tot_p:10.3f} "
              f"{tot_p / tot_m:7.2f} {abs(tot_p - tot_m) / tot_m * 100:8.1f}%")
        rows.append((mode, "total", tot_m, float("nan"), tot_p, tot_p / tot_m,
                     abs(tot_p - tot_m) / tot_m * 100))

    naive_tot_m = prof["naive"]["forward"]["mean_ms"] + prof["naive"]["backward"]["mean_ms"]
    fused_tot_m = prof["fused"]["forward"]["mean_ms"] + prof["fused"]["backward"]["mean_ms"]
    meas_saving = (naive_tot_m - fused_tot_m) / naive_tot_m * 100
    pred_naive = attention_step(attn, hw, mem, fused=False).total_s * 1e3
    pred_fused = attention_step(attn, hw, mem, fused=True).total_s * 1e3
    pred_saving = (pred_naive - pred_fused) / pred_naive * 100
    print(f"\n  measured fusion saving: {meas_saving:5.1f}%    "
          f"predicted fusion saving: {pred_saving:5.1f}%")

    # ---- diagnostic: did the fused path really dispatch to a fused kernel?
    ops = " ".join(o.get("op", "") for o in prof.get("fused_top_ops", [])).lower()
    fused_kernel = any(k in ops for k in ("flash", "efficient_attention", "cutlassf",
                                          "mem_eff", "fmha"))
    if prof.get("fused_top_ops") and not fused_kernel:
        print("\n  !! the 'fused' path's top kernels don't look like a fused")
        print("     flash/memory-efficient attention kernel -- check fused_top_ops in")
        print("     the JSON. PyTorch may have fallen back to the generic math backend")
        print("     (e.g. unsupported dtype/head-dim/GPU), in which case 'fused' here")
        print("     measures the same thing as 'naive' and this comparison is moot.")

    # ---- csv ----------------------------------------------------------
    csv_path = os.path.join(_ROOT, "results", "validation", "attention_silicon.csv")
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

    # ---- plot -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["naive fwd", "naive bwd", "fused fwd", "fused bwd"]
    meas_vals = [prof["naive"]["forward"]["mean_ms"], prof["naive"]["backward"]["mean_ms"],
                prof["fused"]["forward"]["mean_ms"], prof["fused"]["backward"]["mean_ms"]]
    pred_naive_part = attention_step(attn, hw, mem, fused=False)
    pred_fused_part = attention_step(attn, hw, mem, fused=True)
    pred_vals = [pred_naive_part.forward_s * 1e3, pred_naive_part.backward_s * 1e3,
                pred_fused_part.forward_s * 1e3, pred_fused_part.backward_s * 1e3]
    x = range(len(labels))
    w = 0.38
    ax.bar([i - w / 2 for i in x], meas_vals, w, label="measured (GPU)", color="#333")
    ax.bar([i + w / 2 for i in x], pred_vals, w, label="predicted (accel-sim)", color="#1f77b4")
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("latency (ms, log scale)")
    title = (f"Attention silicon anchor: {prof.get('gpu')} "
            f"(measured saving {meas_saving:.0f}%, predicted {pred_saving:.0f}%)")
    if prof.get("synthetic"):
        title = "SYNTHETIC SAMPLE -- " + title
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    plot_path = os.path.join(_ROOT, "results", "plots", "attention_silicon.png")
    fig.savefig(plot_path, dpi=130)
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
