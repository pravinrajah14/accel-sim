"""Score simulator/dataflow.py's `microbatch=` lever against a real GPU
profile from `profile_gradaccum.py`.

Runs locally, no GPU. Same `gpu_specs.py` mapping as the other anchors.

    python validate/silicon/compare_gradaccum.py gradaccum_profile.json
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


def ci95(d, iters):
    return 1.96 * d.get("std_ms", 0.0) / math.sqrt(max(1, iters))


def predict(prof, spec, frac, mem=None):
    mac_bytes = 4 if prof.get("dtype") == "float32" else 2
    # NOTE: `chained=True` (the depth-isolated corrections, calibrated on a
    # uniform 768x768 chain -- docs/silicon_anchor.md) was tried against
    # this anchor's own mixed-shape chain and did not meaningfully improve
    # it -- that calibration is specific to the shape/token-count it was
    # fit on, not a general GPU constant. Left off here rather than
    # presented as a fix it isn't.
    hw = spec.hardware(mac_bytes=mac_bytes)
    if mem is None:
        mem = spec.memory()
    M = prof["tokens"]
    mb = max(1, M // int(frac))
    layers = [Layer(f"l{i}", M=M, K=k, N=n) for i, (k, n) in enumerate(prof["chain_dims"])]
    parts = [training_step(l, hw, mem, microbatch=mb) for l in layers]
    step_ms = sum(p.forward_s + p.backward_s for p in parts) * 1e3
    raw_act_mb = sum(p.activation_bytes for p in parts) / 1e6
    act_mb = mem.framework_overhead_bytes / 1e6 + mem.framework_overhead_scale * raw_act_mb
    return step_ms, act_mb


def fit_memory_floor(prof, spec):
    """OLS fit of measured_mb ~= floor_mb + scale * raw_predicted_mb across
    every fraction in this profile -- the real activation-memory floor
    (weight buffers, autograd bookkeeping, allocator overhead) plus a
    scale correction, both fit at once since the residual isn't flat (it
    shrinks as the fraction grows -- some overhead scales with chunk size,
    some doesn't). 5 real data points spanning a 16x range support a
    2-parameter fit here; not extrapolated beyond that range.
    """
    base_mem = spec.memory()
    fracs = prof["configs"].keys()
    xs, ys = [], []
    for frac in fracs:
        _, raw_mb = predict(prof, spec, frac, mem=base_mem)
        xs.append(raw_mb)
        ys.append(prof["configs"][frac]["peak_activation_mb"])
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxx = sum(x * x for x in xs)
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0, 1.0
    scale = (n * sxy - sx * sy) / denom
    floor_mb = (sy - scale * sx) / n
    return floor_mb, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", help="gradaccum_profile.json from profile_gradaccum.py")
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
        print("      Replace with a real profile_gradaccum.py run on a GPU.\n")

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

    print(f"profile : {prof.get('gpu')}   dtype={prof.get('dtype')}   "
          f"tokens={prof['tokens']}   chain={prof['chain_dims']}")
    print(f"spec    : {spec.name}\n")

    iters = prof.get("iters", 30)
    fracs = sorted(prof["configs"].keys(), key=int)
    rows = []
    print(f"  {'frac':>5s} {'meas step':>10s} {'95%CI':>8s} {'pred step':>10s} {'ratio':>7s}"
          f"  {'meas MB':>9s} {'raw pred MB':>11s} {'ratio':>7s}")
    for frac in fracs:
        d = prof["configs"][frac]
        pred_ms, pred_mb = predict(prof, spec, frac)
        ci = ci95(d, iters)
        m_ms = d["mean_ms"]
        m_mb = d["peak_activation_mb"]
        ratio_ms = pred_ms / m_ms if m_ms else float("nan")
        ratio_mb = pred_mb / m_mb if m_mb else float("nan")
        rows.append((frac, m_ms, ci, pred_ms, ratio_ms, m_mb, pred_mb, ratio_mb))
        print(f"  {frac:>5s} {m_ms:10.3f} {'±'+f'{ci:.2f}':>8s} {pred_ms:10.3f} {ratio_ms:7.2f}"
              f"  {m_mb:9.1f} {pred_mb:11.1f} {ratio_mb:7.2f}")

    base_ms = prof["configs"]["1"]["mean_ms"]
    print()
    for frac in fracs[1:]:
        d = prof["configs"][frac]
        delta = (d["mean_ms"] - base_ms) / base_ms * 100
        print(f"  measured latency cost at fraction {frac}: {delta:+.1f}%")

    fitted_mem = None
    if not prof.get("synthetic"):
        import dataclasses
        floor_mb, scale = fit_memory_floor(prof, spec)
        fitted_mem = dataclasses.replace(spec.memory(), framework_overhead_bytes=floor_mb * 1e6,
                                         framework_overhead_scale=scale)
        print(f"\n  memory-floor fit: measured_mb ~= {floor_mb:.1f} + {scale:.2f} * raw_predicted_mb")
        print(f"\n  {'frac':>5s} {'meas MB':>9s} {'adj pred MB':>11s} {'adj ratio':>9s}"
              f"  {'raw ratio':>9s}")
        for frac, *_rest, m_mb, raw_pred_mb, raw_ratio in rows:
            _, adj_mb = predict(prof, spec, frac, mem=fitted_mem)
            adj_ratio = adj_mb / m_mb if m_mb else float("nan")
            print(f"  {frac:>5s} {m_mb:9.1f} {adj_mb:11.1f} {adj_ratio:9.2f}  {raw_ratio:9.2f}")

    csv_path = os.path.join(_ROOT, "results", "validation", "gradaccum_silicon.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gpu", "spec", "dtype", "fraction", "measured_ms", "ci95_ms",
                    "predicted_ms", "ratio_ms", "measured_mb", "raw_predicted_mb", "raw_ratio_mb",
                    "adj_predicted_mb", "adj_ratio_mb"])
        for frac, m_ms, ci, pred_ms, ratio_ms, m_mb, pred_mb, ratio_mb in rows:
            if fitted_mem is not None:
                _, adj_mb = predict(prof, spec, frac, mem=fitted_mem)
                adj_ratio = adj_mb / m_mb if m_mb else float("nan")
            else:
                adj_mb, adj_ratio = pred_mb, ratio_mb
            w.writerow([prof.get("gpu"), spec.name, prof.get("dtype"), frac,
                        f"{m_ms:.4f}", f"{ci:.4f}", f"{pred_ms:.4f}", f"{ratio_ms:.4f}",
                        f"{m_mb:.4f}", f"{pred_mb:.4f}", f"{ratio_mb:.4f}",
                        f"{adj_mb:.4f}", f"{adj_ratio:.4f}"])
    print(f"\nwrote {csv_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    xs = [int(f) for f in fracs]
    ax1.plot(xs, [prof["configs"][f]["mean_ms"] for f in fracs], "o-", color="#333", label="measured")
    ax1.plot(xs, [predict(prof, spec, f)[0] for f in fracs], "o-", color="#1f77b4", label="predicted")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("microbatch fraction (n chunks)")
    ax1.set_ylabel("step latency (ms)")
    ax1.set_title("Latency vs accumulation")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(xs, [prof["configs"][f]["peak_activation_mb"] for f in fracs], "o-", color="#333", label="measured")
    ax2.plot(xs, [predict(prof, spec, f)[1] for f in fracs], "o-", color="#1f77b4", label="raw predicted")
    if fitted_mem is not None:
        ax2.plot(xs, [predict(prof, spec, f, mem=fitted_mem)[1] for f in fracs], "o--",
                color="#2ca02c", label="floor-fit predicted")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("microbatch fraction (n chunks)")
    ax2.set_ylabel("peak activation memory (MB, log)")
    ax2.set_title("Activation memory vs accumulation")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend()

    title = f"Gradient accumulation silicon anchor: {prof.get('gpu')}"
    if prof.get("synthetic"):
        title = "SYNTHETIC SAMPLE -- " + title
    fig.suptitle(title)
    fig.tight_layout()
    plot_path = os.path.join(_ROOT, "results", "plots", "gradaccum_silicon.png")
    fig.savefig(plot_path, dpi=130)
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
