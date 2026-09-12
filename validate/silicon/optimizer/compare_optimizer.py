"""Score simulator/dataflow.py's optimizer-state-precision lever
(`Optimizer.state_bytes`) against a real GPU profile from
`profile_optimizer.py`.

Runs locally, no GPU. Same `gpu_specs.py` mapping as the other anchors.

    python validate/silicon/compare_optimizer.py optimizer_profile.json
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simulator.dataflow import Layer, optimizer_seconds, Optimizer  # noqa: E402
from validate.silicon import gpu_specs  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_STATES = [("fp32", Optimizer(state_bytes=4)), ("bf16", Optimizer(state_bytes=2))]


def predict_total(prof, spec, opt):
    hw = spec.hardware(mac_bytes=2)
    mem = spec.memory()
    total_s = 0.0
    for k, n in prof["chain_dims"]:
        layer = Layer("l", M=1, K=k, N=n)
        total_s += optimizer_seconds(layer, mem, opt, hw)
    return total_s * 1e3


def refit_elementwise(prof, spec):
    """Fit `vector_flops_per_s` (native/fp32 elementwise rate) and
    `emulation_penalty` (non-native/bf16 multiplier) from this profile's
    per-layer data -- the cleanest, most isolated anchor available (no
    forward/backward confound). Two independent 1D fits, not a joint
    optimization: fp32 is native everywhere, so its residual over the
    existing bytes-only prediction isolates the elementwise rate alone;
    bf16's residual, using that already-fixed rate, isolates the emulation
    penalty alone.
    """
    import dataclasses
    mem = spec.memory()
    # Baseline for the residual must be the bytes-only prediction (no
    # elementwise term at all), regardless of whatever vector_flops_per_s /
    # emulation_penalty gpu_specs.py currently has installed -- otherwise a
    # re-run after applying a previous fit corrupts its own baseline.
    zero_hw = dataclasses.replace(spec.hardware(mac_bytes=2), vector_flops_per_s=0.0)
    sizes = [k * n for k, n in prof["chain_dims"]]

    def residual_seconds(state_name, opt):
        per_layer = prof["results"][state_name]["per_layer"]
        out = []
        for n, layer_result, (k, nn) in zip(sizes, per_layer, prof["chain_dims"]):
            layer = Layer("l", M=1, K=k, N=nn)
            existing_pred_s = optimizer_seconds(layer, mem, opt, zero_hw)
            measured_s = layer_result["mean_ms"] / 1e3
            out.append(measured_s - existing_pred_s)
        return out

    fp32_residuals = residual_seconds("fp32", _STATES[0][1])
    # grid search over candidate native elementwise rates, minimizing summed
    # squared relative error against each layer's residual.
    rate_candidates = [10 ** (8.5 + 0.02 * i) for i in range(126)]   # ~3.2e8 .. 5e10
    best_rate, best_err = None, float("inf")
    for rate in rate_candidates:
        err = 0.0
        for n, r in zip(sizes, fp32_residuals):
            pred = n / rate
            err += ((pred - r) / r) ** 2 if r > 0 else 0.0
        if err < best_err:
            best_rate, best_err = rate, err

    bf16_residuals = residual_seconds("bf16", _STATES[1][1])
    penalty_candidates = [round(0.5 + 0.02 * i, 3) for i in range(226)]   # 0.5 .. 5.0
    best_penalty, best_err2 = None, float("inf")
    for pen in penalty_candidates:
        err = 0.0
        for n, r in zip(sizes, bf16_residuals):
            pred = (n / best_rate) * pen
            err += ((pred - r) / r) ** 2 if r > 0 else 0.0
        if err < best_err2:
            best_penalty, best_err2 = pen, err

    return best_rate, best_penalty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", help="optimizer_profile.json from profile_optimizer.py")
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
        print("      Replace with a real profile_optimizer.py run on a GPU.\n")

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

    print(f"profile : {prof.get('gpu')}   layer sizes={prof['layer_sizes']}   "
          f"torch={prof.get('torch')}")
    print(f"spec    : {spec.name}\n")

    rows = []
    print(f"  {'state':6s} {'measured ms':>12s} {'predicted ms':>13s} {'ratio':>7s}")
    for name, opt in _STATES:
        meas = prof["results"][name]["total_ms"]
        pred = predict_total(prof, spec, opt)
        ratio = pred / meas if meas else float("nan")
        rows.append((name, meas, pred, ratio))
        print(f"  {name:6s} {meas:12.4f} {pred:13.4f} {ratio:7.2f}")

    meas_saved = (rows[0][1] - rows[1][1]) / rows[0][1] * 100
    pred_saved = (rows[0][2] - rows[1][2]) / rows[0][2] * 100
    print(f"\n  measured bf16 saving: {meas_saved:+.1f}%    predicted bf16 saving: {pred_saved:+.1f}%")

    if not prof.get("synthetic"):
        rate, penalty = refit_elementwise(prof, spec)
        print(f"\n  refit (against this profile's per-layer data, bytes-only "
              f"prediction as baseline):")
        print(f"    vector_flops_per_s (native/fp32 elementwise rate) = {rate:.3e}")
        print(f"    emulation_penalty  (non-native/bf16 multiplier)   = {penalty:.2f}x")

    csv_path = os.path.join(_ROOT, "results", "validation", "optimizer_silicon.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gpu", "spec", "state", "measured_ms", "predicted_ms", "ratio"])
        for name, meas, pred, ratio in rows:
            w.writerow([prof.get("gpu"), spec.name, name, f"{meas:.4f}", f"{pred:.4f}", f"{ratio:.4f}"])
    print(f"\nwrote {csv_path}")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels = [r[0] for r in rows]
    x = range(len(labels))
    w = 0.35
    ax.bar([i - w / 2 for i in x], [r[1] for r in rows], w, label="measured (GPU)", color="#333")
    ax.bar([i + w / 2 for i in x], [r[2] for r in rows], w, label="predicted (accel-sim)", color="#1f77b4")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("optimizer step latency (ms)")
    title = f"Optimizer-state precision anchor: {prof.get('gpu')}"
    if prof.get("synthetic"):
        title = "SYNTHETIC SAMPLE -- " + title
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    plot_path = os.path.join(_ROOT, "results", "plots", "optimizer_silicon.png")
    fig.savefig(plot_path, dpi=130)
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
