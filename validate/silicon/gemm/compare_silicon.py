"""Score accel-sim against a real GPU profile from `profile_step.py`.

Runs locally, no GPU.  Loads `silicon_profile.json`, maps the GPU to an
accel-sim `Hardware` + `Memory` (see `gpu_specs.py`), runs the closed-form
`model_step`, and prints a phase-by-phase measured-vs-predicted table with
the mean absolute percentage error.  Writes:

    results/validation/silicon.csv
    results/plots/silicon_anchor.png

    python validate/silicon/compare_silicon.py silicon_profile.json
    python validate/silicon/compare_silicon.py p.json --gpu "A100-SXM4-80GB"
    python validate/silicon/compare_silicon.py p.json \
        --tflops 312 --l2-mb 40 --hbm-gbs 2039 --clock-ghz 1.41
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

from simulator.dataflow import Layer, model_step, ADAM_FP32  # noqa: E402
from validate.silicon import gpu_specs  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_PHASES = ["forward", "backward", "optimizer", "step"]


def build_block(tokens, d_model, d_ff):
    names = ["q_proj", "k_proj", "v_proj", "attn_out"]
    layers = [Layer(n, M=tokens, K=d_model, N=d_model) for n in names]
    layers.append(Layer("mlp_up", M=tokens, K=d_model, N=d_ff))
    layers.append(Layer("mlp_down", M=tokens, K=d_ff, N=d_model))
    return layers


def predict(prof, spec):
    hw = spec.hardware(mac_bytes=prof.get("mac_bytes", 2))
    mem = spec.memory()
    layers = build_block(prof["tokens"], prof["d_model"], prof["d_ff"])
    _, parts = model_step(layers, hw, mem, ADAM_FP32)
    by_name = {p.layer: p for p in parts}
    agg = {
        "forward": sum(p.forward_s for p in parts) * 1e3,
        "backward": sum(p.backward_s for p in parts) * 1e3,
        "optimizer": sum(p.optimizer_s for p in parts) * 1e3,
    }
    agg["step"] = agg["forward"] + agg["backward"] + agg["optimizer"]
    return hw, mem, agg, by_name


def _fmt(x):
    return f"{x:8.3f}"


def phase_confidence(prof):
    """95% CI half-width per phase (ms) for the *measured* mean.

    Each per-layer mean comes from `iters` timed repeats with sample std
    `std_ms`; its standard error is `std_ms / sqrt(iters)`. Summing phases
    across layers, assuming independence (each layer/phase was timed in its
    own loop), the aggregate SE is `sqrt(sum(se_i^2))`. This bounds sampling
    noise only -- not systematic error (thermal throttling, clock variance,
    the kernel-selection scatter documented in docs/silicon_anchor.md).
    """
    pl = prof.get("per_layer") or {}
    n = max(1, prof.get("iters", 50))
    ci = {}
    for ph in ("forward", "backward", "optimizer"):
        var_sum = sum((d.get(ph, {}).get("std_ms", 0.0) ** 2) / n for d in pl.values())
        ci[ph] = 1.96 * math.sqrt(var_sum)
    ci["step"] = math.sqrt(sum(ci[p] ** 2 for p in ("forward", "backward", "optimizer")))
    return ci


def noisy_layers(prof, threshold=0.15):
    """(layer, phase, relative_std) for any per-layer measurement whose
    std/mean exceeds `threshold` -- usually a cold-start / warmup artifact,
    not a real hardware effect."""
    pl = prof.get("per_layer") or {}
    out = []
    for name, d in pl.items():
        for ph in ("forward", "backward", "optimizer"):
            m, s = d.get(ph, {}).get("mean_ms", 0.0), d.get(ph, {}).get("std_ms", 0.0)
            if m > 0 and s / m > threshold:
                out.append((name, ph, s / m))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", help="silicon_profile.json from profile_step.py")
    ap.add_argument("--gpu", help="force a spec name from gpu_specs.catalog()")
    ap.add_argument("--tflops", type=float)
    ap.add_argument("--l2-mb", type=float)
    ap.add_argument("--hbm-gbs", type=float)
    ap.add_argument("--clock-ghz", type=float)
    ap.add_argument("--dram-eff", type=float, default=0.75)
    args = ap.parse_args()

    with open(args.profile) as f:
        prof = json.load(f)

    if prof.get("synthetic"):
        print("\n  !!  SYNTHETIC SAMPLE -- numbers below are illustrative only.")
        print("      Replace with a real profile_step.py run on a GPU.\n")

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

    hw, mem, pred, by_name = predict(prof, spec)
    meas = prof["phase_ms"]

    print(f"profile : {prof.get('gpu')}   dtype={prof.get('dtype')}   "
          f"tokens={prof['tokens']}   torch={prof.get('torch')}")
    print(f"spec    : {spec.name}  {spec.peak_tflops:.0f} TFLOP/s  "
          f"array {hw.rows}x{hw.cols} @ {spec.clock_ghz} GHz  "
          f"L2 {spec.l2_mb:.0f} MB  HBM {spec.hbm_gbs:.0f} GB/s  "
          f"(eff {args.dram_eff})")
    ci = phase_confidence(prof)
    print()
    print(f"  {'phase':10s} {'measured':>10s} {'95% CI':>9s} {'predicted':>10s} "
          f"{'ratio':>8s} {'abs %err':>9s}")
    errs = []
    rows = []
    for ph in _PHASES:
        m, p = meas[ph], pred[ph]
        ratio = p / m if m else float("nan")
        err = abs(p - m) / m * 100 if m else float("nan")
        if ph != "step":
            errs.append(err)
        rows.append((ph, m, p, ratio, err))
        in_ci = "" if m == 0 else ("  (in CI)" if abs(p - m) <= ci[ph] else "")
        print(f"  {ph:10s} {m:10.3f} {'±' + f'{ci[ph]:.3f}':>9s} {p:10.3f} "
              f"{ratio:8.2f} {err:8.1f}%{in_ci}")
    mape = sum(errs) / len(errs)
    print(f"\n  phase MAPE (fwd/bwd/opt): {mape:.1f}%     "
          f"step error: {rows[-1][4]:.1f}%")
    print(f"  (95% CI = sampling noise only, {prof.get('iters', 50)} iters/layer; "
          f"see phase_confidence() docstring -- systematic error is larger)")

    noisy = noisy_layers(prof)
    if noisy:
        print(f"\n  !! high run-to-run variance (std/mean > 15%, likely warmup/cold-start):")
        for name, ph, rel in noisy:
            print(f"     {name}/{ph}: {rel*100:.0f}% -- treat this cell with caution")

    # ---- diagnostic: did the GEMMs even use tensor cores? ---------------
    fwd_ratio = pred["forward"] / meas["forward"] if meas["forward"] else 1.0
    ops = " ".join(o.get("op", "") for o in prof.get("top_ops", [])).lower()
    fallback = any(k in ops for k in ("magma", "sgemm", "simt")) and \
        "cutlass" not in ops
    if fwd_ratio < 0.2 or fallback:
        print("\n  !! forward prediction is >5x below measured. The profiled GEMM")
        print("     kernels look like a NON-tensor-core fallback path"
              + (" (magma/sgemm in top_ops)" if fallback else "") + ".")
        dt, g = prof.get("dtype"), prof.get("gpu", "")
        if dt == "bfloat16" and ("T4" in g or "V100" in g):
            print(f"     {g} (Turing/Volta) has NO bf16 tensor cores -- torch fell")
            print("     back to a ~20x-slower SIMT GEMM. Re-run with DTYPE = \"fp16\"")
            print("     (fp16 tensor cores exist on these) or use A100/L4/H100.")
        else:
            print("     Check 'top_ops' in the JSON for a cuBLAS/cutlass tensor kernel.")
        print("     Until the GEMM path is tensor-core, this is a measurement")
        print("     artifact, not a model error.")

    # ---- per-layer table -------------------------------------------------
    pl = prof.get("per_layer") or {}
    if pl:
        print(f"\n  {'layer':10s} {'phase':10s} {'meas ms':>9s} {'±%':>5s} "
              f"{'pred ms':>9s} {'ratio':>7s}")
        for name, d in pl.items():
            part = by_name.get(name)
            if part is None:
                continue
            pred_layer = {"forward": part.forward_s * 1e3,
                          "backward": part.backward_s * 1e3,
                          "optimizer": part.optimizer_s * 1e3}
            for ph in ("forward", "backward", "optimizer"):
                m = d[ph]["mean_ms"]
                s = d[ph].get("std_ms", 0.0)
                p = pred_layer[ph]
                relstd = f"{s / m * 100:4.0f}" if m else "   -"
                print(f"  {name:10s} {ph:10s} {m:9.3f} {relstd:>5s} {p:9.3f} "
                      f"{(p / m if m else float('nan')):7.2f}")

    # ---- csv -----------------------------------------------------------
    csv_path = os.path.join(_ROOT, "results", "validation", "silicon.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gpu", "spec", "dtype", "tokens", "phase",
                    "measured_ms", "measured_ci95_ms", "predicted_ms",
                    "ratio", "abs_pct_err"])
        for ph, m, p, ratio, err in rows:
            w.writerow([prof.get("gpu"), spec.name, prof.get("dtype"),
                        prof["tokens"], ph, f"{m:.4f}", f"{ci[ph]:.4f}",
                        f"{p:.4f}", f"{ratio:.4f}", f"{err:.2f}"])
    print(f"\nwrote {csv_path}")

    # ---- plot --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = range(len(_PHASES))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [meas[p] for p in _PHASES], w,
           yerr=[ci[p] for p in _PHASES], capsize=3, ecolor="#999",
           label="measured (GPU)", color="#333")
    ax.bar([i + w / 2 for i in x], [pred[p] for p in _PHASES], w,
           label="predicted (accel-sim)", color="#1f77b4")
    for i, ph in enumerate(_PHASES):
        r = pred[ph] / meas[ph] if meas[ph] else float("nan")
        ax.annotate(f"{r:.2f}x", (i, max(meas[ph], pred[ph])),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9, color="gray")
    ax.set_xticks(list(x))
    ax.set_xticklabels(_PHASES)
    ax.set_ylabel("latency (ms)")
    title = f"Silicon anchor: {prof.get('gpu')}  (phase MAPE {mape:.1f}%)"
    if prof.get("synthetic"):
        title = "SYNTHETIC SAMPLE -- " + title
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    plot_path = os.path.join(_ROOT, "results", "plots", "silicon_anchor.png")
    fig.savefig(plot_path, dpi=130)
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
