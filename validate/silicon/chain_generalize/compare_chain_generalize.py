"""Does the chain-depth fix (`compare_chain_depth.py`, fit on a skinny
768x768 shape) generalize across shape-class (occupancy-safe vs skinny)
and token count? See `profile_chain_generalize.py`'s docstring for the
three things this specifically isolates.

Runs locally, no GPU.

    python validate/silicon/compare_chain_generalize.py chain_generalize_profile.json
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

from simulator.dataflow import Layer, training_step  # noqa: E402
from validate.silicon import gpu_specs  # noqa: E402
from validate.silicon.chain_depth.compare_chain_depth import ols_fit  # noqa: E402


def _predict(shape, tokens, depth, hw, mem, with_chain_overhead=False):
    k, n = shape
    layer = Layer("l", M=tokens, K=k, N=n)
    p = training_step(layer, hw, mem)
    fwd_ms = p.forward_s * 1e3 * depth
    bwd_ms = p.backward_s * 1e3 * depth
    if with_chain_overhead:
        fwd_ms += hw.chain_overhead_fwd_s * 1e3
        bwd_ms += hw.chain_overhead_bwd_s * 1e3
    return fwd_ms, bwd_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--gpu", default=None)
    args = ap.parse_args()

    with open(args.profile) as f:
        prof = json.load(f)

    if prof.get("synthetic"):
        print("\n  !!  SYNTHETIC SAMPLE -- numbers below are illustrative only.")
        print("      Replace with a real profile_chain_generalize.py run on a GPU.\n")

    spec = gpu_specs.lookup(args.gpu or prof.get("gpu"))
    if spec is None:
        sys.exit(f"unknown GPU {prof.get('gpu')!r} -- pass --gpu")
    old_hw = spec.hardware(mac_bytes=2)
    new_hw = spec.hardware(mac_bytes=2, chained=True)   # 768x768-derived fit, reused here
    mem = spec.memory()
    shape = tuple(prof["shape"])
    results = prof["results"]

    print(f"profile : {prof.get('gpu')}   shape={shape} (occupancy-safe)")
    print(f"spec    : {spec.name}\n")

    # --- 1. depth sweep at M=8192: fresh OLS fit, compared to the 768x768 fit ---
    depth_names = ["depth1_m8192", "depth2_m8192", "depth4_m8192", "depth8_m8192"]
    print("-- depth sweep @ M=8192 (occupancy-safe shape) --")
    print(f"  {'config':16s} {'depth':>5s} {'meas bwd':>9s} {'old pred':>9s} "
          f"{'new pred':>9s} {'old err%':>9s} {'new err%':>9s}")
    depth_xs, depth_ys = [], []
    rows = []
    for name in depth_names:
        r = results[name]
        depth, m_bwd = r["depth"], r["backward"]["mean_ms"]
        _, old_bwd = _predict(shape, r["tokens"], depth, old_hw, mem)
        _, new_bwd = _predict(shape, r["tokens"], depth, new_hw, mem, with_chain_overhead=True)
        old_err = abs(old_bwd - m_bwd) / m_bwd * 100
        new_err = abs(new_bwd - m_bwd) / m_bwd * 100
        depth_xs.append(depth); depth_ys.append(m_bwd)
        rows.append((name, depth, r["tokens"], m_bwd, old_bwd, new_bwd, old_err, new_err))
        print(f"  {name:16s} {depth:5d} {m_bwd:9.3f} {old_bwd:9.3f} {new_bwd:9.3f} "
              f"{old_err:8.1f}% {new_err:8.1f}%")

    A, B, r2 = ols_fit(depth_xs, depth_ys)
    print(f"\n  fresh fit on THIS shape: measured_backward_ms ~= {A:.4f}*depth + {B:.4f}  (R^2={r2:.3f})")
    print(f"  768x768's own fit was:  measured_backward_ms ~= 0.4905*depth + 1.3610")
    print(f"  -> chain_overhead_bwd_s is {'similar (generalizes)' if abs(B - 1.361) / 1.361 < 0.3 else 'DIFFERENT (does not transfer as a flat constant)'}"
          f" across shape-class: {B:.3f} vs 1.361 ms")

    # --- 2. token-count sweep at depth=4: does the residual (proxy for
    #        overhead) stay flat, or scale with M? ---
    m_names = ["depth4_m512", "depth4_m1024", "depth4_m2048", "depth4_m4096",
              "depth4_m8192", "depth4_m16384"]
    print("\n-- token-count sweep @ depth=4 (occupancy-safe shape) --")
    print(f"  {'config':16s} {'tokens':>7s} {'meas bwd':>9s} {'old pred':>9s} "
          f"{'residual':>9s}")
    m_rows = []
    for name in m_names:
        r = results[name]
        depth, M, m_bwd = r["depth"], r["tokens"], r["backward"]["mean_ms"]
        _, old_bwd = _predict(shape, M, depth, old_hw, mem)
        residual = m_bwd - old_bwd
        m_rows.append((name, M, m_bwd, old_bwd, residual))
        print(f"  {name:16s} {M:7d} {m_bwd:9.3f} {old_bwd:9.3f} {residual:9.3f}")

    residuals = [r[4] for r in m_rows]
    mean_res = sum(residuals) / len(residuals)
    spread = (max(residuals) - min(residuals)) / abs(mean_res) if mean_res else float("inf")
    print(f"\n  mean residual: {mean_res:.3f} ms   spread (max-min)/mean: {spread:.2f}")
    print(f"  -> overhead looks {'roughly M-independent (flat residual)' if spread < 0.5 else 'M-DEPENDENT (residual varies a lot with tokens)'}")

    # --- 3. noise check: repeated config ---
    a = results["depth4_m8192"]["backward"]["mean_ms"]
    b = results["repeat_depth4_m8192"]["backward"]["mean_ms"]
    drift = abs(a - b) / a * 100
    print(f"\n-- repeated-config noise check --")
    print(f"  depth4_m8192 (first run):  {a:.3f} ms")
    print(f"  repeat_depth4_m8192 (last): {b:.3f} ms")
    print(f"  drift: {drift:.1f}% -- {'looks like real session noise, treat small effects with caution' if drift > 5 else 'stable, non-generalization findings are unlikely to be measurement noise'}")

    os.makedirs("results/validation", exist_ok=True)
    csv_path = "results/validation/chain_generalize_silicon.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gpu", "config", "depth", "tokens", "measured_bwd_ms",
                    "old_pred_bwd_ms", "new_pred_bwd_ms", "old_err_pct", "new_err_pct"])
        for name, depth, M, m_bwd, old_bwd, new_bwd, old_err, new_err in rows:
            w.writerow([prof.get("gpu"), name, depth, M, f"{m_bwd:.4f}",
                        f"{old_bwd:.4f}", f"{new_bwd:.4f}", f"{old_err:.2f}", f"{new_err:.2f}"])
    print(f"\nwrote {csv_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(depth_xs, depth_ys, "o", color="#333", label="measured")
    xs_line = [min(depth_xs), max(depth_xs)]
    ax1.plot(xs_line, [A * x + B for x in xs_line], "--", color="#2ca02c", label=f"fresh fit (R^2={r2:.2f})")
    ax1.set_xlabel("depth"); ax1.set_ylabel("backward (ms)")
    ax1.set_title("Depth sweep (occupancy-safe shape)")
    ax1.legend(); ax1.grid(alpha=0.3)

    m_xs = [r[1] for r in m_rows]
    ax2.plot(m_xs, [r[4] for r in m_rows], "o-", color="#d62728")
    ax2.set_xscale("log", base=2)
    ax2.axhline(1.361, ls="--", color="#999", label="768x768's own fitted overhead")
    ax2.set_xlabel("tokens (M)"); ax2.set_ylabel("residual vs old model (ms)")
    ax2.set_title("Does overhead scale with M?")
    ax2.legend(); ax2.grid(alpha=0.3)

    title = f"Chain generalization test: {prof.get('gpu')}"
    if prof.get("synthetic"):
        title = "SYNTHETIC SAMPLE -- " + title
    fig.suptitle(title)
    fig.tight_layout()
    plot_path = "results/plots/chain_generalize_silicon.png"
    fig.savefig(plot_path, dpi=130)
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
