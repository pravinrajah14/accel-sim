"""Test whether the isolated-vs-chained backward gap (`profile_chain_grid.py`)
is explained by chain DEPTH alone, with shape and token count held fixed.

Structural hypothesis: a real chained backward pays a roughly FIXED
one-time overhead (kernel-launch/dispatch amortization across one autograd
graph) plus a per-layer marginal compute cost, while `profile_step.py`'s
isolated calibration bakes a per-layer overhead into every layer
independently (since each of its 6 shapes pays its own full setup cost).
That predicts real chained backward is closely LINEAR in depth:

    measured_backward_ms(depth) ~= A * depth + B

with B a fixed per-chain overhead and A the true per-layer marginal cost.
If a linear fit has low R^2, depth alone doesn't explain the earlier
4-config grid's non-monotonic discount factor, and something else (shape,
token count, or GPU-side noise) dominates instead -- reported honestly
either way, not forced.

Runs locally, no GPU.

    python validate/silicon/compare_chain_depth.py chain_depth_profile.json
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


def ols_fit(xs, ys):
    """Closed-form OLS: y ~= A*x + B. Returns (A, B, r_squared)."""
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxx = sum(x * x for x in xs)
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0, sy / n if n else 0.0, 0.0
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    y_mean = sy / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, b, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--gpu", default=None)
    args = ap.parse_args()

    with open(args.profile) as f:
        prof = json.load(f)

    if prof.get("synthetic"):
        print("\n  !!  SYNTHETIC SAMPLE -- numbers below are illustrative only.")
        print("      Replace with a real profile_chain_depth.py run on a GPU.\n")

    spec = gpu_specs.lookup(args.gpu or prof.get("gpu"))
    if spec is None:
        sys.exit(f"unknown GPU {prof.get('gpu')!r} -- pass --gpu")
    old_hw = spec.hardware(mac_bytes=2)                  # isolated-calibration semantics
    new_hw = spec.hardware(mac_bytes=2, chained=True)    # this anchor's own chained fit
    mem = spec.memory()
    shape = tuple(prof["shape"])
    tokens = prof["tokens"]
    depths = sorted(int(d) for d in prof["results"].keys())

    print(f"profile : {prof.get('gpu')}   shape={shape}   tokens={tokens}")
    print(f"spec    : {spec.name}\n")

    print(f"  {'depth':>5s} {'meas fwd':>9s} {'old bwd':>9s} {'new bwd':>9s} {'meas bwd':>9s} "
          f"{'old disc':>9s} {'new disc':>9s}")
    rows = []
    for depth in depths:
        r = prof["results"][str(depth)]
        m_fwd, m_bwd = r["forward"]["mean_ms"], r["backward"]["mean_ms"]
        _, old_bwd = _predict(shape, tokens, depth, old_hw, mem, with_chain_overhead=False)
        p_fwd, new_bwd = _predict(shape, tokens, depth, new_hw, mem, with_chain_overhead=True)
        old_discount = m_bwd / old_bwd if old_bwd else float("nan")
        new_discount = m_bwd / new_bwd if new_bwd else float("nan")
        rows.append((depth, m_fwd, p_fwd, m_bwd, old_bwd, old_discount, new_bwd, new_discount))
        print(f"  {depth:5d} {m_fwd:9.3f} {old_bwd:9.3f} {new_bwd:9.3f} {m_bwd:9.3f} "
              f"{old_discount:9.3f} {new_discount:9.3f}")

    bwd_xs = [r[0] for r in rows]
    bwd_ys = [r[3] for r in rows]
    A, B, r2 = ols_fit(bwd_xs, bwd_ys)
    print(f"\n  linear fit: measured_backward_ms ~= {A:.4f} * depth + {B:.4f}   (R^2 = {r2:.3f})")
    if r2 > 0.9:
        print(f"  R^2 > 0.9: depth alone explains most of the variance -- a fixed")
        print(f"  per-chain overhead (~{B:.3f} ms) + per-layer marginal cost (~{A:.4f} ms)")
        print(f"  structural model is supported by this data.")
    else:
        print(f"  R^2 <= 0.9: depth alone does NOT cleanly explain the gap even with")
        print(f"  shape and token count held fixed -- do not force a depth-only term.")

    os.makedirs("results/validation", exist_ok=True)
    csv_path = "results/validation/chain_depth_silicon.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gpu", "depth", "measured_fwd_ms", "predicted_fwd_ms",
                    "measured_bwd_ms", "old_predicted_bwd_ms", "old_discount",
                    "new_predicted_bwd_ms", "new_discount"])
        for depth, m_fwd, p_fwd, m_bwd, old_bwd, old_disc, new_bwd, new_disc in rows:
            w.writerow([prof.get("gpu"), depth, f"{m_fwd:.4f}", f"{p_fwd:.4f}",
                        f"{m_bwd:.4f}", f"{old_bwd:.4f}", f"{old_disc:.4f}",
                        f"{new_bwd:.4f}", f"{new_disc:.4f}"])
    print(f"\nwrote {csv_path}")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(bwd_xs, bwd_ys, "o", color="#333", label="measured backward")
    xs_line = [min(bwd_xs), max(bwd_xs)]
    ax.plot(xs_line, [A * x + B for x in xs_line], "--", color="#2ca02c",
           label=f"linear fit (R^2={r2:.2f})")
    ax.plot(bwd_xs, [r[4] for r in rows], "o-", color="#1f77b4", label="old prediction")
    ax.plot(bwd_xs, [r[6] for r in rows], "o-", color="#d62728", label="new prediction (fixed)")
    ax.set_xlabel("chain depth (identical 768x768 layers)")
    ax.set_ylabel("backward latency (ms)")
    title = f"Chain-depth isolation: {prof.get('gpu')}"
    if prof.get("synthetic"):
        title = "SYNTHETIC SAMPLE -- " + title
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = "results/plots/chain_depth_silicon.png"
    fig.savefig(plot_path, dpi=130)
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
