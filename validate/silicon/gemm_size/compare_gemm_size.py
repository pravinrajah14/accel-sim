"""Isolate GEMM size's effect on compute efficiency -- fixed depth=1
(no chaining), fixed M=8192. See `profile_gemm_size.py`'s docstring.

Runs locally, no GPU.

    python validate/silicon/compare_gemm_size.py gemm_size_profile.json
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

from simulator.compute import _occupancy  # noqa: E402
from simulator.dataflow import Layer, training_step  # noqa: E402
from validate.silicon import gpu_specs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--gpu", default=None)
    args = ap.parse_args()

    with open(args.profile) as f:
        prof = json.load(f)

    if prof.get("synthetic"):
        print("\n  !!  SYNTHETIC SAMPLE -- numbers below are illustrative only.")
        print("      Replace with a real profile_gemm_size.py run on a GPU.\n")

    spec = gpu_specs.lookup(args.gpu or prof.get("gpu"))
    if spec is None:
        sys.exit(f"unknown GPU {prof.get('gpu')!r} -- pass --gpu")
    hw = spec.hardware(mac_bytes=2)     # plain isolated-calibration semantics, depth=1
    mem = spec.memory()
    tokens = prof["tokens"]
    sizes = sorted(int(s) for s in prof["results"].keys())

    print(f"profile : {prof.get('gpu')}   tokens={tokens}   depth=1 (no chaining)")
    print(f"spec    : {spec.name}\n")

    print(f"  {'size':>6s} {'wgrad occ':>9s} {'meas fwd':>9s} {'old fwd':>9s} "
          f"{'fwd disc':>9s}  {'meas bwd':>9s} {'old bwd':>9s} {'bwd disc':>9s}")
    rows = []
    for size in sizes:
        r = prof["results"][str(size)]
        m_fwd, m_bwd = r["forward"]["mean_ms"], r["backward"]["mean_ms"]
        layer = Layer("l", M=tokens, K=size, N=size)
        p = training_step(layer, hw, mem)
        old_fwd, old_bwd = p.forward_s * 1e3, p.backward_s * 1e3
        fwd_disc = m_fwd / old_fwd if old_fwd else float("nan")
        bwd_disc = m_bwd / old_bwd if old_bwd else float("nan")
        wgrad_occ = _occupancy(size, size, hw)
        rows.append((size, wgrad_occ, m_fwd, old_fwd, fwd_disc, m_bwd, old_bwd, bwd_disc))
        print(f"  {size:6d} {wgrad_occ:9.3f} {m_fwd:9.3f} {old_fwd:9.3f} {fwd_disc:9.3f}  "
              f"{m_bwd:9.3f} {old_bwd:9.3f} {bwd_disc:9.3f}")

    print(f"\n  discount = measured/old_predicted. <1 means the current model over-predicts")
    print(f"  (needs a HIGHER effective rate to match); >1 means it under-predicts.")
    print(f"  If discount rises smoothly with size even where wgrad occupancy is already 1.0,")
    print(f"  that's a real size-dependent efficiency effect distinct from occupancy.")

    os.makedirs("results/validation", exist_ok=True)
    csv_path = "results/validation/gemm_size_silicon.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gpu", "size", "wgrad_occupancy", "measured_fwd_ms", "old_pred_fwd_ms",
                    "fwd_discount", "measured_bwd_ms", "old_pred_bwd_ms", "bwd_discount"])
        for size, occ, m_fwd, old_fwd, fwd_disc, m_bwd, old_bwd, bwd_disc in rows:
            w.writerow([prof.get("gpu"), size, f"{occ:.4f}", f"{m_fwd:.4f}", f"{old_fwd:.4f}",
                        f"{fwd_disc:.4f}", f"{m_bwd:.4f}", f"{old_bwd:.4f}", f"{bwd_disc:.4f}"])
    print(f"\nwrote {csv_path}")

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(sizes, [r[4] for r in rows], "o-", color="#1f77b4", label="forward discount")
    ax.plot(sizes, [r[7] for r in rows], "o-", color="#d62728", label="backward discount")
    ax.plot(sizes, [r[1] for r in rows], "o--", color="#999", label="wgrad occupancy (old model)")
    ax.axhline(1.0, color="#333", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("square GEMM size (K=N)")
    ax.set_ylabel("ratio")
    title = f"GEMM-size isolation: {prof.get('gpu')}"
    if prof.get("synthetic"):
        title = "SYNTHETIC SAMPLE -- " + title
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    plot_path = "results/plots/gemm_size_silicon.png"
    fig.savefig(plot_path, dpi=130)
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
