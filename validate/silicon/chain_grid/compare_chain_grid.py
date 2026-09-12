"""Refit compute_efficiency/util_tiles against real CHAINED training steps,
instead of the isolated per-GEMM sum profile_step.py's anchor was fit
against -- see profile_chain_grid.py's docstring for why that matters.

Runs locally, no GPU.

    python validate/silicon/compare_chain_grid.py chain_grid_profile.json
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

from simulator.compute import Hardware  # noqa: E402
from simulator.dataflow import Layer, training_step  # noqa: E402
from validate.silicon import gpu_specs  # noqa: E402


def _predict_chain(dims, tokens, hw, mem):
    """Sum training_step's forward_s/backward_s over each layer in the chain
    -- the same per-layer-sum approach compare_recompute.py already uses,
    just for arbitrary chains instead of one fixed shape set."""
    fwd_s = bwd_s = 0.0
    for k, n in dims:
        layer = Layer("l", M=tokens, K=k, N=n)
        p = training_step(layer, hw, mem)
        fwd_s += p.forward_s
        bwd_s += p.backward_s
    return fwd_s * 1e3, bwd_s * 1e3   # -> ms


def _hardware_with(spec, compute_efficiency, util_tiles, mac_bytes=2):
    """Same array-shape derivation as GPUSpec.hardware(), with candidate
    compute_efficiency/util_tiles substituted in (everything else fixed)."""
    import math
    peak_macs = spec.peak_tflops * 1e12 / 2.0
    clock_hz = spec.clock_ghz * 1e9
    side = max(1, round(math.sqrt(peak_macs / clock_hz)))
    return Hardware(rows=side, cols=side, clock_hz=clock_hz, mac_bytes=mac_bytes,
                    compute_efficiency=compute_efficiency,
                    kernel_launch_s=gpu_specs._KERNEL_LAUNCH_S,
                    util_tiles=util_tiles)


def _total_sq_rel_err(configs, spec, mem, compute_efficiency, util_tiles):
    hw = _hardware_with(spec, compute_efficiency, util_tiles)
    err = 0.0
    for cfg in configs.values():
        m_fwd = cfg["forward"]["mean_ms"]
        m_bwd = cfg["backward"]["mean_ms"]
        p_fwd, p_bwd = _predict_chain(cfg["dims"], cfg["tokens"], hw, mem)
        err += ((p_fwd - m_fwd) / m_fwd) ** 2
        err += ((p_bwd - m_bwd) / m_bwd) ** 2
    return err


def refit(configs, spec, mem):
    """Small grid search over (compute_efficiency, util_tiles) minimizing
    summed squared relative error across every config's forward+backward.
    Only 2 unknowns and ~4 configs -- a full optimizer is unnecessary."""
    ce_candidates = [round(0.20 + 0.01 * i, 2) for i in range(71)]      # 0.20..0.90
    ut_candidates = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
    best = (None, None, float("inf"))
    for ce in ce_candidates:
        for ut in ut_candidates:
            e = _total_sq_rel_err(configs, spec, mem, ce, ut)
            if e < best[2]:
                best = (ce, ut, e)
    return best[0], best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--gpu", default=None)
    args = ap.parse_args()

    with open(args.profile) as f:
        prof = json.load(f)

    spec = gpu_specs.lookup(args.gpu or prof.get("gpu"))
    if spec is None:
        sys.exit(f"unknown GPU {prof.get('gpu')!r} -- pass --gpu")
    mem = spec.memory()
    configs = prof["configs"]

    print(f"profile : {prof.get('gpu')}   dtype={prof.get('dtype')}")
    print(f"spec    : {spec.name}  (old constants: "
          f"compute_efficiency={gpu_specs._COMPUTE_EFFICIENCY}  "
          f"util_tiles={gpu_specs._UTIL_TILES})")

    old_hw = _hardware_with(spec, gpu_specs._COMPUTE_EFFICIENCY, gpu_specs._UTIL_TILES)

    print(f"\n  {'config':16s} {'phase':9s} {'measured':>10s} {'old pred':>10s} "
          f"{'old %err':>9s}")
    old_rows = []
    for name, cfg in configs.items():
        p_fwd, p_bwd = _predict_chain(cfg["dims"], cfg["tokens"], old_hw, mem)
        for ph, m, p in (("forward", cfg["forward"]["mean_ms"], p_fwd),
                        ("backward", cfg["backward"]["mean_ms"], p_bwd)):
            err = abs(p - m) / m * 100
            old_rows.append((name, ph, m, p, err))
            print(f"  {name:16s} {ph:9s} {m:10.3f} {p:10.3f} {err:8.1f}%")

    new_ce, new_ut = refit(configs, spec, mem)
    new_hw = _hardware_with(spec, new_ce, new_ut)

    print(f"\n  refit   : compute_efficiency {gpu_specs._COMPUTE_EFFICIENCY} -> {new_ce}   "
          f"util_tiles {gpu_specs._UTIL_TILES} -> {new_ut}")
    print(f"\n  {'config':16s} {'phase':9s} {'measured':>10s} {'new pred':>10s} "
          f"{'new %err':>9s}  {'old %err':>9s}")
    new_rows = []
    idx = 0
    for name, cfg in configs.items():
        p_fwd, p_bwd = _predict_chain(cfg["dims"], cfg["tokens"], new_hw, mem)
        for ph, m, p in (("forward", cfg["forward"]["mean_ms"], p_fwd),
                        ("backward", cfg["backward"]["mean_ms"], p_bwd)):
            err = abs(p - m) / m * 100
            old_err = old_rows[idx][4]
            new_rows.append((name, ph, m, p, err, old_err))
            print(f"  {name:16s} {ph:9s} {m:10.3f} {p:10.3f} {err:8.1f}%  {old_err:8.1f}%")
            idx += 1

    os.makedirs("results/validation", exist_ok=True)
    with open("results/validation/chain_grid_silicon.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gpu", "config", "phase", "measured_ms", "old_pred_ms", "old_pct_err",
                    "new_pred_ms", "new_pct_err"])
        for (name, ph, m, old_p, old_err), (_, _, _, new_p, new_err, _) in zip(
                old_rows, new_rows):
            w.writerow([prof.get("gpu"), name, ph, f"{m:.4f}", f"{old_p:.4f}",
                        f"{old_err:.2f}", f"{new_p:.4f}", f"{new_err:.2f}"])
    print("\nwrote results/validation/chain_grid_silicon.csv")

    names = list(configs.keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, ph in zip(axes, ("forward", "backward")):
        m = [configs[n][ph]["mean_ms"] for n in names]
        old_p = [_predict_chain(configs[n]["dims"], configs[n]["tokens"], old_hw, mem)[0 if ph == "forward" else 1] for n in names]
        new_p = [_predict_chain(configs[n]["dims"], configs[n]["tokens"], new_hw, mem)[0 if ph == "forward" else 1] for n in names]
        x = range(len(names))
        w = 0.27
        ax.bar([i - w for i in x], m, width=w, label="measured")
        ax.bar(x, old_p, width=w, label="old constants")
        ax.bar([i + w for i in x], new_p, width=w, label="refit constants")
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel("ms")
        ax.set_title(ph)
        ax.legend(fontsize=8)
    fig.suptitle(f"{prof.get('gpu')} -- chained training step, old vs refit constants")
    fig.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    fig.savefig("results/plots/chain_grid_silicon.png", dpi=140)
    print("wrote results/plots/chain_grid_silicon.png")


if __name__ == "__main__":
    main()
