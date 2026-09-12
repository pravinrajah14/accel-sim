"""Shared helpers for the parameter sweeps."""

import os
import sys

# allow `python experiments/sweep_*.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simulator.compute import Hardware  # noqa: E402
from simulator.memory import Memory  # noqa: E402
from simulator.dataflow import model_step  # noqa: E402
from simulator.workloads import GPT2_BLOCK  # noqa: E402

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def step_ms(hw: Hardware, mem: Memory, layers=GPT2_BLOCK) -> float:
    total, _ = model_step(layers, hw, mem)
    return total * 1e3


def dominant_bound(hw: Hardware, mem: Memory, layers=GPT2_BLOCK) -> str:
    """Bound (compute/memory) of the phase that contributes the most time."""
    _, parts = model_step(layers, hw, mem)
    worst = max(parts, key=lambda p: p.backward_s)
    return worst.backward_bound


def find_inflection(xs, bounds):
    """First x where the dominant bound flips.  Returns (x_lo, x_hi) or None."""
    for i in range(1, len(xs)):
        if bounds[i] != bounds[i - 1]:
            return xs[i - 1], xs[i]
    return None


def plot_sweep(xs, ys, bounds, *, xlabel, title, fname, xlog=False):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ys, "o-", color="#333", zorder=3)

    # colour the markers by bound
    for x, y, b in zip(xs, ys, bounds):
        ax.plot(x, y, "o", ms=8, zorder=4,
                color="#1f77b4" if b == "compute" else "#d62728")

    infl = find_inflection(xs, bounds)
    if infl:
        xmid = (infl[0] * infl[1]) ** 0.5 if xlog else sum(infl) / 2
        ax.axvline(xmid, ls="--", color="gray")
        ax.annotate(f"bottleneck shift\n{infl[0]:g} -> {infl[1]:g}",
                    xy=(xmid, max(ys)), xytext=(0, -10),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=9, color="gray")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("training step latency (ms)")
    ax.set_title(title)
    if xlog:
        ax.set_xscale("log", base=2)
    ax.grid(alpha=0.3)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", ls="", color="#1f77b4", label="compute-bound"),
        Line2D([0], [0], marker="o", ls="", color="#d62728", label="memory-bound"),
    ])

    fig.tight_layout()
    out = os.path.join(PLOT_DIR, fname)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    if infl:
        print(f"bottleneck shift between {xlabel} = {infl[0]:g} and {infl[1]:g}")
    else:
        print(f"no bottleneck shift over the swept range ({xlabel})")
