"""Activation recomputation: a win only while memory-bound.

Recomputation regenerates the input activation by re-running the forward
GEMM in the backward pass instead of reading it from DRAM -- trading
traffic for compute. Sweeps DRAM bandwidth (array 256x256, 8 MiB SRAM) and
plots the training step with recompute on vs off.

The two curves cross at the roofline knee: left of it (memory-bound)
recompute is a real win; right of it (compute-bound) it is dead weight.
"""

import os

from _common import Hardware, Memory, plt, PLOT_DIR, GPT2_BLOCK, model_step

HW = Hardware(rows=256, cols=256, clock_hz=1e9)
SRAM = 8 << 20
BW_GBS = [25, 50, 100, 150, 200, 300, 400, 600, 800, 1600]

xs, off, on = [], [], []
for bw in BW_GBS:
    mem = Memory(sram_bytes=SRAM, dram_bandwidth_bytes_s=bw * 1e9)
    a, _ = model_step(GPT2_BLOCK, HW, mem, recompute=False)
    r, _ = model_step(GPT2_BLOCK, HW, mem, recompute=True)
    xs.append(bw)
    off.append(a * 1e3)
    on.append(r * 1e3)
    print(f"{bw:5d} GB/s   no-recompute {a*1e3:7.2f} ms   recompute {r*1e3:7.2f} ms"
          f"   {(a - r) / a * 100:+5.1f}%")

# crossover: first bw where recompute stops helping
cross = next((f"{xs[i-1]}-{xs[i]}" for i in range(1, len(xs))
             if (off[i] - on[i]) <= 0 < (off[i - 1] - on[i - 1])), None)
print(f"\nrecompute crossover: {cross} GB/s")

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(xs, off, "o-", color="#1f77b4", label="store activations")
ax.plot(xs, on, "o-", color="#d62728", label="recompute activations")
if cross:
    lo, hi = (int(v) for v in cross.split("-"))
    xm = (lo * hi) ** 0.5
    ax.axvline(xm, ls="--", color="gray")
    ax.annotate("roofline knee:\nrecompute stops helping", xy=(xm, max(off)),
                xytext=(6, -6), textcoords="offset points", fontsize=8,
                color="gray", va="top")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("DRAM bandwidth (GB/s)")
ax.set_ylabel("training step latency (ms)")
ax.set_title("Activation recomputation vs DRAM bandwidth (array 256×256, 8 MiB)")
ax.grid(alpha=0.3, which="both")
ax.legend()
fig.tight_layout()
out = os.path.join(PLOT_DIR, "recomputation.png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")
