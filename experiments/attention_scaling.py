"""When does attention overtake the FFN GEMMs?

The four projection GEMMs + MLP scale linearly in token count (M = batch x
seq). Attention's QK^T and (softmax)@V are B*H independent (S,S)-shaped
GEMMs with no cross-instance reuse, plus a softmax pass that materializes
the full (B,H,S,S) score tensor -- both O(S^2) in sequence length. So at
short sequences the GEMMs dominate; past some crossover, attention does.
This is the quantitative version of the standard "attention is quadratic"
result -- and the real-world motivation for fused (flash-attention-style)
kernels that never materialize the S x S score tensor (see docs/attention.md
"Open" section -- that fusion is not modeled here).
"""

from _common import Hardware, Memory, plt, PLOT_DIR, model_step
import os

from simulator.attention import Attention
from simulator.dataflow import Layer, ADAM_FP32
from simulator.workloads import BATCH, D_MODEL, D_FF, N_HEADS, D_HEAD

HW = Hardware(rows=256, cols=256, clock_hz=1e9)
MEM = Memory(sram_bytes=8 << 20, dram_bandwidth_bytes_s=100e9)
SEQS = [128, 256, 512, 1024, 2048, 4096, 8192]


def block_at(seq):
    m = BATCH * seq
    gemms = [
        Layer("q_proj", M=m, K=D_MODEL, N=D_MODEL),
        Layer("k_proj", M=m, K=D_MODEL, N=D_MODEL),
        Layer("v_proj", M=m, K=D_MODEL, N=D_MODEL),
        Layer("attn_out", M=m, K=D_MODEL, N=D_MODEL),
        Layer("mlp_up", M=m, K=D_MODEL, N=D_FF),
        Layer("mlp_down", M=m, K=D_FF, N=D_MODEL),
    ]
    attn = Attention("attention", batch=BATCH, seq=seq, n_heads=N_HEADS, d_head=D_HEAD)
    return gemms, attn


xs, gemm_ms, attn_ms = [], [], []
for seq in SEQS:
    gemms, attn = block_at(seq)
    g, _ = model_step(gemms, HW, MEM, ADAM_FP32)
    a, _ = model_step([attn], HW, MEM, ADAM_FP32)
    xs.append(seq)
    gemm_ms.append(g * 1e3)
    attn_ms.append(a * 1e3)
    frac = a / (a + g) * 100
    print(f"seq {seq:5d}   gemms {g*1e3:8.3f} ms   attention {a*1e3:9.3f} ms   "
          f"attention share {frac:5.1f}%")

crossover = None
for i in range(1, len(xs)):
    if (attn_ms[i - 1] < gemm_ms[i - 1]) != (attn_ms[i] < gemm_ms[i]):
        crossover = (xs[i - 1], xs[i])
print(f"\ncrossover (attention overtakes the GEMMs): "
      f"{crossover[0]} -> {crossover[1]} tokens/sequence" if crossover
      else "\nno crossover in the swept range")

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(xs, gemm_ms, "o-", color="#1f77b4", label="4 projections + MLP (linear in seq)")
ax.plot(xs, attn_ms, "o-", color="#d62728", label="attention: QK^T + softmax + (softmax)V (quadratic)")
if crossover:
    xmid = (crossover[0] * crossover[1]) ** 0.5
    ax.axvline(xmid, ls="--", color="gray")
    ax.annotate(f"crossover\n{crossover[0]} -> {crossover[1]}", xy=(xmid, max(gemm_ms + attn_ms)),
                xytext=(0, -10), textcoords="offset points", ha="center", va="top",
                fontsize=9, color="gray")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("sequence length (tokens), batch=8 fixed")
ax.set_ylabel("training step latency (ms, log scale)")
ax.set_title("GEMMs (linear) vs attention (quadratic) vs sequence length")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9)
fig.tight_layout()
out = os.path.join(PLOT_DIR, "attention_scaling.png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")
