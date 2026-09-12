# Attention

*`simulator/attention.py` — `Attention`, `attention_step()`.*

## What it adds

Everything before this was six linear layers (`workloads.GPT2_BLOCK`) — the
projections and the MLP. This adds the actual attention op: multi-head
`softmax(QK^T / sqrt(d_head)) @ V`. `workloads.GPT2_BLOCK_FULL` is the six
GEMMs plus attention — a complete transformer layer for the first time.

## Why it's modeled differently from a GEMM

A linear layer `Y = X @ W` has one weight, reused across the whole batch —
that's what "weight-stationary" means, and it's why a big batch is cheap per
token. Attention has **no such reuse**: Q, K, V are different for every
`(batch, head)` pair, so it is `B * H` *independent* small GEMMs, each paying
the array's fill/drain overhead fresh. `n_heads=12, d_head=64` for GPT-2
small means `B * H = 96` instances at the default batch of 8.

On top of that, `softmax` is priced as a **memory-only** pass over the
materialized `(B, H, S, S)` score tensor — one extra read + write beyond
what the QK^T write / `(softmax)@V` read already cost. This tensor is
`O(S^2)` in sequence length, unlike everything else in the model, which is
linear in token count.

## What it found

`experiments/attention_scaling.py` sweeps sequence length (batch fixed at
8) and compares the 4 projections + MLP (linear in `S`) against attention
(quadratic in `S`):

| seq | GEMMs | attention | attention share |
|----:|------:|----------:|:---------------:|
| 128 | 3.96 ms | 0.76 ms | 16% |
| 256 | 5.10 ms | 2.52 ms | 33% |
| 512 | 7.36 ms | 9.06 ms | 55% |
| **1024** (GPT-2's own context) | 11.89 ms | 34.23 ms | **74%** |
| 2048 | 20.95 ms | 132.88 ms | 86% |
| 8192 | 75.31 ms | 2077.69 ms | 97% |

**Crossover at 256 → 512 tokens.** By GPT-2's own sequence length, attention
is already three-quarters of the training step — and it only gets worse.
This is the quantitative version of the standard "attention is quadratic"
result, and the real-world reason FlashAttention exists: a fused kernel that
never materializes the full `S x S` score tensor removes almost exactly the
`softmax` traffic term that dominates here.

## Two stated approximations (v1 scope)

1. **Softmax has no compute-cycle model** — only a memory pass (read + write
   the score tensor once each). The `exp`/max-reduction compute cost is
   assumed dominated by that memory pass at the sequence lengths here; for a
   very small, very fast array this could stop being true.
2. **Backward is priced at exactly 2x forward** (compute and bytes) — the
   same forward:backward ratio the rest of the model uses for GEMMs, without
   deriving the softmax-Jacobian chain (`dQ`, `dK`, `dV`) GEMM-by-GEMM the
   way `_backward_seconds` does for a linear layer. Getting this exactly
   right would mean modeling the backward-through-softmax elementwise
   Jacobian, which is a real GEMM-adjacent cost but not one this version
   derives.

Neither approximation is validated against Timeloop or silicon (the silicon
anchor's `profile_step.py` still profiles only the six weight GEMMs — see
docs/silicon_anchor.md).

## Fourth lever: fusion (`attention_step(..., fused=True)`)

The existing lever taxonomy (optimizer precision, recompute, microbatch) is
all about *not* materializing something that doesn't need to hit DRAM.
Attention fusion is the same idea applied to the score tensor: a
flash-attention-style kernel tiles over `K`/`V`, keeps a running (row-max,
row-sum) on chip, and never writes the full `S x S` score tensor to DRAM.
That removes both the softmax read+write pass *and* the QK^T-output /
scoreV-input round trip — what crosses DRAM drops from `O(S^2)` (materializing
scores) to `O(S)` (just `Q, K, V, O`). Compute is unchanged; only traffic and
stored activations drop.

Unlike the other levers, this one is **not SRAM-capacity-gated** — flash
attention's on-chip working set is one tile (`O(block_size · d_head)`),
independent of `S` by construction, so the benefit applies unconditionally
once `fused=True` (see `simulator/attention.py`'s docstring for why that's a
reasonable simplification at the SRAM sizes this codebase considers).

`experiments/attention_fusion.py` sweeps sequence length (batch 8, array
256², 8 MiB, 100 GB/s):

| seq | naive | fused | saved | activations (naive → fused) |
|----:|------:|------:|:-----:|:---:|
| 128 | 0.76 ms | 0.42 ms | 45% | 3.1 MB → 0.10 MB |
| 1024 | 34.23 ms | 2.01 ms | 94% | 201.3 MB → 0.79 MB |
| 8192 | 2077.7 ms | 46.8 ms | **98%** | 12.9 GB → 6.3 MB (2048×) |

The saving *grows* with sequence length — exactly where naive attention
hurts most (`attention_scaling.py`). Backward pays for it the same way
`recompute` does for a GEMM: real FlashAttention recomputes score tiles from
`Q`/`K`/`V` during backward instead of storing them, so `activation_bytes`
drops to a small per-row statistics footprint rather than the full score
tensor — modeled here, but backward is still priced at 2× fused-forward
rather than deriving the recomputation cost GEMM-by-GEMM.

## Silicon anchor result — Tesla T4, fp16

`validate/silicon/attention/profile_attention.py` / `compare_attention.py` measure
naive vs fused (PyTorch's `scaled_dot_product_attention`) attention on a
real GPU. On Turing (T4, sm75 — no FlashAttention CUDA kernel, which needs
Ampere+), SDPA dispatches to PyTorch's memory-efficient attention backend
(`fmha_cutlassF`/`fmha_cutlassB` in `fused_top_ops`) — a genuine fused
kernel that never materializes the score tensor, just not literally
"flash attention." Result (`results/validation/attention_silicon.csv`):

| mode | phase | measured | predicted | ratio | err |
|------|-------|---------:|----------:|:-----:|:---:|
| naive | forward | 6.08 ms | 3.57 ms | 0.59 | 41% |
| naive | backward | 12.66 ms | 7.13 ms | 0.56 | 44% |
| **naive** | **total** | **18.73 ms** | **10.70 ms** | **0.57** | **43%** |
| fused | forward | 1.80 ms | 0.50 ms | 0.28 | 73% |
| fused | backward | 8.19 ms | 1.21 ms | 0.15 | 85% |
| **fused** | **total** | **10.00 ms** | **1.71 ms** | **0.17** | **83%** |

(Fused-backward `predicted` above already includes the recompute-of-`S`/`P`
fix below — the pre-fix number was 0.99 ms, ratio 0.12.)

**Naive is ~1.75× optimistic — in the same direction and rough magnitude as
the uncalibrated GEMM anchor, unsurprising since the calibration constants
(`compute_efficiency`, `util_tiles`) were fit to GEMM shapes, not attention's
`B·H`-independent-instance shape.**

**Fused is ~5.8× optimistic, and that is not the same story.** Measured
fusion saves the training step **46.6%**; the model predicted **84.0%**.
The gap is almost entirely the backward pass: measured fused backward
(8.19 ms) is barely cheaper than measured *naive* backward (12.66 ms) — real
memory-efficient/flash-style backward kernels **recompute** `S`/`P` from
`Q`/`K` before they can compute `dV`/`dP`/`dQ`/`dK`, since `P` was never
stored. That recomputation is now modeled (`attention_step`'s
`bwd_compute = 2×forward_compute + qk_compute_s` when fused — derived from
how the algorithm actually works, not fit to this data point). It moves the
model's own backward:forward ratio from a flat 2.0× to **2.4×** — the right
direction — against a measured **4.5×**. **The fix narrows the gap, it
doesn't close it**, and the remainder is deliberately *not* patched with
another guessed constant from n=1. See Open.

This is the fused-attention analogue of the retracted per-phase dataflow
finding: a stated simplification, checked against real hardware, shown
wrong in a specific and now-partially-corrected way — improved with a real
mechanism where one could be derived, and left honestly open where it
couldn't.

## Open

- **The recompute fix (2.4×) still undershoots the measured ratio (4.5×).**
  Two candidate causes, neither of which is a single fitted constant: (a) no
  cost primitive anywhere in this codebase prices elementwise/reduction
  compute (softmax's backward Jacobian, `dS` from `dP` and `P`, is real
  work with zero charge today — GEMMs and DRAM bytes are the only two things
  priced); (b) the T4 (Turing/sm75) runs PyTorch's memory-efficient
  (not literal Flash) attention backend, which may just be a less efficient
  kernel for this pattern than a newer GPU's. Disentangling (a) from (b)
  needs a second GPU, not more fitting on this one.
  **Tried for (a):** `simulator/compute.py` gained an `elementwise_seconds`
  primitive, fit against the optimizer silicon anchor's isolated Adam data
  (`docs/optimizer_state.md`) — a genuinely different, real elementwise-
  compute cost the model had never priced. Applying that same fitted rate
  to softmax backward's `(B,H,S,S)` element count predicted a ~22ms
  overhead by itself at GPT-2's shapes, dwarfing the whole measured fused
  step: the optimizer anchor's fit covers 0.5–2.4M-element tensors,
  attention's score tensor is 100M+, 40–200x outside that range. Rejected,
  not adopted (`simulator/attention.py`'s inline note) — this gap needs its
  own attention-scale calibration, which needs new data, not a borrowed
  constant.
- **Naive-mode calibration is GEMM-fit, not attention-fit** — the
  `compute_efficiency`/`util_tiles` constants (`docs/silicon_anchor.md`)
  were never tuned against attention's very different shape (`B·H`
  independent tiny-`M`/`N`-huge-`K` instances); a joint calibration across
  both anchors would likely help both. Compounding this: those constants
  were fit against `profile_step.py`'s *isolated* per-GEMM timing sum, which
  `validate/silicon/chain_grid/profile_chain_grid.py` shows overstates a real chained
  step's backward cost by ~2x (`docs/recomputation.md`) — attention's own
  measured GEMM instances run as one real fused kernel launch per `B*H`
  batch, closer to the chained regime than the isolated one, so this
  mismatch likely affects naive attention's calibration too, not just the
  linear-layer anchors.
- Multi-query / grouped-query attention (fewer K/V heads than Q heads) —
  cuts the `B * H` instance count for K/V, a real and common optimization.
- Fold fusion into `simulator/autotune.py`'s lever search (currently GEMM
  levers only).
