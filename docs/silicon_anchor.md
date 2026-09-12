# Silicon anchor

*`validate/silicon/` — one directory per anchor (`gemm/`, `attention/`,
`recompute/`, `gradaccum/`, `optimizer/`, `chain_grid/`, `chain_depth/`,
`chain_generalize/`, `gemm_size/`), plus a shared `gpu_specs.py`. See
`validate/silicon/README.md` for the full anchor list, the current
GPU-mapping table, and how to run each one — this doc is the detailed
narrative for the flagship (GEMM) anchor and the isolated-vs-chained
investigation that followed it.*

## Why

Every other validation in this repo compares accel-sim to **Timeloop**, which
is itself an analytical model. Two models agreeing tells you they share
assumptions, not that either is right. The silicon anchors put predictions
next to a **wall-clock measurement from a real GPU** and report the error —
whatever it is.

## What the flagship (GEMM) anchor measures

The six weight GEMMs of a GPT-2-small block (`q/k/v/attn_out` 768→768,
`mlp_up` 768→3072, `mlp_down` 3072→768), M = 8192 tokens, fp16 (bf16 was
tried first and discarded — see below). For each layer, in isolation:

- **forward**  `Y = X·W`
- **backward** `loss.backward()` → dgrad + wgrad
- **optimizer** one `torch.optim.Adam` step (fp32 m, v state)

timed with CUDA events, 50 iterations after 15 warmup. Nothing else — no
attention score matmuls, no LayerNorm / GELU / dropout / residual. This is
`simulator/`'s exact v1 scope, so the anchor isolates *"is the GEMM + traffic
+ optimizer arithmetic right"* from *"does accel-sim model a whole
transformer"* (it doesn't — that's separate work, see `docs/attention.md`).

## Running it

```bash
python validate/silicon/gemm/profile_step.py --out silicon_profile.json --dtype fp16   # GPU box, Colab T4 ok
python validate/silicon/gemm/compare_silicon.py silicon_profile.json                   # laptop, no GPU needed
```

Outputs: stdout table + `results/validation/silicon.csv` +
`results/plots/silicon_anchor.png`. `sample_profile.json` is a **synthetic**
file (flagged `"synthetic": true`) so the compare path runs with no
hardware; its numbers are invented.

## First run, raw model: 2–5× optimistic

Every known idealisation makes the *raw* model (ideal-array defaults)
**under**-predict real latency: no launch overhead, no epilogue cost,
`max(compute, memory)` assumes perfect overlap, Adam as a pure stream, a GPU
is not a systolic array. The first real Tesla T4 run confirmed this
directly — every phase predicted low (forward ~0.45×, backward ~0.21×,
optimizer ~0.42× of measured) — and the GPU-realism knobs below are the
response. (Before that: the first bf16 attempt on the T4 was discarded —
Turing has no bf16 tensor cores, so torch fell back to a ~20× slower SIMT
GEMM; `compare_silicon.py` detects that pattern now, and every anchor in
this repo uses fp16 on T4/V100 as a result.)

## Model changes made in response

All **default-off** — `Hardware()` / `Memory()` with no args reproduce the
ideal array exactly, so §1–3 of `RESULTS.md` and the Timeloop cross-check are
unchanged. `gpu_specs.py` turns them on, calibrated to this one T4:

| knob | default / GPU | mechanism |
|------|:---:|-----------|
| `Hardware.compute_efficiency` | 1.0 / **0.55** | sustained ÷ datasheet GEMM rate (compute-side `dram_efficiency`) — applied in `matmul_seconds` |
| `Hardware.util_tiles` | 1 / **128** | `_occupancy = min(1, output_tiles / util_tiles)` — a GEMM with too few `rows×cols` output tiles under-fills the machine (the skinny-`wgrad` penalty) |
| `Memory.library_gemm` | off / **on** | when `S/4K < min(M,N)`, a fixed GEMM kernel re-sweeps the inputs `≈ 2·M·N·K / (S/4K)` instead of retiling `K` to the Hong-Kung optimum |
| `Hardware.kernel_launch_s` | 0 / **8 µs** | `+ 12 × launch` on the Adam step |

Plus a **harness fix**: the profiled inputs did not `require_grad`, so
`.backward()` had been timing `wgrad` only, not `dgrad + wgrad`.

## Re-score — Tesla T4, fp16, clean profile

| phase | measured | predicted | ratio | abs %err |
|-------|---------:|----------:|:-----:|:--------:|
| forward   |  4.12 ms |  4.11 ms | 1.00 |  0.3 % |
| backward  | 20.62 ms | 17.79 ms | 0.86 | 13.7 % |
| optimizer |  1.78 ms |  1.28 ms | 0.72 | 28.0 % |
| **step**  | **26.52 ms** | **23.19 ms** | **0.87** | **12.6 %** |

Phase MAPE **14 %**, step within **13 %** — in the range of calibrated
inference models (LLMCompass 4–11 %), a fair bar now that this is *also*
calibrated.

**After the optimizer elementwise-compute fix** (`docs/optimizer_state.md`)
this phase table changed: optimizer error got *worse* in isolation (28% →
59%, now over-predicting instead of under) because this anchor's optimizer
timing bracket is Adam interleaved with forward/backward in the same loop —
a different measurement context than the isolated Adam-only anchor the fix
was fit against, so it doesn't transfer perfectly. But the **step-level
error improved** (12.6% → 6.7%): the now-larger optimizer over-prediction
partially cancels backward's existing under-prediction in the sum. Reported
plainly rather than tuned away — the aggregate improving for a reason
that isn't "the model got more correct everywhere" is exactly the kind of
result worth flagging, not hiding.

**Caveats, plainly:**

- **3 fitted constants, 18 points, one GPU.** `util_tiles = 128` and the
  `4·K·T` cache fraction are knees fit to this T4, not first-principles.
- **Per-layer error is ±1.6×.** `mlp_up` → 0.9×, its mirror-shape `mlp_down`
  → 1.5×: the model is symmetric under M↔N, cuBLAS is not (different kernel
  per shape). Not fixable without a kernel database. `proj` GEMMs sit at
  ~0.5× — small-GEMM launch / wave-quantisation still under-counted.
- The **aggregate flatters the model** — per-layer errors partly cancel.

## Confidence, not just point estimates

`compare_silicon.py` now reports a 95% CI on each measured phase (from the
per-layer `std_ms` / `sqrt(iters)`, propagated across layers) and flags any
per-layer cell whose run-to-run std exceeds 15% of its mean. On the T4 run,
`q_proj/optimizer` (109% relative std) and three `v_proj` cells (22–66%) are
flagged — almost certainly warmup/cold-start artifacts on the first couple
of iterations of the very first layers timed, not a hardware effect. The CI
bounds *sampling noise* only (all a few tenths of a ms); it says nothing
about the *systematic* error this whole document is about, which is far
larger (12.6% on the step, ±1.6× per layer).

## Open

- **Second anchor** on an A100 / L4 — do `compute_efficiency = 0.55`,
  `util_tiles = 128` transfer, or are they T4-specific? That is the real
  test of whether this calibration means anything.
- `proj`-layer under-prediction (~0.5×) — a small-GEMM fixed-overhead term
  would help but adds a 4th knob.
- **This whole anchor's ground truth is confirmed to be the wrong target,
  and the fix found so far does not generalize.** `_bench_layer` times
  each of the 6 GEMMs in total isolation and sums the results to
  approximate a "step" — not one real chained forward+backward through a
  multi-layer autograd graph, which is how a model actually trains.
  `validate/silicon/chain_grid/profile_chain_grid.py` ran 4 real chained configs on a
  T4 and confirmed the direction: every config's real chained backward is
  1.4-3.5x **faster** than what this anchor's calibration predicts. A
  2-parameter refit of `compute_efficiency`/`util_tiles` against those 4
  configs didn't generalize (depth, shape, and token count all vary at
  once), so `validate/silicon/chain_depth/profile_chain_depth.py` isolated depth alone
  (fixed 768×768 shape, fixed 8192 tokens, depths 1/2/4/6/8). That
  succeeded cleanly: `measured_backward_ms ≈ 0.49·depth + 1.36` (R² =
  0.974). The driver turned out to be `_occupancy`'s `util_tiles` penalty
  on wgrad's skinny output shape, not `compute_efficiency` (which would
  need to exceed 1.0 to fit — physically meaningless); relieving it
  (`Hardware.backward_util_tiles`, set to 1 for T4) plus a fixed one-time
  `chain_overhead_fwd_s`/`chain_overhead_bwd_s` brings this one shape's
  error down to 6-19% across all 5 depths.

  **But this calibration is specific to (768×768, 8192 tokens) and does
  not transfer.** Tried against the other 3 chain-grid configs (different
  shapes/depths/token counts): 2 of 4 got *worse* (`wide_shallow`
  forward 42%→65%, `small_batch` forward 34%→187% — the fixed overhead
  terms, in absolute ms, overwhelm a config whose whole measured time is
  under 1ms). Tried against the real recompute/gradaccum anchors' own
  mixed-shape chain (2× 768×768 + 768×3072 + 3072×768): backward error
  barely moved (106.1% → 105.2%). Left **opt-in** (`GPUSpec.hardware(...,
  chained=True)`), used only by `compare_chain_depth.py` itself, not
  applied by default anywhere else — adopting it broadly would misrepresent
  a narrow, non-generalizing fit as a general one. `docs/recomputation.md`
  has the recompute-anchor-specific numbers this was first noticed from.

  **Two follow-up investigations, both ending in a real but negative
  result — reported in full since the reasoning matters more than the
  outcome:**

  1. *Does the fixed-overhead effect exist independent of occupancy, and
     does it scale with token count?* `profile_chain_generalize.py` ran a
     depth sweep + token-count sweep on a 2048×2048 shape verified
     occupancy-safe throughout. The depth sweep gave a **perfect** fit
     (R²=1.000, `measured_backward_ms ≈ 3.64·depth + 2.62`) — the
     structural form holds again — but the constants are shape-size-
     specific (nothing like 768×768's 0.49/1.36), and a quick check found
     a single unchained GEMM (depth=1) landing within 4.4% of the
     `compute_efficiency=1.0` ideal, suggesting efficiency itself rises
     with GEMM size — a real GPU effect never modeled, but entangled with
     the chain-depth effect, not isolated by this design.

  2. *Isolate GEMM size alone* (`profile_gemm_size.py`, depth=1 fixed — no
     chaining at all, tokens=8192 fixed, only a square layer's size
     varies 256→4096, crossing the occupancy threshold). Backward's
     discount plateaued cleanly at 1536-4096 (implied efficiency 0.90-1.03,
     essentially ce=1.0) — looked like a clean, adoptable fix. **It was
     reverted after checking the actual compute/memory bound**: those
     GEMMs are **memory-bound** under `_backward_seconds`'s
     `max(compute, memory)` (a 2048×2048 fp16 weight matrix is 8MB,
     bigger than this GPU's 4MB L2, triggering the traffic-penalty
     branch) — so `compute_efficiency` never reaches the final result,
     and the clean-looking "efficiency plateau" was actually invisible
     noise. Confirmed by the arithmetic: matching the measured time as a
     memory-bound quantity would need ~451 GB/s on a GPU with a 320 GB/s
     physical maximum — impossible. **The real gap for large, well-tiled
     backward GEMMs is in `gemm_traffic`'s unvalidated penalty-term
     heuristic** (Limitation #1), not compute efficiency — a different
     model component than anything touched in this investigation.
     Forward showed the same compute/memory-bound switching across its
     own sweep (compute-bound at 512-1536, memory-bound at 256/2048+), so
     no forward fix was adopted from this data either, for the same
     reason.

  **Where this line of investigation stops, and why:** five real
  experiments (chain-grid, chain-depth, chain-generalize, and now
  GEMM-size) surfaced four entangled effects (shape-transfer, chain-depth
  overlap, GEMM-size efficiency, and now the DRAM-traffic-penalty
  heuristic) without a generalizing fix beyond the one narrow,
  correctly-scoped 768×768 result already adopted. Two proposed fixes
  were caught and reverted before shipping specifically because they were
  re-verified against their own compute/memory bound rather than trusted
  on discount numbers alone (`bwd_matmul`'s inline comment in
  `simulator/dataflow.py` and the constant's own comment in `gpu_specs.py`
  record both). Closing this for real now points at a different anchor
  entirely — a dedicated DRAM-traffic-penalty validation (does
  `gemm_traffic`'s `library_gemm` re-streaming formula match real cuBLAS
  behavior for large operands?) — not another compute-side sweep.
