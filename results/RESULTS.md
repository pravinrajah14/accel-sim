# Canonical results

Single source of truth for the numbers in the README and docs. Regenerate
with `./run_all.sh`. Current as of the **traffic-model rewrite** (see history
+ the retraction note at the bottom).

## Baseline training step

GPT-2-small block (6 GEMMs, batch 8 × seq 1024 = 8192 tokens), array
128×128 @ 1 GHz, 100 GB/s DRAM, naive backward, fp32 streamed Adam:

| SRAM | forward | backward | optimizer | **step** |
|-----:|--------:|---------:|----------:|---------:|
| 2 MiB | 4.73 | 9.96 | 2.26 | **16.96 ms** (memory-bound) |
| 8 MiB | 3.21 | 8.11 | 2.26 | **13.58 ms** (compute-bound) |

## 1 — Bottleneck shift (`experiments/sweep_*.py`)

Each knob swept, other two fixed. The array and bandwidth sweeps hold SRAM
at 8 MiB so they sit in the penalty-free (Timeloop-validated-exact) traffic
regime; the SRAM sweep's low tail uses the heuristic penalty term.

| swept knob | fixed | compute→memory crossover |
|------------|-------|:------------------------:|
| array edge | SRAM 8 MiB, BW 100 GB/s | **128 → 192 PEs** |
| DRAM bandwidth | array 128², SRAM 8 MiB | **50 → 100 GB/s** |
| on-chip SRAM | array 128², BW 100 GB/s | **4 → 8 MiB** |

Above each crossover the curve is flat — spending on that knob is wasted.
The *shape* (sharp crossover, flat floor) is robust across model revisions;
the exact crossover point has moved ±1 grid step as the model tightened.

## 2 — Levers

**Optimizer state** (`experiments/optimizer_cost.py`, array 256², 100 GB/s):
fp32 streamed = 2.27 ms; resident reaches a 0.76 ms floor (master weights
only) at 24 MiB for fp32, 16 MiB for bf16, **8 MiB for fp8**.

**Activation recomputation** (`experiments/recomputation.py`, array 256²,
8 MiB, sweep bandwidth): **+13 %** in the memory-bound plateau
(≤ ~150 GB/s), crossover **200–300 GB/s**, **−31 %** deep in the
compute-bound region. Silicon anchor (`validate/silicon/recompute/profile_recompute.py`
/ `compare_recompute.py`, real `torch.utils.checkpoint`, Tesla T4): forward
transfers from the GEMM anchor's calibration (~13-18% error), backward does
**not** — model over-predicts backward by ~2x, traced to a measurement-
methodology gap (isolated-per-GEMM timing vs a real chained step), the
starting point of the full isolated-vs-chained investigation
(`docs/silicon_anchor.md`). The lever's own delta (stored vs recompute)
still checks out qualitatively: measured +0.2%, predicted +1.5%, both
saying this config is compute-bound (`docs/recomputation.md`).

**Gradient accumulation** (`experiments/microbatching.py`, array 256²,
32 MiB, 100 GB/s): each ×2 accumulation halves stored activations
(113 → 57 → 28 → 14 MB) for **+5 %, +14 %, +33 %** latency respectively.
2–4× is nearly free; beyond that the weight re-reads dominate. Same
memory-vs-cost axis as MONET's activation checkpointing ("4 % → 13 MB"),
better ratio here (no extra compute). Silicon anchor
(`validate/silicon/gradaccum/profile_gradaccum.py` / `compare_gradaccum.py`, real
peak-memory measurement via `torch.cuda.max_memory_allocated`, Tesla T4):
latency shows the same isolated-vs-chained mismatch as the recompute
anchor. **Memory was fixed**: measured peak activation memory was 3-12×
higher than the raw prediction at every fraction; a 2-parameter fit
(`Memory.framework_overhead_bytes` + `framework_overhead_scale`,
`measured_mb ≈ 57.6 + 2.27·predicted_mb`) brings every fraction within 2%
of measured. Not baked into `gpu_specs.py`'s defaults — it's fit to this
GPU and this chain's total weight size, reported per-anchor
(`docs/microbatching.md`).

**Optimizer state precision, isolated** (`validate/silicon/optimizer/profile_optimizer.py`
/ `compare_optimizer.py`, manual Adam update at fp32 vs bf16 state, no
forward/backward, Tesla T4): first run was a **sign flip, not a magnitude
miss** — measured bf16 saving was **−33.3%** (bf16 slower) against a
predicted **+20.2%** (bf16 faster); Turing has no native bf16 arithmetic,
so PyTorch's emulation cost outweighs the smaller-tensor bandwidth saving.
**Fixed**: `simulator/compute.py` gained an `elementwise_seconds` compute
primitive (Adam's real `mul`/`addcmul`/`sqrt`/`div` chain was priced at
zero before) plus a native-dtype-support gate
(`Hardware.native_low_precision_bytes`, `emulation_penalty`), both fit
against this anchor's own per-layer data. Re-scored: fp32 ratio 0.34→0.78,
bf16 ratio 0.20→0.85, predicted bf16 saving −44.9% (right sign, matches
measured −33.3% in direction and order of magnitude). The lever's claim now
correctly depends on hardware: free latency on Ampere+ (native bf16),
a real cost on Turing/Volta (`docs/optimizer_state.md`).

Two further levers — backward-pass reuse and a GEMM-operand fp8/precision
lever — were built, then **removed** (not just relabeled) once neither
could be checked against real hardware; see §10.

## 3 — Autotuner (`simulator/autotune.py`, `experiments/autotune.py`)

Exhaustive search of the 40-config lever space (optimizer × recompute ×
microbatch), **~1 ms** total. Config-dependent picks:

| hardware | autotuned plan | step |
|----------|----------------|-----:|
| 128² / 2 MiB / 50 GB/s | fp8 resident, recompute | 28.4 ms |
| 128² / 8 MiB / 100 GB/s | fp8 resident, store acts | 12.1 ms |
| 256² / 16 MiB / 100 GB/s | bf16 resident, recompute | 8.9 ms |
| 256² / 64 MiB / 400 GB/s | fp32 resident, store acts | 3.0 ms |

Under an activation budget it trades minimally (256²/16 MiB: 64→32 MB budget
costs +0.6 ms). `pareto()` returns the full latency-vs-activation frontier.

**Stacked** — array 256×256, 100 GB/s:

| config | step |
|--------|-----:|
| baseline (fp32 streamed Adam) | 11.9 ms |
| + fp8 resident optimizer state | 10.4 ms |

## 4 — Retracted: per-phase dataflow

An earlier revision claimed output-stationary `wgrad` saves up to 34 %.
**Disproved by the traffic validation** — a well-scheduled weight-stationary
`wgrad` reaches the same minimal DRAM traffic by tiling the contraction.
DRAM traffic is dataflow-independent; the dataflow moves ~1–3 % of cycles.
See `docs/dataflow_choice.md`.

## 5 — Timeloop validation (`validate/`, `results/validation/validation.csv`)

Pinned (size-1) mappings, 1024 tokens (staircase uses 4096):

| family | GEMM / dataflow | n | cycles MAPE | DRAM MAPE |
|--------|-----------------|:-:|:-----------:|:---------:|
| `ws-fwd` | forward, weight-stationary | 6 | 1.2 % | **0.0 %** |
| `os-wgrad` | `wgrad`, output-stationary | 6 | 0.2 % | **0.0 %** |
| `wsw-stair` | `wgrad`, weight-stationary, glb sweep | 3 | 0.7 % | **0.0 %** |

Timeloop's cycle count and DRAM traffic both equal the model's compulsory
figures exactly; the model is high on cycles only by its fixed overhead
constant. **Both the cycle model and the compulsory-traffic term are
validated.** The traffic *penalty* term (smallest tensor spills) is a
first-order heuristic — not directly validated, though the staircase confirms
penalty = 0 in the regime where it should be.

## 6 — Attention (`simulator/attention.py`, `experiments/attention_scaling.py`)

Extends the model from 6 linear layers to a full transformer layer:
multi-head `softmax(QK^T/√d)@V`, `workloads.GPT2_BLOCK_FULL`. Modeled as
`B·H` independent small GEMMs (no cross-instance weight reuse, unlike a
linear layer) plus a memory-only softmax pass over the `O(S²)` score tensor.
Backward priced at 2× forward (compute + bytes) — a stated approximation,
not derived GEMM-by-GEMM the way linear-layer backward is.

Sequence-length sweep (batch 8, array 256², 8 MiB, 100 GB/s):

| seq | GEMMs | attention | attention share |
|----:|------:|----------:|:---------------:|
| 128 | 3.96 ms | 0.76 ms | 16% |
| 512 | 7.36 ms | 9.06 ms | 55% |
| **1024** | 11.89 ms | 34.23 ms | **74%** |
| 8192 | 75.31 ms | 2077.69 ms | 97% |

**Crossover 256→512 tokens.** At GPT-2's own context length attention is
already 74% of the step. Quantifies the standard "attention is quadratic"
result and the motivation for FlashAttention-style fusion.

**Fourth lever — fusion** (`experiments/attention_fusion.py`, same hardware):
a flash-attention-style kernel removes the `O(S²)` score-tensor traffic
entirely (`O(S)` remains — just Q/K/V/O). Not SRAM-gated like
`Optimizer.state_resident` (see docs/attention.md for why). Saving *grows*
with sequence length:

| seq | naive | fused | saved | activations |
|----:|------:|------:|:-----:|:---:|
| 128 | 0.76 ms | 0.42 ms | 45% | 3.1 → 0.10 MB |
| 1024 | 34.23 ms | 2.01 ms | 94% | 201.3 → 0.79 MB |
| 8192 | 2077.7 ms | 46.8 ms | **98%** | 12.9 GB → 6.3 MB (2048×) |

**Silicon anchor — Tesla T4, fp16** (`validate/silicon/attention/profile_attention.py`,
`results/validation/attention_silicon.csv`): naive is **~1.75× optimistic**
(0.57× ratio) — same direction/magnitude as the uncalibrated GEMM anchor.
Fused was **~6.7× optimistic** (0.15× ratio, 86.1% predicted fusion saving
vs 46.6% measured), traced to real fused/memory-efficient-attention kernels
**recomputing** `S`/`P` from `Q`/`K` before computing `dV`/`dP`/`dQ`/`dK`
(confirmed via `fused_top_ops`: `fmha_cutlassF`/`fmha_cutlassB`, PyTorch's
memory-efficient-attention backend — T4 is Turing/sm75, so not literal
FlashAttention, but a genuine fused kernel). **Fixed the mechanism, not the
number**: `attention_step` now charges that recompute explicitly
(`2×forward_compute + qk_compute_s` when fused, derived from the algorithm,
not fit to this data point) — model's own backward:forward ratio moves
2.0×→**2.4×**, narrowing fused to **~5.8× optimistic** (0.17×, 84.0%
predicted saving) against a measured **4.5×** ratio / 46.6% saving. The
fix is real and in the right direction; the remaining gap is left open
rather than patched from n=1. An `elementwise_seconds` primitive was later
added (§2, for the optimizer fix) and tried here too — rejected, its fitted
rate doesn't extrapolate to attention's much larger score tensor (see
docs/attention.md's Open section).

## 7 — Model import (`simulator/import_model.py`)

Replaces hand-typed `workloads.py` shapes with `layers_from_torch(model,
example_input)`: traces any `nn.Module` via `torch.fx`, walks the graph for
`nn.Linear` calls, returns the `Layer` list `model_step` already takes.
`experiments/import_model_demo.py` proves it on a shape nobody hand-typed —
a Llama-7B-shaped SwiGLU MLP (`d_model=4096, d_ff=11008`) — and gets exact
shapes back (`gate_proj`/`up_proj` M=8192 K=4096 N=11008, `down_proj`
K=11008 N=4096), then runs the normal cost model on them. v1 is
`nn.Linear` only; attention still needs manual construction. Not a core
dependency (`pip install torch` separately). `docs/import_model.md`.

## 8 — Silicon anchor (`validate/silicon/`)

Scores the model against a **real GPU** instead of against Timeloop.
`profile_step.py` times the six weight GEMMs' forward / backward / Adam phases
in isolation on CUDA; `compare_silicon.py` maps the device to a
`Hardware`/`Memory` (`gpu_specs.py`) and tabulates measured vs predicted.
Scope: the six weight GEMMs only (no attention / norms) — tests the GEMM +
traffic + optimizer arithmetic, not transformer coverage.

**Run 1 — Tesla T4, fp16, 8192 tokens.** Raw model (ideal-array defaults) was
**2–5× optimistic**: forward 0.45×, backward 0.21×, optimizer 0.42×, step
0.27× (predicted / measured). All errors one direction. Decomposition:
forward — T4 sustained ~54 % of its 65 TFLOP/s fp16 peak, model assumed
~100 %; backward — worse (~17 %), skinny `wgrad` (`M`=768) + `dY` (50 MB)
re-streaming the 4 MB L2; optimizer — real Adam is ~12 tiny elementwise
kernel launches, not one bandwidth stream.

**Model changes made in response (all default-off → abstract model and the
Timeloop cross-check unchanged; §1–3 numbers above are unaffected):**

| knob | where | effect |
|------|-------|--------|
| `Hardware.compute_efficiency` (GPU 0.55) | `matmul_seconds` | sustained-vs-datasheet GEMM rate — the compute analogue of `dram_efficiency` |
| `Hardware.util_tiles` (GPU 128) | `matmul_seconds` `_occupancy` | a GEMM whose output is < 128 array-tiles runs at reduced occupancy → skinny `wgrad` penalty |
| `Memory.library_gemm` (GPU on) | `gemm_traffic` | fixed-kernel libraries sweep full `K` per output tile instead of retiling to the Hong-Kung optimum → contraction re-stream when `S/4K < min(M,N)` |
| `Hardware.kernel_launch_s` (GPU 8 µs) | `optimizer_seconds` | +12 launch latencies for the Adam kernel chain |

Also fixed a **harness bug**: the profiled input tensors did not
`require_grad`, so `.backward()` measured only `wgrad`, not `dgrad + wgrad`.

**Re-score — Tesla T4, fp16, clean profile** (`results/validation/silicon.csv`,
`results/plots/silicon_anchor.png`):

| phase | measured | predicted | ratio | abs %err |
|-------|---------:|----------:|:-----:|:--------:|
| forward   |  4.12 ms |  4.11 ms | 1.00 |  0.3 % |
| backward  | 20.62 ms | 17.79 ms | 0.86 | 13.7 % |
| optimizer |  1.78 ms |  1.28 ms | 0.72 | 28.0 % |
| **step**  | **26.52 ms** | **23.19 ms** | **0.87** | **12.6 %** |

Phase MAPE 14 %. Three fitted constants (`compute_efficiency` 0.55,
`util_tiles` 128, `kernel_launch_s` 8 µs) on 18 points — **calibrated to this
one GPU.** Per-layer it is ±1.6×: `mlp_up` lands at 0.9×, its mirror-shape
`mlp_down` at 1.5× — the model is symmetric under M↔N but cuBLAS picks
different kernels, and that is not modellable without a kernel database. The
small `proj` GEMMs sit at ~0.5× (fixed launch / wave-quantisation overheads
the model still under-counts). The aggregate looks better than the per-layer
scatter because the errors partly cancel.

Legitimate but provisional: **a second anchor (A100 / L4) is needed** to
check the constants transfer. See `docs/silicon_anchor.md`.

**Updated after the optimizer elementwise-compute fix** (§2,
`docs/optimizer_state.md`): this table's `optimizer` phase got *worse* in
isolation (0.72×/28% → 1.59×/59%, now over- not under-predicting) — the fix
was fit against an Adam-only anchor with a different measurement context
(no interleaved forward/backward), so it doesn't transfer perfectly here.
But `step` improved (0.87×/12.6% → 0.93×/6.7%): the larger optimizer
over-prediction partially offsets backward's existing under-prediction.
Reported as what it is — a partial cancellation, not the model becoming
more correct everywhere — see `docs/silicon_anchor.md`.

## 9 — Removed: backward reuse & GEMM-operand precision

Two levers were built, exercised in the model, and then **deleted** — not
relabeled, not caveated, removed — once it was clear neither could be
checked against the hardware actually available:

- **Backward-pass reuse** (`BackwardReuse`, `NAIVE_BACKWARD`/`FUSED_BACKWARD`)
  claimed up to 27% latency from keeping `W`/`dY`/`dX` resident across
  `dgrad`/`wgrad`. The mechanism is specific to a systolic array's
  weight-stationary residency; a GPU's GEMM library exposes no control to
  turn that behavior on or off, so there was no experiment that could ever
  confirm or refute the claim on hand.
- **GEMM-operand precision** (`Precision`, `FP16`/`BF16`/`FP8`,
  `at_precision()`) claimed fp8 GEMM compute saves 40–49% by doubling
  tensor-core throughput. The only GPU this project has silicon-anchored
  against is a Tesla T4 (Turing) — no fp8 tensor cores exist on that
  hardware at all, so the claim was untestable with the resources on hand,
  not merely untested.

Both are gone from `simulator/` entirely (not hidden behind a flag): the
`reuse` parameter is gone from `training_step`/`model_step`, and
`Precision`/`at_precision` no longer exist in `compute.py`. The autotuner's
search space drops from 80 to 40 configs. Nothing else in the model changed
— `GPT2_BLOCK`'s baseline numbers, the Timeloop cross-check, and the GEMM
and attention silicon anchors are all unaffected, since none of them ever
exercised either lever.

The remaining four levers (optimizer precision/residency, recomputation,
gradient accumulation, attention fusion) all had a real path to hardware
validation, and as of this revision all have been run on a real T4 (§2, §6).

---

## Model revision history

| revision | change | baseline step | array flip | BW flip | SRAM flip |
|----------|--------|:-------------:|:----------:|:-------:|:---------:|
| v1 | fwd + naive per-GEMM bwd + Adam `7·param` | 31.6 ms | 48→64 | 400→800 | 16→32 MiB |
| backward rewrite | `_backward_seconds` with true operands | 22.0 ms | 64→96 | 200→400 | 4→8 MiB |
| optimizer v2 | `dW` not double-charged; precision/residency knobs | 21.96 ms | 64→96 | 200→400 | 4→8 MiB |
| recomputation | `recompute` keyword | 21.96 ms | 64→96 | 200→400 | 4→8 MiB |
| per-phase dataflow | `⌈held/sram⌉` refetch; **"OS wgrad wins" — retracted** | 31.24 ms | 64→96 | 200→400 | 8→16 MiB |
| traffic rewrite | `gemm_traffic` = compulsory + √-penalty; DRAM validated 0.0 % | 18.62 ms | 64→96 | 100→200 | 4→8 MiB |
| penalty + Hong-Kung | penalty constant → classic `2·MNK/√S`; array/BW sweeps moved to 8 MiB (penalty-free) | 16.96 ms @ 2 MiB / 13.58 @ 8 MiB | 128→192 | 50→100 | 4→8 MiB |
| microbatch + autotune | `microbatch=` gradient accumulation + `activation_bytes`; `simulator/autotune.py` exhaustive 80-config lever search (~1.5 ms). No change to defaults. | 16.96 ms | 128→192 | 50→100 | 4→8 MiB |
| GPU-realism knobs | `compute_efficiency`/`util_tiles`/`library_gemm`/`kernel_launch_s`, all default-off; silicon anchor 13% step error on a T4. No change to defaults. | 16.96 ms | 128→192 | 50→100 | 4→8 MiB |
| attention | `simulator/attention.py` + `GPT2_BLOCK_FULL`; crossover 256→512 tokens, 74% of the step at seq 1024. GEMM-only defaults (`GPT2_BLOCK`) unchanged. | 16.96 ms | 128→192 | 50→100 | 4→8 MiB |
| model import | `simulator/import_model.py` -- `torch.fx` tracing instead of hand-typed shapes; confidence intervals added to `compare_silicon.py`. No change to defaults. | 16.96 ms | 128→192 | 50→100 | 4→8 MiB |
| attention fusion | `attention_step(..., fused=True)` / `model_step(..., fused_attention=)`; not SRAM-gated. Saves 45%→98% of attention latency, seq 128→8192. No change to `GPT2_BLOCK` defaults. | 16.96 ms | 128→192 | 50→100 | 4→8 MiB |
| attention silicon anchor | Real T4 result: naive 0.57× (~1.75× optimistic, GEMM-like), fused 0.15× (~6.7× optimistic — fused-backward recomputation cost not modeled). No change to any default. | 16.96 ms | 128→192 | 50→100 | 4→8 MiB |
| fused-backward recompute | `attention_step` charges `2×fwd_compute + qk_compute_s` when fused (was flat `2×fwd`) — derived from the algorithm, not fit to data. Model's bwd:fwd ratio 2.0×→2.4×; fused narrows 6.7×→**5.8×** optimistic. Remaining gap left open (no elementwise-compute primitive; possible T4-kernel inefficiency). | 16.96 ms | 128→192 | 50→100 | 4→8 MiB |
| **lever removal** | Cut `BackwardReuse` (no GPU-exposed control to test it against) and the GEMM-operand `Precision`/fp8 lever (needs Hopper/Ada+ tensor cores; the T4 anchor is Turing) — removed rather than kept as unvalidatable claims (§9). Autotuner 80→40 configs. `GPT2_BLOCK` defaults never used either lever, so unchanged. | 16.96 ms | 128→192 | 50→100 | 4→8 MiB |
| recompute/gradaccum/optimizer silicon anchors | Real T4 results for the last 3 levers. Recompute: forward transfers (~13-18% err), backward over-predicts 2× — opposite direction from every other backward result. Gradaccum: same backward mismatch on latency; real peak memory 3-12× above predicted. Optimizer: bf16 state measured **33% slower** than fp32 vs model's predicted +20% faster — Turing has no native bf16 arithmetic. No change to any default; findings documented, not patched with new fitted constants (§2). | 16.96 ms | 128→192 | 50→100 | 4→8 MiB |
| elementwise-compute + memory-floor fixes | `elementwise_seconds` + a native-dtype-support gate fix the optimizer bf16 sign flip (predicted saving now −44.9%, matches measured −33.3% in direction). A fitted `framework_overhead_bytes`/`framework_overhead_scale` fixes gradaccum's memory prediction (ratio now ~1.0, was 3-12× low). Reusing the elementwise rate for attention's softmax backward was tried and rejected (40-200× outside its fitted tensor-size range). Root cause of recompute/gradaccum's backward miss identified: `profile_step.py`'s isolated-per-GEMM timing overstates real chained backward. No change to any default outside two new opt-in fields. (`docs/optimizer_state.md`, `docs/microbatching.md`.) | 16.96 ms | 128→192 | 50→100 | 4→8 MiB |
| isolated-vs-chained investigation (4 rounds: chain-grid, chain-depth, chain-generalize, GEMM-size) | Confirmed real chained backward is genuinely faster than the isolated-sum calibration (1.4-3.5×), and found one narrow, correctly-scoped fix (`Hardware.backward_util_tiles` + `chain_overhead_fwd_s`/`bwd_s`, a 768×768/8192-token chain, R²=0.974) — opt-in only, confirmed not to generalize past that shape. Two more promising-looking fixes were caught and reverted after failing their own bound checks (a 2-constant refit that only helped 2 of 4 configs; a backward-efficiency boost invisible because those GEMMs are memory-bound — matching the measured time that way would need 451 GB/s on a 320 GB/s GPU). Real remaining lead: `gemm_traffic`'s unvalidated traffic-penalty heuristic (§Limitations #1), a different model component. No change to any default; full five-round trail with every rejected fix's reasoning in `docs/silicon_anchor.md`. | 16.96 ms | 128→192 | 50→100 | 4→8 MiB |

The Timeloop cycle validation (ws 1.2 %) and the compulsory-traffic
validation (0.0 %) are unchanged from the traffic rewrite on.
