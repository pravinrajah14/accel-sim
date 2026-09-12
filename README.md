# accel-sim — a lightweight analytical training-step simulator

## Abstract

`accel-sim` is a fast (<1 ms per query) analytical model that estimates the
**per-training-step latency** of individual neural-network layers on a single
systolic-array accelerator, including the parts that inference-only models
skip: the backward pass (~2× the forward MACs, as two additional GEMMs) and
the Adam optimizer state (momentum + variance ≈ 2× resident parameter memory,
plus a bandwidth-bound update stream).

Running the six GEMMs of a GPT-2-small transformer block (batch 8, sequence
1024), the model produces three results.

**1 — the bottleneck shift.** The training step crosses from compute-bound
to memory-bound as you scale any one of the three hardware knobs, sharply —
and above each crossover the curve is flat, so spending on that knob is
wasted:

| swept knob (others fixed)                 | bottleneck shift  |
|-------------------------------------------|-------------------|
| array size (SRAM 8 MiB, BW 100 GB/s)      | between **128×128 and 192×192 PEs** |
| DRAM bandwidth (array 128², SRAM 8 MiB)   | between **50 and 100 GB/s** |
| on-chip SRAM (array 128², BW 100 GB/s)    | between **4 and 8 MiB** |

The *shape* — sharp crossover, then a flat floor where more of that knob is
wasted — is robust; the exact crossover point has moved ±1 grid step as the
model tightened (see `results/RESULTS.md` revision log).

**2 — a ranked menu of training-latency levers**, each with its capacity
precondition:

| lever | mechanism | payoff |
|-------|-----------|--------|
| **low-precision resident optimizer** | `m`,`v` in fp8, kept in SRAM not spilled | optimizer **−3×** (2.3 → 0.8 ms), threshold at 8 MiB SRAM (fp8) vs 24 MiB (fp32) |
| **activation recomputation** | regenerate `X` in backward instead of reading it | **+13 %** while memory-bound, **−31 %** while compute-bound; crossover ~250 GB/s |
| **gradient accumulation** | `n` microbatches, accumulate `dW`, optimize once | each ×2 accumulation **halves stored activations for ~+5–15 % latency** (compounding) — 2–4× is nearly free |
| **attention fusion** | never materialize the `S×S` score tensor | **45%→98%** of attention latency, growing with sequence length (§Attention) |

A **backward-pass reuse** lever and a **GEMM-operand fp8 precision** lever
were built and then removed — neither could be checked against real
hardware with the resources on hand. See Limitations and §9 of
`results/RESULTS.md`.

**3 — a closed-form autotuner.** The first three levers above are discrete,
and every evaluation is µs, so `simulator/autotune.py` brute-forces the
whole 4×2×5 = **40-config space in ~1 ms** and returns the min-latency plan
(optionally under an activation-memory budget). The pick is config-dependent
— tight SRAM → fp8 optimizer + recompute; generous → keep it simple. MONET
explores *one* lever with a genetic algorithm because each of its
evaluations is a mapping search; a closed-form model makes exhaustive
search free. [`docs/autotune.md`](docs/autotune.md).

Stacked on the GPT-2 block (array 256×256, 100 GB/s): baseline **11.9 ms** →
+ fp8 resident Adam **10.4 ms**. Docs:
[`optimizer_state`](docs/optimizer_state.md),
[`recomputation`](docs/recomputation.md),
[`microbatching`](docs/microbatching.md),
[`autotune`](docs/autotune.md).

> **Retracted:** an earlier revision claimed output-stationary `wgrad` saved
> up to 34 %. The traffic validation disproved it — a well-scheduled
> weight-stationary `wgrad` reaches the same DRAM traffic by tiling the
> contraction. DRAM traffic is dataflow-independent.
> [`docs/dataflow_choice.md`](docs/dataflow_choice.md).

All current numbers live in [`results/RESULTS.md`](results/RESULTS.md), which
also logs how each model revision moved them. Plots:
[`results/plots/`](results/plots/).

The model is cross-checked against **Timeloop** (the standard mapping-search
tool) on the GPT-2 GEMM shapes: **cycles** within 1.2 % (weight-stationary)
/ 0.2 % (output-stationary), and **DRAM traffic exact (0.0 %)** on all 15
configs including a weight-stationary `wgrad` SRAM staircase (see
[Timeloop validation](#timeloop-validation--validate)).

## Why this exists

Fast analytical accelerator models (Timeloop, MAESTRO, SCALE-Sim, LLMCompass)
model the **forward pass / inference only**. Training-aware simulators
(ASTRA-Sim, SimAI) are cluster-scale and slow; Calculon models training but
at parallelism/cluster scope. **MONET** (2026) is the closest match —
fast single-accelerator training modeling — via a mapping-search framework
(Stream). `accel-sim` is a deliberately minimal alternative: closed-form (no
search, ~µs/query), cross-validated against **Timeloop**, and framed around
which single-chip knobs and software levers actually move the training step.
A learning project first, but a real point in that design space.

## Model

For how the pieces fit together — the modelled accelerator and the software
layering — see [`docs/architecture.md`](docs/architecture.md).

### Compute — `simulator/compute.py`
Systolic array, `matmul_cycles(M, N, K, hw, dataflow)`:

```
cycles = ceil(M·N·K / (rows·cols))   # MAC term
ws:    + K + rows + cols             # shift the K×N operand in
os:    + rows + cols                 # no shift-in
```

Timeloop-validated: 1.2 % (ws) / 0.2 % (os), always high by the fixed
overhead constant. The dataflow only moves ~1–3 % of cycles.

### Memory — `simulator/memory.py`
One hierarchy level: on-chip SRAM + off-chip DRAM.
`gemm_traffic(M, N, K, hw, mem)` — **dataflow-independent**:

```
compulsory = (M·K + K·N + M·N) · b     # each tensor crosses DRAM once
penalty    = 0                          if the smallest tensor + a working
                                        strip fits in SRAM
           ≈ 2·M·N·K·b / √(S / 2b)      otherwise (re-streaming under tiling)
```

The scheduler keeps the **smallest** of the three tensors resident and
streams the other two (tiling the contraction if needed). For forward that's
the weight matrix; for `wgrad` it's `dW` — both small — so neither phase
thrashes at realistic SRAM. The compulsory term is Timeloop-validated
**exact**; the penalty is a first-order heuristic. Latency assumes
double-buffering: `max(compute_s, memory_s)`.

### Training step — `simulator/dataflow.py`
Per layer `Y = X·W`:

| phase     | GEMM(s)                              | relative MACs |
|-----------|--------------------------------------|---------------|
| forward   | `Y  = X·W`      (M,K)×(K,N)          | 1×            |
| backward  | `dX = dY·Wᵀ`    (M,N)×(N,K)  and  `dW = Xᵀ·dY` (K,M)×(M,N) | ~2× |
| optimizer | Adam update: master weight r+w; `m`,`v` r+w; `dW` free (from wgrad) | bandwidth-bound |

**Optimizer state** (`Optimizer`, default `ADAM_FP32`). Two levers:
`state_bytes` (4 fp32 / 2 bf16 / 1 fp8 for `m`,`v`) and `state_resident`
(keep `m`,`v` in SRAM instead of spilling, when they fit).
[`docs/optimizer_state.md`](docs/optimizer_state.md).

**Activation recomputation** (`recompute=`, keyword). Regenerate the layer
input in the backward pass instead of reading it — trades traffic for
compute, a win only while memory-bound. [`docs/recomputation.md`](docs/recomputation.md).

**Gradient accumulation** (`microbatch=`, keyword). `n` microbatches,
accumulate `dW`, run the optimizer once; `StepBreakdown.activation_bytes`
reports the stored-activation footprint. [`docs/microbatching.md`](docs/microbatching.md).

### Workloads — `simulator/workloads.py`
GPT-2-small block, `d_model=768`, `d_ff=3072`, M = batch·seq = 8192 tokens:
`q_proj`, `k_proj`, `v_proj`, `attn_out` (768×768), `mlp_up` (768×3072),
`mlp_down` (3072×768). Attention score/context batched matmuls are out of
scope for v1.

## Reproduce

```bash
cd accel-sim
./run_all.sh
```

That installs deps, runs the sanity tests, regenerates the three sweep plots
in `results/plots/`, and (if the Docker daemon is up) runs the Timeloop
cross-check. Individual pieces:

```bash
python3 -m pytest -q tests/
python3 experiments/sweep_array_size.py
python3 experiments/sweep_sram.py
python3 experiments/sweep_bandwidth.py
python3 experiments/optimizer_cost.py     # Adam state precision x residency
python3 experiments/recomputation.py      # recompute crossover vs bandwidth
python3 experiments/microbatching.py      # memory-vs-latency of grad accumulation
python3 experiments/autotune.py           # exhaustive lever search, ~1 ms
```

## Timeloop validation — `validate/`

Timeloop models a single GEMM. The cross-check covers **both cycles and DRAM
traffic** (no notion of backward/optimizer — the 2× FLOP rule and Adam byte
count are validated by construction).

### Result — 15 configs

| family | GEMM / dataflow | n | cycles MAPE | DRAM MAPE |
|--------|-----------------|:-:|:-----------:|:---------:|
| `ws-fwd` | forward, weight-stationary | 6 | 1.2 % | **0.0 %** |
| `os-wgrad` | `dW = Xᵀ·dY`, output-stationary | 6 | 0.2 % | **0.0 %** |
| `wsw-stair` | `dW = Xᵀ·dY`, weight-stationary, SRAM sweep | 3 | 0.7 % | **0.0 %** |

(`results/validation/validation.csv`.) Timeloop's cycle count **and** DRAM
byte count equal the model's compulsory figures exactly. The model is high
on cycles only by its fixed overhead constant. The `wsw-stair` family was
the one that killed the earlier "output-stationary `wgrad` wins" claim —
weight-stationary `wgrad` hits compulsory traffic at 8 MiB SRAM (25 MB
stationary operand) by tiling the contraction, which the old
`⌈held/sram⌉` model got 30–90 % wrong.

**Not validated:** the traffic *penalty* term (smallest tensor spills) — a
first-order heuristic. The staircase confirms penalty = 0 where it should be
but doesn't pin the coefficient.

### How it works

`gen_timeloop_configs.py` stamps the vendored **`simple_weight_stationary`**
and **`simple_output_stationary`** example designs (`validate/bench/`, pinned
to commit `2d55108` of `Accelergy-Project/timeloop-accelergy-exercises`) with
a mapspace constrained to a *single* mapping that stages the streamed
operands in the global buffer (minimal traffic). Mapspace size 1 → the
mapper just reports that mapping's cycles + per-tensor DRAM accesses.

Each GEMM is a 1×1 convolution; the token dimension is one sequence (1024;
4096 for the staircase).

```bash
# needs Docker Desktop running; pulls ~4 GB image on first run
python3 validate/gen_timeloop_configs.py --arrays 64 128
python3 validate/gen_timeloop_configs.py --staircase
validate/run_timeloop.sh
python3 validate/compare.py       # -> results/validation/validation.csv
```

Image: `timeloopaccelergy/accelergy-timeloop-infrastructure:latest`
(multi-arch; the run script sets `LD_LIBRARY_PATH=/usr/local/lib`, needed by
the arm64 build).

## Silicon anchor — `validate/silicon/`

Timeloop is a model. This harness scores the predictions against a **real
GPU**. `profile_step.py` (torch + CUDA; a free Colab T4 works) times the
forward / backward / Adam phases of each GPT-2-block weight GEMM in isolation
— exactly what `training_step` predicts — and emits a portable JSON.
`compare_silicon.py` runs locally with no GPU: it maps the device to an
accel-sim `Hardware`/`Memory` (`gpu_specs.py` — peak tensor FLOP/s → a
throughput-equivalent square array, L2 → `sram_bytes`, HBM → bandwidth) and
prints a phase-by-phase measured-vs-predicted table + MAPE, writing
`results/validation/silicon.csv` and `results/plots/silicon_anchor.png`.

Scope is deliberately the six weight GEMMs only (no attention, no norms) —
this tests the GEMM + traffic + optimizer model, not transformer coverage.

**Run 1 — Tesla T4, fp16.** The raw model (ideal-array defaults) was
**2–5× optimistic**, all one direction. Adding four **default-off** GPU-mode
knobs — `compute_efficiency` (sustained-vs-datasheet GEMM rate, 0.55),
`util_tiles` (skinny-GEMM occupancy, 128), `library_gemm` (fixed-kernel
contraction re-stream), `kernel_launch_s` (Adam kernel chain) — plus a
harness fix (inputs now `require_grad`), the calibrated model lands at:

| phase | ratio (pred/meas) | abs %err |
|---|:--:|:--:|
| forward | 1.00 | 0.3 % |
| backward | 0.86 | 13.7 % |
| optimizer | 0.72 | 28.0 % |
| **step** | **0.87** | **12.6 %** |

**13 % step error** — but on 3 fitted constants / one GPU, and per-layer it
is ±1.6× (mirror-shape `mlp_up`/`mlp_down` straddle at 0.9× / 1.5× because
cuBLAS kernel selection isn't modellable). The abstract model and Timeloop
cross-check are untouched (all knobs default to the ideal array). A second
anchor (A100 / L4) is needed to check the constants transfer.
`docs/silicon_anchor.md`.

```bash
# on a GPU box:
python validate/silicon/gemm/profile_step.py --out silicon_profile.json
# back on a laptop:
python validate/silicon/gemm/compare_silicon.py silicon_profile.json
```

## Attention — `simulator/attention.py`

Everything above is six linear layers. This adds the actual attention op —
multi-head `softmax(QK^T/√d)@V` — for a first complete transformer layer
(`workloads.GPT2_BLOCK_FULL`). Modeled as `B·H` *independent* small GEMMs
(no cross-batch weight reuse, unlike a linear layer) plus a memory-only pass
for `softmax` over the `O(S²)` score tensor. Backward is priced at 2× forward
by assumption (see Limitations) rather than derived GEMM-by-GEMM.

`experiments/attention_scaling.py` sweeps sequence length: attention
overtakes the four projections + MLP at **256→512 tokens**, and is already
**74%** of the training step at GPT-2's own 1024-token context:

| seq | GEMMs | attention | attention share |
|----:|------:|----------:|:---:|
| 128 | 3.96 ms | 0.76 ms | 16% |
| 512 | 7.36 ms | 9.06 ms | 55% |
| 1024 | 11.89 ms | 34.23 ms | 74% |
| 8192 | 75.31 ms | 2077.69 ms | 97% |

The quantitative version of "attention is quadratic," and the concrete
motivation for a flash-attention-style fusion lever.

**Fourth lever: fusion.** `attention_step(..., fused=True)` models exactly
that — never materializing the `S×S` score tensor cuts DRAM traffic from
`O(S²)` to `O(S)`. Unlike `Optimizer.state_resident` it isn't SRAM-gated
(see `docs/attention.md`). `experiments/attention_fusion.py`: saves 45% of
attention latency at seq 128, **98% at seq 8192** (12.9 GB → 6.3 MB of
stored activations — a 2048× cut). The saving grows exactly where naive
attention hurts most.

**Silicon anchor result — Tesla T4.** Naive attention is **~1.75× optimistic**
(0.57× ratio) — in line with the uncalibrated GEMM anchor. Fused was
**~6.7× optimistic**: measured fusion saves 46.6% of the step; the model
predicted 86.1%. Root cause (confirmed via `fused_top_ops`: PyTorch's
memory-efficient-attention CUTLASS backend on this Turing GPU) — real fused
kernels **recompute** `S`/`P` from `Q`/`K` before computing
`dV`/`dP`/`dQ`/`dK`, since `P` was never stored, and the model's flat
`2× forward` backward rule didn't charge for it. **Fixed the mechanism**:
`attention_step` now derives that recompute cost from the algorithm itself
(not fit to this data point) — model's backward:forward ratio moves
2.0×→**2.4×**, narrowing fused to **~5.8× optimistic** against a measured
4.5× ratio. The fix is real and directionally correct; the remaining gap
is left open (no elementwise-compute cost primitive exists in this
codebase; possibly also a Turing-specific kernel-efficiency effect) rather
than papered over with another guessed constant. `docs/attention.md`.

## Importing real models — `simulator/import_model.py`

Every workload above was hand-typed. `layers_from_torch(model,
example_input)` traces an arbitrary `nn.Module` with `torch.fx`, walks the
graph for `nn.Linear` calls, and returns the same `Layer` list `model_step`
already takes — no manual shape-copying. `experiments/import_model_demo.py`
proves it on a shape nobody hand-typed: a Llama-7B-shaped SwiGLU MLP
(`d_model=4096, d_ff=11008`), traced directly from a real module and run
through the exact same cost model. v1 is `nn.Linear` only — attention still
needs manual `Attention(...)` construction (`docs/import_model.md`). Not a
core dependency: `pip install torch` separately.

## Related work

* **Timeloop / Accelergy** (Parashar et al., ISPASS 2019) — loop-nest
  mapping search + energy; forward pass only. The validation oracle here:
  the model's cycles match to 1.2 % / 0.2 % and its DRAM traffic exactly, on
  the GPT-2 GEMM shapes, running Timeloop's own `simple_weight_stationary` /
  `simple_output_stationary` example designs. It also **falsified** an
  earlier finding of this project (per-phase dataflow).
* **MAESTRO** (Kwon et al., MICRO 2019) — data-centric reuse analysis via
  explicit dataflow directives; inference only.
* **SCALE-Sim** (Samajdar et al., ISPASS 2020) — cycle-accurate systolic
  array, forward pass; the compute model here is a one-equation analytical
  reduction of the same weight/output-stationary mechanics.
* **LLMCompass** (Zhang et al., ISCA 2024) — analytical single-accelerator
  hardware-design model, ~4–11 % latency error vs measured. **Inference
  only.**
* **MONET** (2026) — the closest prior work: fast single-accelerator
  *training* modeling (fwd + bwd + memory footprint) on heterogeneous
  dataflow accelerators, built on the experimentally-verified **Stream**
  inference framework, evaluated on ResNet-18 + small GPT-2. `accel-sim` is a
  lighter, closed-form take on the same problem (no mapping search; ~µs per
  query) cross-validated against Timeloop rather than Stream, and adds the
  explicit lever analysis (optimizer precision / recompute / gradient
  accumulation / attention fusion).
* **Calculon** (Isaev et al., SC 2023) — analytical LLM-*training*
  co-design, but at **cluster / parallelism** scope, transformer-only.
* **Hong & Kung** (1981) — the `Ω(M·N·K/√S)` matmul I/O lower bound; the
  `gemm_traffic` tiling penalty uses its classic 2× form.
* **MLSYSIM** (2026) — first-principles "systems-walls" modeling from µW to
  GW; coarser than microarchitecture. Names the fast-vs-training-vs-single-
  chip gap that motivated this project.
* **Roofline models** — the compute-vs-memory crossover is a roofline
  result; the contributions on top are the full training step and the
  systems levers a per-GEMM roofline cannot see.

## Limitations

1. **Traffic penalty term is a heuristic.** The compulsory traffic
   (`M·K + K·N + M·N`) is Timeloop-validated exact; the `≈ 2·M·N·K·b/√S`
   penalty for when the smallest tensor spills is first-order and not
   directly validated (the staircase only confirms penalty = 0 where it
   should be).
2. **Cycle model assumes perfect operand double-buffering.** `matmul_cycles`
   charges only the MAC term + a fixed fill/drain; Timeloop agrees for the
   pinned mappings.
3. **`max(compute, memory)`** assumes perfect double-buffered overlap, and
   the phases (fwd / bwd / optimizer) are summed, not overlapped.
4. **Attention's naive backward is 2× forward by assumption**, and fused
   backward's recompute term (`2×forward + qk_compute_s`) is derived but
   still ~2.4× vs a measured 4.5× on the one GPU checked. Applying the new
   `elementwise_seconds` primitive (below) to this gap was tried and
   rejected — it was fit on 0.5-2.4M-element tensors, attention's score
   tensor is 100M+, 40-200× outside that range, and it predicted a
   physically implausible ~22ms overhead. No layernorm, GELU, residuals, or
   multi-chip. (`docs/attention.md`.)
5. **Gradient accumulation charges weight re-reads `n×` in full**; a schedule
   that kept a layer's weights resident across its own microbatches would pay
   less. Real peak activation memory ran 3-12× above the model's raw
   stored-tensor count; fixed with a 2-parameter floor+scale fit reported
   per-anchor, not baked into the model's defaults. (`docs/microbatching.md`.)
6. **Recompute/gradaccum's backward is over-predicted ~2×** on a real T4 —
   traced to a measurement-methodology gap, not a shape gap:
   `profile_step.py` (the GEMM anchor) timed each GEMM **in isolation** and
   summed the results; a real chained backward on this GPU runs about half
   as fast as that isolated sum. A five-round investigation (chain-grid,
   chain-depth, chain-generalize, GEMM-size) found one narrow, correctly-
   scoped fix (one shape only, opt-in) and traced the rest to a different,
   unvalidated model component (`gemm_traffic`'s traffic-penalty
   heuristic) — full trail in `docs/silicon_anchor.md`.
7. **Lower optimizer-state precision only saves latency on hardware with
   native arithmetic for that dtype.** Fixed for Turing/Volta (T4, V100) via
   `Hardware.native_low_precision_bytes` + `emulation_penalty`, calibrated
   against the isolated Adam anchor; fp8 state has zero hardware data on any
   GPU and is treated as non-native on pre-Ampere by analogy, not
   measurement. (`docs/optimizer_state.md`.)
8. **Two levers were removed, not just left unvalidated** — backward-pass
   reuse (no GPU-exposed control to test it against) and GEMM-operand fp8
   precision (needs tensor cores the validation GPU doesn't have). See
   `results/RESULTS.md` §9 if reviving either becomes possible.

## Next steps

**Pushing every anchor phase toward ~5% error, not just "documented and
open."** All five anchors (GEMMs, attention, recompute, gradient
accumulation, optimizer precision) have real Tesla T4 data. Two gaps have
been fixed with new, honestly-calibrated model terms since: the optimizer's
bf16-on-Turing sign flip (an `elementwise_seconds` compute primitive + a
native-dtype gate) and gradaccum's activation-memory floor (a fitted
additive+multiplicative correction). One attempted fix was tried and
explicitly rejected rather than forced in (reusing the optimizer's
elementwise rate for attention's softmax backward — see Limitations #4).

What's left, in priority order:

* **A dedicated DRAM-traffic-penalty validation.** Five real experiments
  chased the isolated-vs-chained backward gap across shape, depth,
  token-count, and GEMM-size — one narrow fix survived (opt-in, one shape),
  two more were caught and reverted after failing their own compute/memory
  bound check. The real remaining lead: `gemm_traffic`'s unvalidated
  traffic-penalty heuristic (Limitation #1), not compute efficiency — a
  different model component than anything tuned so far. Full five-round
  trail, including exactly why each fix was rejected:
  `docs/silicon_anchor.md`.
* **Attention's softmax-backward compute** stays open and needs its own
  attention-scale calibration (a dedicated isolated softmax/elementwise
  microbenchmark at real `(B,H,S,S)` tensor sizes) — the optimizer anchor's
  constant doesn't transfer to that scale, confirmed rather than assumed.
* **A second GPU (A100/L4)** — tests whether bf16 native support flips the
  optimizer-state result back to a saving (the model now predicts it
  should), whether the fused-attention backward gap (5.8× on Turing)
  shrinks on newer hardware, and whether the GEMM/backward calibration
  constants need to be per-GPU or can be derived from datasheet specs
  directly.

Backward-pass reuse and GEMM-operand fp8 precision are not in this list —
they were **removed from the codebase** rather than queued, since neither
had a real path to validation on hand (§Limitations, `results/RESULTS.md`
§9). Feature ideas shelved until validation catches up: folding fusion into
`autotune.py`'s search, auto-detecting attention in `layers_from_torch`.

## Repo layout

```
accel-sim/
  simulator/
    compute.py     matmul_cycles — cycles only
    autotune.py    exhaustive closed-form lever search (40 configs)
    memory.py      gemm_traffic — compulsory + tiling penalty
    dataflow.py    fwd + bwd + optimizer; Optimizer, recompute, microbatch
    attention.py   multi-head QK^T -> softmax -> (softmax)@V; fusion
    import_model.py  torch.fx -> Layer list, for arbitrary models
    workloads.py   GPT2_BLOCK (6 GEMMs) / GPT2_BLOCK_FULL (+ attention)
  experiments/
    sweep_array_size.py  sweep_sram.py  sweep_bandwidth.py
    optimizer_cost.py  recomputation.py
    microbatching.py  autotune.py  attention_scaling.py  attention_fusion.py
    import_model_demo.py  (Llama-shaped MLP, traced not hand-typed)
    dataflow_choice.py   (records the retracted finding)
  validate/
    gen_timeloop_configs.py  run_timeloop.sh  compare.py
    bench/                   vendored Timeloop example designs (ws + os, pinned)
    silicon/                 5 real-GPU anchors: profile_step.py/compare_silicon.py (GEMMs),
                             profile_attention.py/compare_attention.py (attention),
                             profile_recompute.py/compare_recompute.py (recompute),
                             profile_gradaccum.py/compare_gradaccum.py (gradient accumulation),
                             profile_optimizer.py/compare_optimizer.py (optimizer state), gpu_specs.py
  docs/
    architecture.md  optimizer_state.md
    recomputation.md  microbatching.md  autotune.md  attention.md
    import_model.md  silicon_anchor.md  dataflow_choice.md (retraction)
  tests/test_sanity.py
  results/
    RESULTS.md              canonical numbers + model revision log
    plots/*.png             sweeps + dataflow_choice + optimizer_cost + recomputation + attention_scaling + attention_fusion
    validation/validation.csv
  run_all.sh
```
