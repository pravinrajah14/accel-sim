# Optimizer-state cost

*What the `Optimizer` config in `simulator/dataflow.py` models.*

## The cost

Adam updates every parameter each step using four fp32 numbers: the master
weight, the gradient `dW`, the first moment `m`, the second moment `v`. The
update itself is a handful of FLOPs — it is entirely a DRAM-bandwidth cost.
The v1 model charged `param_count × 7 × 4` bytes (read {param, grad, m, v},
write {param, m, v}).

Two things were wrong / missing:

1. **The gradient is free.** `dW` was just produced by `wgrad` and is still
   on chip; a fused optimizer step consumes it there. `optimizer_seconds`
   now drops the `dW` read.
2. **`m` and `v` don't have to be fp32, and don't have to stream.** Both are
   standard production techniques (8-bit Adam; keeping optimizer state
   resident on small models).

## The two levers

```
Optimizer(state_bytes=4, state_resident=False)   # ADAM_FP32 (default)
```

* **`state_bytes`** — precision of `m`, `v`: 4 (fp32) / 2 (bf16) / 1 (fp8).
  `m` and `v` are `2 · n · state_bytes` bytes and are streamed read+write, so
  halving the precision halves the dominant term.

* **`state_resident`** — keep `m`, `v` in SRAM across the step instead of
  spilling. Taken only when `2 · n · state_bytes ≤ sram_bytes`. Above that
  threshold `m`, `v` never touch DRAM and only the master weight is streamed.

Traffic per layer:

```
bytes = 2·n·4                         # master weight read + write (always)
      + (0 if resident and m,v fit    # m, v read + write
         else 2 · 2·n·state_bytes)
```

## What it reveals

`experiments/optimizer_cost.py` (GPT-2 block, array 256×256, 100 GB/s, sweep
SRAM):

| optimizer state | reaches the floor at | per-step optimizer time |
|-----------------|:--------------------:|:-----------------------:|
| fp32, streamed  | —                    | 2.27 ms (baseline) |
| fp32, resident  | 24 MiB               | 0.76 ms |
| bf16, resident  | 16 MiB               | 0.76 ms |
| fp8, resident   | **8 MiB**            | 0.76 ms |

Same 3× floor (0.76 ms — only the master weights left), but lower precision
reaches it with **⅓ the SRAM**. Precision buys you the residency threshold.

## Combined with the other levers

On the GPT-2 block at array 256×256, 32 MiB SRAM, 100 GB/s:

| config | step latency |
|--------|:------------:|
| fp32 streamed Adam | 11.9 ms |
| + fp8 resident Adam state | **10.4 ms**  (−13 %) |

The tool's output is a ranked menu of single-accelerator training-latency
levers, each quantified and each with its capacity precondition.

## Limitations

* `state_resident` is decided per layer. Keeping every layer's `m`, `v`
  resident *simultaneously* for a whole model needs the sum to fit, not each
  layer individually.
* Assumes the optimizer step is fused right after `wgrad` so `dW` is hot;
  a non-fused schedule pays the `dW` round-trip.
* No modelling of stochastic-rounding or master-weight-in-bf16 schemes.

## Silicon anchor

`validate/silicon/optimizer/profile_optimizer.py` / `compare_optimizer.py` — times a
manual Adam update (`m`, `v` running averages) at fp32 vs bf16 state for the
four GPT-2 layer sizes, isolating exactly the `state_bytes` lever with no
forward/backward in the loop.

**First run — Tesla T4** (before the fix below):

| state | measured | predicted | ratio |
|-------|---------:|----------:|:-----:|
| fp32 | 2.89 ms | 0.97 ms | 0.34 |
| bf16 | 3.85 ms | 0.78 ms | 0.20 |

**measured bf16 saving: −33.3%    predicted bf16 saving: +20.2%**

A sign flip, not a magnitude miss: the model said bf16 state should be
*faster* (fewer bytes to stream), and on this GPU it was measurably
*slower*. Cause: **Turing (T4, sm75) has no native bf16 arithmetic** — that
arrived with Ampere. PyTorch has to promote/emulate the `.float()` up-casts
in the manual Adam step, and that emulation cost outweighs the bandwidth
saving from the smaller `m`/`v` tensors. `profile_optimizer.py`'s docstring
called this out as a real possibility before the run — it was confirmed,
not a bug.

Also: even the fp32 baseline was under-predicted 3x here (0.34 ratio) —
`optimizer_seconds` was pure bytes/bandwidth with **zero compute term**, so
Adam's real elementwise arithmetic (`mul`/`addcmul`/`sqrt`/`div`) was priced
at zero regardless of hardware or precision. That's *why* a smaller byte
count always looked like a win no matter what the GPU actually supported.

### The fix: an elementwise-compute primitive + a native-dtype gate

`simulator/compute.py` gained `elementwise_seconds(n_elements, hw, native=)`,
using a new `Hardware.vector_flops_per_s` (peak elementwise-ALU throughput,
distinct from tensor-core `peak_macs_per_s`) and `Hardware.emulation_penalty`
(multiplier when the dtype isn't natively supported). `Hardware.native_low_precision_bytes`
says which `state_bytes` values run at native throughput — `{4}` only for
Turing/Volta in `gpu_specs.py` (T4, V100), `{4, 2, 1}` everywhere else,
defaulting to `{4, 2, 1}` so the abstract model is unchanged. Both new
constants were fit in one pass, in `compare_optimizer.py`'s
`refit_elementwise`, against **this same anchor's** per-layer data — the
native (fp32) residual over the existing bytes-only prediction isolates
the elementwise rate; the non-native (bf16) residual, using that already-
fixed rate, isolates the emulation penalty. Fit: `vector_flops_per_s` =
4.571e9 elements/s, `emulation_penalty` = 1.94x.

**Re-scored with the fix**:

| state | measured | predicted | ratio |
|-------|---------:|----------:|:-----:|
| fp32 | 2.89 ms | 2.26 ms | 0.78 |
| bf16 | 3.85 ms | 3.28 ms | 0.85 |

**measured bf16 saving: −33.3%    predicted bf16 saving: −44.9%**

The sign is now correct (both say bf16 is *slower*, not faster), and the
fp32 error dropped from 66% to 22%. The model now slightly over-corrects
(predicts an even bigger bf16 penalty than measured) — a reasonable
residual for a single-GPU, single-anchor fit; not chased further with a
second constant.

**Practical conclusion:** the `state_bytes` lever's recommendation (lower
precision optimizer state is free latency, `results/RESULTS.md` §2) is only
true on hardware with native support for that dtype's arithmetic — Ampere+
(A100/L4/H100) for bf16, Hopper+ for fp8 (fp8 state has zero hardware data
either way — `_PRE_AMPERE` GPUs are treated as non-native for it too, by
analogy, not by measurement). `test_native_bf16_gpu_still_finds_lower_precision_cheaper`
(`tests/test_sanity.py`) confirms the gate doesn't regress Ampere+: bf16
state is still predicted cheaper there, since `emulation_penalty` never
applies when the dtype is in `native_low_precision_bytes`.

**Tried and rejected:** applying the *same* fitted `vector_flops_per_s` to
`attention_step`'s fused-backward softmax-recompute gap
(`docs/attention.md`), on the theory that both are "elementwise compute
priced at zero." The rate was fit against 0.5–2.4M-element tensors
(Adam's `m`/`v` arrays); attention's `(B,H,S,S)` score tensor is 100M+
elements at GPT-2's shapes — 40–200x outside the fitted range. Applied
there, it predicts a ~22ms overhead by itself, dwarfing the entire measured
fused-attention step. Not adopted; see `simulator/attention.py`'s inline
note. The softmax-backward compute gap for attention stays open and needs
its own dedicated, attention-scale calibration.

Separately: even before this fix, isolating Adam (this anchor) showed a
*worse* fp32 error (66%) than the ~28% seen when Adam ran as part of a full
training step (`docs/silicon_anchor.md`) — isolated, this benchmark pays
the full kernel-launch overhead with nothing else on the GPU to overlap it
with, while inside a real step that overhead partially hides behind other
work. After the fix, re-scoring the *original* GEMM anchor's own optimizer
phase (which measures Adam interleaved with forward/backward, a different
measurement context than this anchor's Adam-only loop) got *worse* in
isolation (28% → 59% error) even as the anchor it was fit against improved —
the fitted rate doesn't transfer perfectly across measurement contexts, only
partially, and the step-level GEMM-anchor error still improved overall
(12.6% → 6.7%) because the errors partly offset. See `docs/silicon_anchor.md`.
