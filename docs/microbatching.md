# Gradient accumulation / microbatching

*The `microbatch=` keyword on `training_step` / `model_step`.*

## The knob

Process `M` tokens as `n = ⌈M / microbatch⌉` chunks, accumulate `dW` across
them, run the optimizer **once**. The classic way to train at a large
effective batch when activation memory doesn't fit.

## What the model does

```
mb  = microbatch (tokens per chunk),  n = ceil(M / mb)
forward_s   = n · forward(mb, N, K)          # weights re-read once per chunk
backward_s  = n · backward(mb, ...)          # + fixed overhead x n
            + 2·(n-1)·|dW| / bw    if dW does not fit in SRAM   (spill the
                                              accumulator between chunks)
optimizer_s = optimizer(...)                 # once
activation_bytes = mb · K · b                # stored X, per chunk (0 if recompute)
```

The smaller GEMMs also have lower arithmetic intensity (the K·N weight term
is a bigger fraction of `mb·K + K·N + mb·N`), so they drift toward
memory-bound — `_phase_seconds` captures this for free.

## What it reveals

`experiments/microbatching.py` — GPT-2 block, 8192 tokens, array 256×256,
100 GB/s, 32 MiB SRAM:

| microbatch | n | step latency | stored activations |
|-----------:|:-:|:------------:|:------------------:|
| 8192 (none) | 1 | 11.9 ms (+0 %) | 113 MB (100 %) |
| 4096 | 2 | 12.5 ms (**+5 %**) | 57 MB (**50 %**) |
| 2048 | 4 | 13.6 ms (+14 %) | 28 MB (25 %) |
| 1024 | 8 | 15.9 ms (+33 %) | 14 MB (12 %) |
| 512 | 16 | 20.4 ms (+71 %) | 7 MB (6 %) |

**The rule: each halving of the microbatch halves stored activations and
costs ~5–15 % latency, compounding.** So 2–4× accumulation is nearly free
and cuts activation memory 2–4×; past that the latency cost runs away
(weight re-reads + per-GEMM overhead dominate).

Compare MONET's activation-checkpointing result ("4 % cost → 13 MB saved" on
ResNet-18): same memory-vs-cost axis, different mechanism (checkpointing
recomputes; accumulation shrinks the working set). Accumulation gets a
better ratio here because it touches no extra compute — just more, smaller
passes.

## Interaction with the other levers

- **vs recomputation:** both cut activation memory. Accumulation is cheaper
  per MB saved when compute headroom is scarce; recomputation is better when
  you're already memory-bound (spare FLOPs are free). The model lets you
  pick per config.
- **vs the optimizer floor:** accumulation does **not** lower the optimizer
  term (it still runs once, same bytes). Its value is enabling a large
  effective batch — which raises every GEMM's arithmetic intensity and
  amortizes the fixed optimizer cost over more tokens.

## Limitations

- Weight re-reads are charged `n×` in full; a schedule that keeps a layer's
  weights resident across its own microbatches would pay less.
- The near-penalty-threshold regime, where smaller GEMMs *lower* latency by
  dropping below the tiling penalty, is real in the model but rests on the
  heuristic penalty term — don't lean on it.
- No modelling of the accumulation FP precision (bf16 vs fp32 accumulator).

## Silicon anchor

`validate/silicon/gradaccum/profile_gradaccum.py` / `compare_gradaccum.py` — a real
4-layer chain (same shapes as the recompute anchor) run at `microbatch`
fractions 1/2/4/8/16 of the full token count, measuring both step latency and
**peak activation memory** (`torch.cuda.max_memory_allocated`) against the
`microbatch=` lever directly.

**Run — Tesla T4, fp16, 8192 tokens** (`results/validation/gradaccum_silicon.csv`):

| frac | measured step | predicted step | ratio | measured MB | predicted MB | ratio |
|-----:|--------------:|----------------:|:-----:|-------------:|---------------:|:-----:|
| 1 | 11.47 ms | 18.83 ms | 1.64 | 256.0 | 88.1 | 0.34 |
| 2 | 11.67 ms | 13.92 ms | 1.19 | 161.2 | 44.0 | 0.27 |
| 4 | 11.77 ms | 11.27 ms | 0.96 | 108.1 | 22.0 | 0.20 |
| 8 | 12.77 ms | 15.84 ms | 1.24 | 81.4 | 11.0 | 0.14 |
| 16 | 15.36 ms | 26.96 ms | 1.76 | 68.8 | 5.5 | 0.08 |

`frac=1` here is the same config as the recompute anchor's "stored" case
(11.47 ms vs 11.51 ms measured independently — good cross-check on
measurement reproducibility), and shows the same ~1.6-1.8x
model-over-predicts-backward pattern documented in `docs/recomputation.md`
— not a new finding, the same shape-calibration mismatch showing up again.

**The real finding was on memory, not latency.** Predicted activation bytes
are the theoretical stored-tensor count only (`mb · K · b` per chunk);
measured peak memory was 3-12x higher at every fraction, and the gap
*grew* as the fraction grew (0.34 → 0.08) — a large, roughly fixed memory
floor in a real PyTorch training loop (weight buffers, autograd graph
bookkeeping, the CUDA caching allocator's own overhead and fragmentation)
that doesn't shrink as chunks get smaller, while the model's prediction
(correctly) kept shrinking toward zero.

### The fix: a fitted memory floor + scale

`simulator/memory.py` gained `Memory.framework_overhead_bytes` (a fixed
additive floor) and `Memory.framework_overhead_scale` (a multiplier on the
raw predicted activation bytes) — both default to the abstract model's
existing behavior (0 and 1.0). `compare_gradaccum.py`'s `fit_memory_floor`
does a closed-form OLS fit of `measured_mb ≈ floor + scale · raw_predicted_mb`
across all 5 fractions in one profile — a 2-parameter fit is justified here
(unlike the attention extrapolation attempt in `docs/optimizer_state.md`)
because it's fit and evaluated across the *same* 16x range of real data
points, not extrapolated to a wildly different scale.

Fit on the T4 anchor: `measured_mb ≈57.6 + 2.27 · raw_predicted_mb`.

| frac | measured MB | raw predicted MB | raw ratio | floor-fit predicted MB | fit ratio |
|-----:|------------:|------------------:|:---------:|------------------------:|:---------:|
| 1 | 256.0 | 88.1 | 0.34 | 257.7 | 1.01 |
| 2 | 161.2 | 44.0 | 0.27 | 157.6 | 0.98 |
| 4 | 108.1 | 22.0 | 0.20 | 107.6 | 0.99 |
| 8 | 81.4 | 11.0 | 0.14 | 82.6 | 1.01 |
| 16 | 68.8 | 5.5 | 0.08 | 70.1 | 1.02 |

Every fraction now lands within 2% — a near-exact fit, not a coincidence:
some of the framework overhead scales with the activation tensor size
(extra copies, gradient buffers proportional to activations) and some is a
genuine fixed floor (weight/optimizer-state buffers, CUDA context), and the
2-parameter model captures both. **This floor is fit to this specific GPU
and this specific 4-layer chain's total weight size — it is not baked into
`gpu_specs.py`'s defaults** (unlike `compute_efficiency`/`util_tiles`),
since a much bigger model would plausibly need a bigger fixed floor. Treat
it as a per-anchor calibration to report alongside the raw prediction, not
a new universal constant. `activation_bytes` on its own is still correctly
understood as a lower bound on real memory use, never an estimate of it.

**The model's own qualitative claim (halving the microbatch roughly halves
activation memory) was also not what the raw numbers showed**: measured MB
only dropped 256→161→108→81→69 (a ~3.7x range end to end) against a raw
predicted 88→44→22→11→5.5 (16x range) — accumulation still helps, just far
less than the raw model said. The floor-fit prediction reproduces this
correctly (its own MB range is 257.7→70.1, ~3.7x, matching the measured
range almost exactly) because the additive floor is what shrinks the
*relative* effect of accumulation, even though the underlying per-chunk
tensor count still drops by the full 16x.
