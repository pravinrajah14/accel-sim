# Activation recomputation

*The `recompute` keyword on `training_step` / `model_step`.*

## The trade

`wgrad` (`dW = Xᵀ · dY`) needs the layer input `X`. Normally `X` is kept from
the forward pass and read back from DRAM during the backward pass.

**Recomputation** (a.k.a. gradient / activation checkpointing) discards `X`
after the forward pass consumes it and regenerates it during the backward
pass by re-running the forward GEMM, just before `wgrad` needs it.

| | store `X` | recompute `X` |
|---|---|---|
| DRAM | read `M·K·b` in backward | — |
| compute | — | + one forward GEMM, `M·N·K / (rows·cols)` cycles |

So it converts a fixed amount of DRAM traffic into a fixed amount of
compute. Whether that is a win depends entirely on which one the layer is
bottlenecked on.

## What the model does

```
_backward_seconds(..., recompute=True):
    compute_s   += matmul_cycles(M, N, K, hw) / clock    # regenerate X
    wgrad_bytes -= M*K*b                                  # X no longer read
```

Everything else (the optimizer, the `gemm_traffic` penalty) is unchanged
and composes with it.

## What it reveals

`experiments/recomputation.py` — GPT-2 block, array 256×256, 8 MiB SRAM,
sweeping DRAM bandwidth:

| DRAM BW | store | recompute | Δ |
|--------:|------:|----------:|---:|
| 25 GB/s  | 47.6 ms | 41.5 ms | **+13 %** |
| 100 GB/s | 11.9 ms | 10.4 ms | **+13 %** |
| 200 GB/s |  6.0 ms |  5.6 ms | +6 % |
| 300 GB/s |  4.0 ms |  4.6 ms | −15 % |
| 800 GB/s |  3.0 ms |  3.9 ms | **−30 %** |

The two curves cross at the **roofline knee** (~250 GB/s for this config).
Left of it the layer is memory-bound and the freed DRAM traffic is real
latency; right of it the layer is compute-bound and the extra forward GEMM
is pure overhead. In the memory-bound plateau the benefit is a flat ~13 %
because both curves scale as `1/bandwidth` there.

This is a result the forward-only tools cannot produce — there is no
backward pass in them to checkpoint against.

## How it composes with the other levers

Recomputation *frees SRAM*: the activation it would have kept resident is
gone, so `dY` (the backward-reuse capacity gate) and `m`,`v` (the optimizer
gate) fit at a smaller SRAM size. The levers are not independent — the
tool's value is showing the combined frontier, not three separate curves.

## Silicon anchor

`validate/silicon/recompute/profile_recompute.py` / `compare_recompute.py` time a real
4-layer chain (GPT-2's own shapes: 768→768→768→3072→768) forward+backward,
**stored** (normal autograd) vs **recompute** (`torch.utils.checkpoint` —
PyTorch's actual implementation of this exact idea), and check the measured
delta against what `training_step(recompute=)` predicts. Same
`gpu_specs.py` calibration knobs as the other two anchors — directly
comparable, not a separate fit.

**Run — Tesla T4, fp16, 8192 tokens** (`results/validation/recompute_silicon.csv`):

| mode | phase | measured | predicted | ratio | abs %err |
|------|-------|---------:|----------:|:-----:|:--------:|
| stored | forward | 4.11 ms | 3.57 ms | 0.87 | 13.2% |
| stored | backward | 7.40 ms | 15.26 ms | **2.06** | **106.1%** |
| recompute | forward | 4.36 ms | 3.57 ms | 0.82 | 18.2% |
| recompute | backward | 7.18 ms | 15.54 ms | **2.17** | **116.6%** |

Forward transfers about as well as the GEMM anchor's own forward number
(13-18% error, same optimistic direction). Backward does **not** transfer:
the model *over*-predicts by ~2x here — the opposite direction from every
other backward measurement in this repo (GEMM anchor 0.86x, attention
anchor 0.15-0.17x, both under-predict). This isn't a *shape* problem —
this anchor's 4 shapes are a subset of `profile_step.py`'s own 6 — it's a
measurement-methodology one: `profile_step.py` timed each GEMM **in
isolation** and summed the results (its "backward 20.62 ms" number);
this anchor chains the same shapes through one real autograd graph, the
way a model actually trains, and the real chained number (7.40 ms) is
less than half the isolated-sum-calibrated prediction (15.26 ms). This is
where the whole isolated-vs-chained investigation started; the full
five-round trail (what was tried, what generalized, what didn't, and why)
lives in `docs/silicon_anchor.md`.

**The lever itself still checks out despite the absolute miss.** The
*delta* between stored and recompute — the actual thing `recompute=`
models — measures +0.2% and predicts +1.5%. Both are small and both say
the same thing: this config is compute-bound, recomputation barely costs
anything here. The 2x absolute error on backward affects both modes almost
identically, so it mostly cancels out of the difference — don't assume that
cancellation holds in general, it's one data point, not a proof.

## Limitations

* Recompute cost is modelled as exactly one forward GEMM per layer. Real
  checkpointing recomputes a *segment* between checkpoints, so the true cost
  depends on checkpoint spacing (here: every layer).
* No modelling of the extra SRAM pressure during the recompute itself.
* `X` for `wgrad` is this layer's input = the previous layer's output; the
  model charges *this* layer's forward as the proxy recompute cost.
