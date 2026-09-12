# Closed-form systems autotuner

*`simulator/autotune.py` — `autotune()` and `pareto()`.*

## What it does

Given a model + hardware, exhaustively search the systems-lever space for the
minimum-latency training step — optionally under a stored-activation budget —
and return the plan.

**Lever space** (all discrete):

| lever | options |
|-------|---------|
| optimizer state | fp32 streamed / fp32 resident / bf16 resident / fp8 resident |
| activation recomputation | off / on |
| microbatch size | tokens / 2 / 4 / 8 / 16 |

= **4 × 2 × 5 = 40 configs.** Every evaluation is a closed-form
`model_step` (~µs), so the full exhaustive search runs in **~1 ms** — less
than one generation of MONET's NSGA-II genetic algorithm, and covering three
interacting levers instead of one. (A fourth lever, backward-pass reuse, was
in this search until it was removed from the model entirely — untestable on
any GPU on hand; see `results/RESULTS.md` §9. The search dropped from 80 to
40 configs accordingly, not just this doc's numbers.)

## Why this is the point

MONET explores *one* lever (activation checkpointing) with a genetic
algorithm because each evaluation goes through Stream's mapping search
(seconds). A closed-form model flips the economics: the search is free, so
you brute-force everything and get the exact optimum, not a Pareto
approximation, instantly. It's fast enough to sit *inside* a training
framework's startup.

## What it reveals

`experiments/autotune.py` — the picked plan is **config-dependent** (that's
why you autotune):

| hardware | plan | step |
|----------|------|-----:|
| 128² / 2 MiB / 50 GB/s (edge) | **fp8** resident, **recompute**, no accum | 28.4 ms |
| 128² / 8 MiB / 100 GB/s | **fp8** resident, store acts, no accum | 12.1 ms |
| 256² / 16 MiB / 100 GB/s | **bf16** resident, **recompute**, no accum | 8.9 ms |
| 256² / 64 MiB / 400 GB/s (generous) | **fp32** resident, store acts, no accum | 3.0 ms |

The pattern the model produces on its own:
- **tight SRAM** → aggressive optimizer precision (fp8), to fit
- **big array / low bandwidth** → recomputation on (spare FLOPs, memory-bound)
- **generous SRAM + bandwidth** → keep it simple, just resident fp32 optimizer

Under an activation budget it trades minimally: at 256²/16 MiB, dropping the
budget from 64 MB → 32 MB costs +0.6 ms (adds 2× accumulation).

`pareto()` returns the full latency vs stored-activation-memory frontier
(40 configs → a handful of Pareto-optimal points).

## Limitations

- Inherits every limitation of the underlying model (heuristic traffic
  penalty, no phase overlap, no elementwise-compute pricing).
- The lever grid is coarse (5 microbatch sizes, 4 optimizer configs); finer
  grids are still sub-10 ms but not implemented.
- Objective is min-latency; a real autotuner would also weigh energy and
  convergence effects of batch size, which the model does not capture.
- Attention fusion (a real, silicon-anchored lever — `docs/attention.md`) is
  not yet in this search; the pareto/autotune calls only see `Layer` items.
