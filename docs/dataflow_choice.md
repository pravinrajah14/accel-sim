# Per-phase dataflow — retracted finding

## The claim (v5 of the model)

The weight-gradient GEMM `dW = Xᵀ·dY` has a tiny output (feature × feature)
and a huge contraction (the token count). An earlier version of
`simulator/memory.py` modelled weight-stationary traffic as
`held_operand + streamed·⌈held/sram⌉`, which said WS-`wgrad` thrashes badly
on its token-length operand — and that switching just `wgrad` to
output-stationary recovered **up to 34 % of the training step**.

## Why it was wrong

The traffic validation (`validate/`) built a weight-stationary `wgrad`
staircase — `dY` = 25 MB, sweeping SRAM — and asked Timeloop for the DRAM
traffic:

| SRAM | old model (ws wgrad) | Timeloop |
|-----:|---------------------:|---------:|
| 8 MB | 69 MB | **36 MB** |
| 16 MB | 47 MB | **36 MB** |
| 32 MB | 36 MB | 36 MB |

Timeloop hits **compulsory traffic (`X + dY + dW`) at 8 MB SRAM**, far below
the 25 MB `dY`. It tiles the *contraction* (token) dimension: process the
tokens in chunks, keep the small `dW` accumulator resident, stream a `dY`- and
`X`-slab per chunk. No thrash.

The old model's `⌈held/sram⌉` refetch assumed you hold the whole `dY` and
re-stream everything when it doesn't fit. A real scheduler doesn't do that
for `wgrad` — it tiles the contraction.

## What's true instead

* **DRAM traffic does not depend on the dataflow.** For a well-scheduled
  GEMM it is `compulsory + tiling-penalty`, and the penalty is 0 whenever the
  *smallest* of the three tensors fits in SRAM (you keep that one resident
  and stream the other two). `wgrad`'s smallest tensor is `dW`; forward's is
  the weight matrix — both small — so neither phase thrashes at realistic
  SRAM sizes.
* The dataflow choice moves **~1–3 % of cycles** (`matmul_cycles` still
  models it: `ws` overhead `K + rows + cols`, `os` overhead `rows + cols`).
  For `wgrad` that is ~3 % fewer cycles with output-stationary — real but
  not a headline.

## What replaced it

`simulator/memory.py::gemm_traffic` — compulsory + a
`≈ 2·M·N·K·b / √(S/2b)` tiling penalty, applied only when the smallest
tensor plus a working strip does not fit. The compulsory term is
**Timeloop-validated exact** (0 % error, 12 configs); the penalty term is a
first-order heuristic (limitation).

## Lesson

This is the value of validation: the finding was specific, defensible-
sounding, and wrong, and the cross-check caught it before it went anywhere.
The other levers (`optimizer_state`, `recomputation`) operate on the
compulsory term and survived the revision. A later, unrelated lever
(`backward_reuse`) operated on this same term too, but was removed for a
different reason — not disproven, just untestable on any real GPU (see
`results/RESULTS.md` §9).
