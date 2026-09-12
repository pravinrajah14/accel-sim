# Importing real model shapes

*`simulator/import_model.py` — `layers_from_torch()`.*

## What it replaces

Every workload before this was hand-typed: `workloads.GPT2_BLOCK` exists
because someone read GPT-2's config and wrote six `Layer(...)` calls by
hand. That doesn't scale past one model. `layers_from_torch(model,
example_input)` traces an arbitrary `nn.Module` with `torch.fx`, walks the
graph for every `nn.Linear` it actually executes, and returns the `Layer`
list `model_step` already expects — no manual shape-copying.

```python
from simulator.import_model import layers_from_torch
layers = layers_from_torch(model, example_input)     # any nn.Module
total, parts = model_step(layers, hw, mem)
```

`torch.fx.symbolic_trace` gives the graph; `ShapeProp` runs the model once
on `example_input` (a real tensor, not a symbolic shape) to annotate every
node with its concrete output shape. A `Linear`'s input shape's leading
dims (everything but the last) collapse into `M`, matching how
`workloads.py` already flattens `batch * seq` into one token count.

## Proof it generalizes

`experiments/import_model_demo.py` traces a Llama-7B-shaped SwiGLU MLP
(`d_model=4096, d_ff=11008` — nobody hand-typed these) and gets exact
shapes back:

```
gate_proj    M=8192  K=4096  N=11008
up_proj      M=8192  K=4096  N=11008
down_proj    M=8192  K=11008  N=4096
```

then runs `model_step` on them like any other workload. Same code path,
zero new `Layer(...)` lines written.

## v1 scope

- **`nn.Linear` only** — matching the rest of accel-sim's GEMM-only
  coverage (README limitations). Conv layers, embeddings, and anything
  else in the traced graph are silently skipped.
- **Attention is not auto-detected.** There is no one canonical `nn.Module`
  for "multi-head attention" to pattern-match across model families (HF
  alone has several), and the shapes `Attention` needs — batch, seq,
  n_heads, d_head — aren't all recoverable from a `Linear`'s in/out features.
  Build one by hand (`simulator/attention.py`) and append it to the
  imported list, the way `workloads.GPT2_BLOCK_FULL` does.
- **Needs a real example input**, not a symbolic shape — `torch.fx` traces
  the *execution* your input actually takes, so a `Linear` behind a branch
  that input doesn't reach won't appear. This is a feature for models with
  real control flow (MoE routers, early-exit), not just a limitation.
- Not a core dependency — `pip install torch` separately, same convention
  as `validate/silicon/`. Every other module in `simulator/` still imports
  with nothing but `numpy`/`matplotlib`.

## Open

- Auto-detect common attention patterns (`nn.MultiheadAttention`,
  `F.scaled_dot_product_attention` call sites) where the shapes *are*
  recoverable from the traced tensors, falling back to manual `Attention`
  construction otherwise.
- Conv2d support, if a convolutional workload ever becomes relevant here.
