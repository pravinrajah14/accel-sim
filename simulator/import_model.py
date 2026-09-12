"""Import layer shapes from a real PyTorch model, instead of hand-typing them.

Every workload so far (`workloads.GPT2_BLOCK`) is hand-typed: someone read
GPT-2's config and wrote down six `Layer(...)` calls. That does not scale to
"point it at a model." This traces an arbitrary `nn.Module` with
`torch.fx` + a real example input (so shapes are concrete, not symbolic),
walks the graph for `nn.Linear` calls, and returns the `Layer` list
`workloads.py`/`dataflow.model_step` already expect.

    from simulator.import_model import layers_from_torch
    import torchvision.models as m           # or any nn.Module, incl. HF models
    model = SomeTransformerBlock(...)
    example = torch.randn(8, 1024, 768)       # (batch, seq, d_model)
    layers = layers_from_torch(model, example)
    total, parts = model_step(layers, hw, mem)

Needs `torch` (not a core accel-sim dependency -- `pip install torch`
separately, same as `validate/silicon/`).

v1 scope: `nn.Linear` only, matching the rest of accel-sim's GEMM-only
coverage. Attention is not auto-detected -- there is no one canonical
`nn.Module` for "multi-head attention" to pattern-match against across
model families, and the shapes it needs (batch, seq, n_heads, d_head) are
not all recoverable from a Linear's in/out features alone. Build an
`Attention(...)` by hand (`simulator/attention.py`) and append it, the way
`workloads.GPT2_BLOCK_FULL` does.
"""

from .dataflow import Layer


def layers_from_torch(model, example_input, *, only=None):
    """Trace `model` with `example_input` and return one `Layer` per
    `nn.Linear` the trace actually executes, in call order.

    `example_input` must be a real tensor (or tuple of tensors) shaped the
    way you'd actually call the model -- shapes come from *running* the
    trace, not from static analysis, so a `nn.Linear` behind a data-dependent
    branch that this input doesn't take will not appear.

    `only`: optional iterable of dotted module-name substrings to keep (e.g.
    `only=["attn", "mlp"]`) -- everything else (embeddings, the LM head,
    etc.) is skipped. Default: every `nn.Linear` in the trace.
    """
    try:
        import torch
        import torch.fx as fx
        from torch.fx.passes.shape_prop import ShapeProp
    except ImportError as e:
        raise ImportError(
            "layers_from_torch needs torch: pip install torch") from e

    model.eval()
    traced = fx.symbolic_trace(model)
    args = example_input if isinstance(example_input, tuple) else (example_input,)
    with torch.no_grad():
        ShapeProp(traced).propagate(*args)

    layers = []
    seen_names = set()
    for node in traced.graph.nodes:
        if node.op != "call_module":
            continue
        try:
            submod = traced.get_submodule(node.target)
        except AttributeError:
            continue
        if not isinstance(submod, torch.nn.Linear):
            continue
        name = str(node.target)
        if only is not None and not any(k in name for k in only):
            continue

        in_node = node.args[0]
        meta = in_node.meta.get("tensor_meta") if hasattr(in_node, "meta") else None
        if meta is None:
            continue  # couldn't recover a concrete shape for this call
        shape = tuple(meta.shape)

        m = 1
        for d in shape[:-1]:
            m *= d
        k = submod.in_features
        n = submod.out_features

        # de-duplicate a module called more than once (e.g. weight sharing)
        uniq = name
        i = 1
        while uniq in seen_names:
            i += 1
            uniq = f"{name}#{i}"
        seen_names.add(uniq)

        layers.append(Layer(uniq, M=m, K=k, N=n))

    if not layers:
        raise ValueError(
            "no nn.Linear found in the trace for this input -- check the "
            "model actually calls Linear layers on this input shape, or "
            "that `only` isn't filtering everything out")
    return layers
