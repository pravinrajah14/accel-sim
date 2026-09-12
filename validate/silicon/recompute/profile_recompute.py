"""Silicon anchor for activation recomputation (checkpointing).

The model's claim (`docs/recomputation.md`): recomputing a layer's input X
during backward costs about one extra forward GEMM, traded for one DRAM
read of X. Whether that's a win depends on whether the layer is compute- or
memory-bound. This measures the real trade with `torch.utils.checkpoint` --
PyTorch's actual implementation of the exact same idea -- on a real 4-layer
chain using GPT-2's own GEMM shapes (768->768->768->3072->768), and checks
the measured backward-time delta against what `training_step(recompute=)`
predicts.

    python validate/silicon/profile_recompute.py --out recompute_profile.json

Then locally:

    python validate/silicon/compare_recompute.py recompute_profile.json
"""

import argparse
import json
import platform
import statistics
import sys

_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}

# GPT-2's own shapes, chained sequentially so torch.utils.checkpoint has a
# real multi-layer graph to checkpoint across (mirrors q_proj -> attn_out ->
# mlp_up -> mlp_down; matches simulator/workloads.py's dimensions).
CHAIN_DIMS = [(768, 768), (768, 768), (768, 3072), (3072, 768)]


def _bench(torch, nn, checkpoint, device, dtype, M, iters, warmup, use_checkpoint):
    layers = [nn.Linear(k, n, bias=False).to(device=device, dtype=dtype)
             for k, n in CHAIN_DIMS]
    params = [p for l in layers for p in l.parameters()]
    x = torch.randn(M, CHAIN_DIMS[0][0], device=device, dtype=dtype, requires_grad=True)

    def forward_stored():
        out = x
        for l in layers:
            out = l(out)
        return out

    def forward_checkpointed():
        out = x
        for l in layers:
            out = checkpoint.checkpoint(l, out, use_reentrant=False)
        return out

    forward_fn = forward_checkpointed if use_checkpoint else forward_stored

    fwd, bwd = [], []
    for i in range(warmup + iters):
        for p in params:
            p.grad = None
        x.grad = None
        # 4 events: bracket the loss reduction into "forward" (see
        # profile_step.py's docstring note on this same fix).
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        ev[0].record()
        out = forward_fn()
        ev[1].record()
        loss = out.float().square().mean()
        ev[2].record()
        loss.backward()
        ev[3].record()
        torch.cuda.synchronize()
        if i >= warmup:
            fwd.append(ev[0].elapsed_time(ev[2]))
            bwd.append(ev[2].elapsed_time(ev[3]))

    def stat(v):
        return {"mean_ms": statistics.fmean(v),
                "std_ms": statistics.pstdev(v) if len(v) > 1 else 0.0}
    return {"forward": stat(fwd), "backward": stat(bwd)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="recompute_profile.json")
    ap.add_argument("--dtype", choices=list(_DTYPES), default="fp16")
    ap.add_argument("--tokens", type=int, default=8 * 1024)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=15)
    args = ap.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.utils.checkpoint as checkpoint
    except ImportError:
        sys.exit("torch is required: pip install torch")
    if not torch.cuda.is_available():
        sys.exit("no CUDA device -- run this on a GPU box (Colab T4 is fine)")

    device = torch.device("cuda")
    dtype = getattr(torch, _DTYPES[args.dtype])
    M = args.tokens
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}   dtype={args.dtype}   tokens={M}   chain={CHAIN_DIMS}   "
          f"iters={args.iters} (+{args.warmup} warmup)")

    stored = _bench(torch, nn, checkpoint, device, dtype, M, args.iters, args.warmup,
                    use_checkpoint=False)
    print(f"  stored       fwd {stored['forward']['mean_ms']:8.3f}  "
          f"bwd {stored['backward']['mean_ms']:8.3f} ms")
    recompute = _bench(torch, nn, checkpoint, device, dtype, M, args.iters, args.warmup,
                       use_checkpoint=True)
    print(f"  recompute    fwd {recompute['forward']['mean_ms']:8.3f}  "
          f"bwd {recompute['backward']['mean_ms']:8.3f} ms")
    d_total = ((recompute["forward"]["mean_ms"] + recompute["backward"]["mean_ms"]) -
              (stored["forward"]["mean_ms"] + stored["backward"]["mean_ms"]))
    stored_total = stored["forward"]["mean_ms"] + stored["backward"]["mean_ms"]
    print(f"  measured delta: {d_total:+.3f} ms ({d_total / stored_total * 100:+.1f}%)")

    out = {
        "gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda,
        "dtype": _DTYPES[args.dtype], "tokens": M, "chain_dims": CHAIN_DIMS,
        "iters": args.iters, "warmup": args.warmup,
        "stored": stored, "recompute": recompute, "host": platform.platform(),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}  ->  run  python validate/silicon/compare_recompute.py {args.out}")


if __name__ == "__main__":
    main()
