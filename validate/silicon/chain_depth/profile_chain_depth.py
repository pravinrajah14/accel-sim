"""Silicon anchor: isolate chain DEPTH from shape and token count.

`profile_chain_grid.py`'s 4 configs varied depth, per-layer shape, and
token count all at once -- across them, the isolated-vs-chained "discount"
factor (measured backward / profile_step.py-calibrated prediction) did not
move monotonically with depth (2 layers: 0.36, 6 layers: 0.40, 8 layers:
0.29), so no depth-dependent term could be fit without confusing depth's
effect with shape's. This script holds shape (768x768, GPT-2's own proj
dimension) and token count (8192) fixed and varies ONLY depth (1/2/4/6/8
identical layers), chained as one real forward+backward+Adam call same as
profile_chain_grid.py -- never in isolation.

    python validate/silicon/profile_chain_depth.py --out chain_depth_profile.json

Then locally:

    python validate/silicon/compare_chain_depth.py chain_depth_profile.json
"""

import argparse
import json
import platform
import statistics
import sys

_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}

SHAPE = (768, 768)     # square -> any depth composes trivially
DEPTHS = [1, 2, 4, 6, 8]
TOKENS = 8 * 1024


def _bench(torch, nn, device, dtype, depth, M, iters, warmup):
    layers = [nn.Linear(*SHAPE, bias=False).to(device=device, dtype=dtype)
             for _ in range(depth)]
    params = [p for l in layers for p in l.parameters()]
    opt = torch.optim.Adam(params, lr=1e-4)
    x = torch.randn(M, SHAPE[0], device=device, dtype=dtype, requires_grad=True)

    fwd, bwd, optt = [], [], []
    for i in range(warmup + iters):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        ev[0].record()
        out = x
        for l in layers:
            out = l(out)
        ev[1].record()
        loss = out.float().square().mean()
        ev[2].record()
        loss.backward()
        ev[3].record()
        opt.step()
        opt.zero_grad(set_to_none=True)
        x.grad = None
        ev[4].record()
        torch.cuda.synchronize()
        if i >= warmup:
            fwd.append(ev[0].elapsed_time(ev[2]))
            bwd.append(ev[2].elapsed_time(ev[3]))
            optt.append(ev[3].elapsed_time(ev[4]))

    def stat(v):
        return {"mean_ms": statistics.fmean(v),
                "std_ms": statistics.pstdev(v) if len(v) > 1 else 0.0}
    return {"forward": stat(fwd), "backward": stat(bwd), "optimizer": stat(optt)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="chain_depth_profile.json")
    ap.add_argument("--dtype", choices=list(_DTYPES), default="fp16")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=15)
    args = ap.parse_args()

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        sys.exit("torch is required: pip install torch")
    if not torch.cuda.is_available():
        sys.exit("no CUDA device -- run this on a GPU box (Colab T4 is fine)")

    device = torch.device("cuda")
    dtype = getattr(torch, _DTYPES[args.dtype])
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}   dtype={args.dtype}   shape={SHAPE}   tokens={TOKENS}   "
          f"iters={args.iters} (+{args.warmup} warmup)")

    results = {}
    for depth in DEPTHS:
        r = _bench(torch, nn, device, dtype, depth, TOKENS, args.iters, args.warmup)
        results[str(depth)] = r
        print(f"  depth {depth:2d}  fwd {r['forward']['mean_ms']:8.3f}  "
              f"bwd {r['backward']['mean_ms']:8.3f}  "
              f"opt {r['optimizer']['mean_ms']:6.3f} ms")

    out = {
        "gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda,
        "dtype": _DTYPES[args.dtype], "shape": SHAPE, "tokens": TOKENS,
        "depths": DEPTHS,
        "iters": args.iters, "warmup": args.warmup,
        "results": results, "host": platform.platform(),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}  ->  run  python validate/silicon/compare_chain_depth.py {args.out}")


if __name__ == "__main__":
    main()
