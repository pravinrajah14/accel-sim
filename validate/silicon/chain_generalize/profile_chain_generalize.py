"""Silicon anchor: does the chain-depth fix generalize across shape-class
and token count?

`profile_chain_depth.py` found a clean fixed-overhead + per-layer-marginal
model for a 768x768 chain at M=8192 (R^2=0.97), traced to `_occupancy`'s
util_tiles penalty overstating wgrad's real chained cost for that SKINNY
shape. But applying that same fix to other configs made 2 of 4 worse:
`wide_shallow` (whose wgrad already has occupancy=1.0 under the OLD model
-- no penalty to relieve) was STILL measured ~2.8x faster than predicted,
proving a SEPARATE, occupancy-independent chain effect exists; and
`small_batch` (M=1024) blew up because the fixed-ms overhead swamps a
config whose whole measured time is under 1ms.

This script isolates those two remaining unknowns in one combined design:

  1. Depth sweep (1/2/4/8) on 2048x2048 -- a shape with occupancy=1.0 for
     BOTH dgrad and wgrad (verified: ceil(2048/143)^2 = 225 >= util_tiles),
     so occupancy never confounds this. If this ALSO shows a clean
     fixed-overhead + per-layer-marginal relationship, that overhead is a
     real, occupancy-independent, shape-class-transferable effect -- not
     an artifact of the 768x768-specific occupancy fix.
  2. Token-count sweep (512/1024/2048/4096/8192/16384) at fixed depth=4 on
     the same shape -- tests whether the fixed overhead scales with M
     (explaining small_batch's failure) or is genuinely M-independent.
  3. One repeated config (depth=4, M=8192, run again at the end) -- bounds
     how much of any "doesn't generalize" result is real physics vs.
     Colab session/tenancy noise, which hasn't been ruled out.

    python validate/silicon/profile_chain_generalize.py --out chain_generalize_profile.json

Then locally:

    python validate/silicon/compare_chain_generalize.py chain_generalize_profile.json
"""

import argparse
import json
import platform
import statistics
import sys

_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}

SHAPE = (2048, 2048)   # occupancy=1.0 for dgrad and wgrad both -- no confound

CONFIGS = [
    ("depth1_m8192",  1, 8192),
    ("depth2_m8192",  2, 8192),
    ("depth4_m8192",  4, 8192),
    ("depth8_m8192",  8, 8192),
    ("depth4_m512",   4, 512),
    ("depth4_m1024",  4, 1024),
    ("depth4_m2048",  4, 2048),
    ("depth4_m4096",  4, 4096),
    ("depth4_m16384", 4, 16384),
    ("repeat_depth4_m8192", 4, 8192),   # noise check -- same as depth4_m8192, run last
]


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
    ap.add_argument("--out", default="chain_generalize_profile.json")
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
    print(f"GPU: {gpu}   dtype={args.dtype}   shape={SHAPE}   "
          f"iters={args.iters} (+{args.warmup} warmup)")

    results = {}
    for name, depth, M in CONFIGS:
        r = _bench(torch, nn, device, dtype, depth, M, args.iters, args.warmup)
        results[name] = {"depth": depth, "tokens": M, **r}
        print(f"  {name:20s} fwd {r['forward']['mean_ms']:8.3f}  "
              f"bwd {r['backward']['mean_ms']:8.3f}  "
              f"opt {r['optimizer']['mean_ms']:6.3f} ms")

    out = {
        "gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda,
        "dtype": _DTYPES[args.dtype], "shape": SHAPE,
        "iters": args.iters, "warmup": args.warmup,
        "results": results, "host": platform.platform(),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}  ->  run  python validate/silicon/compare_chain_generalize.py {args.out}")


if __name__ == "__main__":
    main()
