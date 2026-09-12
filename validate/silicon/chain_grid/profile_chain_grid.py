"""Silicon anchor: real chained multi-layer training steps, several shapes.

Every other GEMM-level anchor in this repo before this one measured each
GEMM **in isolation** (`profile_step.py`'s `_bench_layer`: its own tensor,
its own optimizer, its own timing loop, called once per shape and summed).
A real model never trains that way -- it chains layers through one
autograd graph. Comparing `validate/silicon/profiles/t4_fp16.json`'s
per-layer isolated backward sum for the recompute anchor's 4 shapes
(q_proj + k_proj + mlp_up + mlp_down = 16.36 ms) against the real chained
measurement for those same shapes (`profiles/t4_fp16_recompute.json`'s
"stored" backward, 7.40 ms) shows isolated timing overstates real chained
backward by roughly 2x on this GPU -- and `compute_efficiency`/`util_tiles`
were calibrated against the isolated sum, so they inherit that bias.

This script re-anchors against **real chained** forward+backward+Adam,
across several shapes/depths/token counts, so `compare_chain_grid.py` can
refit `compute_efficiency`/`util_tiles` against the ground truth that
actually matters: a full training step, not a sum of isolated benchmarks.

    python validate/silicon/profile_chain_grid.py --out chain_grid_profile.json

Then locally:

    python validate/silicon/compare_chain_grid.py chain_grid_profile.json
"""

import argparse
import json
import platform
import statistics
import sys

_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}

# Each config is a real chained forward+backward+Adam step -- never timed in
# isolation. "gpt2_block_full" is the exact 6-GEMM shapes from
# simulator/workloads.py's GPT2_BLOCK (the original isolated GEMM anchor's
# own shapes), so it's a direct apples-to-apples replacement for that
# anchor's "step" ground truth. The other three vary depth/width/token
# count so the refit isn't a single-point fit.
CONFIGS = {
    "gpt2_block_full": {
        "dims": [(768, 768), (768, 768), (768, 768), (768, 768),
                 (768, 3072), (3072, 768)],
        "tokens": 8 * 1024,
    },
    "wide_shallow": {
        "dims": [(768, 4096), (4096, 768)],
        "tokens": 8 * 1024,
    },
    "narrow_deep": {
        "dims": [(512, 512)] * 8,
        "tokens": 8 * 1024,
    },
    "small_batch": {
        "dims": [(768, 768), (768, 768), (768, 3072), (3072, 768)],
        "tokens": 1024,
    },
}


def _bench(torch, nn, device, dtype, dims, M, iters, warmup):
    layers = [nn.Linear(k, n, bias=False).to(device=device, dtype=dtype)
             for k, n in dims]
    params = [p for l in layers for p in l.parameters()]
    opt = torch.optim.Adam(params, lr=1e-4)          # fp32 m,v state
    x = torch.randn(M, dims[0][0], device=device, dtype=dtype, requires_grad=True)

    fwd, bwd, optt = [], [], []
    for i in range(warmup + iters):
        # 5 events: bracket the loss reduction into "forward" (same fix as
        # profile_step.py / profile_recompute.py).
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
    ap.add_argument("--out", default="chain_grid_profile.json")
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
    print(f"GPU: {gpu}   dtype={args.dtype}   iters={args.iters} (+{args.warmup} warmup)")

    results = {}
    for name, cfg in CONFIGS.items():
        r = _bench(torch, nn, device, dtype, cfg["dims"], cfg["tokens"],
                  args.iters, args.warmup)
        results[name] = {"dims": cfg["dims"], "tokens": cfg["tokens"], **r}
        print(f"  {name:16s}  fwd {r['forward']['mean_ms']:8.3f}  "
              f"bwd {r['backward']['mean_ms']:8.3f}  "
              f"opt {r['optimizer']['mean_ms']:6.3f} ms")

    out = {
        "gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda,
        "dtype": _DTYPES[args.dtype],
        "iters": args.iters, "warmup": args.warmup,
        "configs": results, "host": platform.platform(),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}  ->  run  python validate/silicon/compare_chain_grid.py {args.out}")


if __name__ == "__main__":
    main()
