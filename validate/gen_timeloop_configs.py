"""Generate fully-pinned Timeloop configs into validate/bench/.

Two families, mapspace collapsed to size 1 so the mapper just reports the
cycle count of the one pinned mapping:

  ws_<shape>_a<A>        -- forward GEMM, weight-stationary
      spatial  : C = A on meshX,  M = A on meshY
      pe_spad  : C = K/A,  M = N/A            (weight tiles held)
      glb      : Q = tokens                   (stream activations)

  os_wgrad_<shape>_a<A>  -- weight-gradient GEMM dW = X^T @ dY,
                           output-stationary
      Timeloop conv mapping:  C = tokens (contraction),
                              M = dW rows,  Q = dW cols
      spatial  : M = A on meshX,  Q = A on meshY   (output tile held)
      pe_spad  : M = rows/A,  Q = cols/A           (output tiles)
      glb      : C = tokens                        (stream both inputs)

Both need the mapped output dims divisible by A.

Usage:  python3 validate/gen_timeloop_configs.py [--arrays 64 128]
"""

import argparse
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "bench")
DESIGNS = os.path.join(BENCH, "example_designs")
WS_ARCH = os.path.join(DESIGNS, "simple_weight_stationary", "arch.yaml")
OS_ARCH = os.path.join(DESIGNS, "simple_output_stationary", "arch.yaml")
BASE_PROBLEM = os.path.join(DESIGNS, "_include", "default_problem.yaml")

TL_TOKENS = 1024
# forward GEMM shapes: (K_in, N_out)
FWD_SHAPES = {
    "proj_768x768": (768, 768),
    "mlp_up_768x3072": (768, 3072),
    "mlp_down_3072x768": (3072, 768),
}
# wgrad output (dW) shapes: (rows, cols)  -- rows=in_feat, cols=out_feat
WGRAD_SHAPES = {
    "wgrad_proj_768x768": (768, 768),
    "wgrad_mlp_up_768x3072": (768, 3072),
    "wgrad_mlp_down_3072x768": (3072, 768),
}
GLB_WORD_BITS = 64
SRAM_KB = 16384


def _f(**kw):
    dims = {"R": 1, "S": 1, "P": 1, "Q": 1, "N": 1, "C": 1, "M": 1}
    dims.update(kw)
    return "[" + ", ".join(f"{k}={v}" for k, v in dims.items()) + "]"


def _problem(C, M, Q):
    src = open(BASE_PROBLEM).read()
    _, _, shape = src.partition("  shape:")
    inst = (f"  instance:\n    N: 1\n    C: {C}\n    M: {M}\n    Q: {Q}\n"
            "    P: 1\n    R: 1\n    S: 1\n    H: 1\n    W: 1\n"
            "    Hpad: 0\n    Wpad: 0\n    Hstride: 1\n    Wstride: 1\n"
            "    Hdilation: 1\n    Wdilation: 1\n")
    return "problem:\n  version: 0.4\n" + inst + "  shape:" + shape


def _size_glb(src):
    depth = SRAM_KB * 1024 * 8 // GLB_WORD_BITS
    src = re.sub(r"(name: shared_glb.*?depth:\s*)\d+", rf"\g<1>{depth}",
                 src, flags=re.DOTALL)
    # the toy arch caps glb at 16 words/cycle; lift it so staging the streamed
    # operands in glb is not an artificial cycle bottleneck (we want the
    # traffic of this mapping, at MAC-rate cycles).
    src = re.sub(r"(name: shared_glb.*?read_bandwidth:\s*)\d+",
                 rf"\g<1>65536", src, flags=re.DOTALL)
    src = re.sub(r"(name: shared_glb.*?write_bandwidth:\s*)\d+",
                 rf"\g<1>65536", src, flags=re.DOTALL)
    return src


def make_ws(K, N, tokens, A):
    ct, mt = K // A, N // A
    src = open(WS_ARCH).read()
    src = re.sub(r"meshX:\s*\d+,\s*meshY:\s*\d+", f"meshX: {A}, meshY: {A}", src)
    src = re.sub(
        r"spatial:\s*\n\s*permutation: \[C, M\]\s*\n\s*split: 1\s*\n\s*factors: \[.*?\]",
        ("spatial:\n        permutation: [C, M, R, S, N, P, Q]\n"
         f"        split: 1\n        factors: {_f(C=A, M=A)}"), src)
    src = src.replace(
        "    constraints:\n"
        "      dataspace: {bypass: [Inputs, Outputs], keep: [Weights]}\n"
        "      temporal: {permutation: [P, Q, R, S]}",
        "    constraints:\n"
        "      dataspace: {bypass: [Inputs, Outputs], keep: [Weights]}\n"
        "      temporal:\n        permutation: [C, M, P, Q, R, S, N]\n"
        f"        factors: {_f(C=ct, M=mt)}")
    # shared_glb stages the streamed operands (Inputs, Outputs) so the pe_spad
    # weight-tile loop re-streams them from SRAM, not DRAM -> minimal traffic.
    src = src.replace(
        "      read_bandwidth: 16\n      write_bandwidth: 16\n",
        "      read_bandwidth: 16\n      write_bandwidth: 16\n"
        "    constraints:\n"
        "      dataspace: {keep: [Inputs, Outputs], bypass: [Weights]}\n"
        "      temporal:\n        permutation: [Q, C, M, R, S, P, N]\n"
        f"        factors: {_f(Q=tokens)}\n")
    src = _dram_pin(src)
    src = _size_glb(src)
    src = re.sub(r"(name: pe_spad.*?depth:\s*)\d+",
                 rf"\g<1>{max(192, ct * mt)}", src, flags=re.DOTALL)
    return src, _problem(K, N, tokens)


def make_os_wgrad(rows, cols, tokens, A):
    mt, qt = rows // A, cols // A
    src = open(OS_ARCH).read()
    src = re.sub(r"meshX:\s*\d+,\s*meshY:\s*\d+", f"meshX: {A}, meshY: {A}", src)
    # spatial: hold the M x Q output tile across the array
    src = re.sub(
        r"spatial:\s*\n\s*permutation: \[C, M\]\s*\n\s*split: 1\s*\n\s*factors: \[.*?\]",
        ("spatial:\n        permutation: [M, Q, R, S, N, P, C]\n"
         f"        split: 1\n        factors: {_f(M=A, Q=A)}"), src)
    # pe_spad keeps Outputs -> the M/A x Q/A output tiles
    src = src.replace(
        "    constraints:\n"
        "      dataspace: {bypass: [Inputs, Weights], keep: [Outputs]}\n"
        "      temporal: {permutation: [R, S, P, Q]}",
        "    constraints:\n"
        "      dataspace: {bypass: [Inputs, Weights], keep: [Outputs]}\n"
        "      temporal:\n        permutation: [M, Q, P, R, S, N, C]\n"
        f"        factors: {_f(M=mt, Q=qt)}")
    # shared_glb stages the two streamed operands (Weights = X, Inputs = dY)
    # so the pe_spad output-tile loop re-streams them from SRAM, not DRAM.
    src = src.replace(
        "      read_bandwidth: 16\n      write_bandwidth: 16\n",
        "      read_bandwidth: 16\n      write_bandwidth: 16\n"
        "    constraints:\n"
        "      dataspace: {keep: [Weights, Inputs], bypass: [Outputs]}\n"
        "      temporal:\n        permutation: [C, M, Q, R, S, P, N]\n"
        f"        factors: {_f(C=tokens)}\n")
    src = _dram_pin(src)
    src = _size_glb(src)
    src = re.sub(r"(name: pe_spad.*?depth:\s*)\d+",
                 rf"\g<1>{max(192, mt * qt)}", src, flags=re.DOTALL)
    return src, _problem(tokens, rows, cols)


def make_ws_wgrad(rows, cols, tokens, A, glb_kb):
    """Weight-stationary mapping of dW = X^T @ dY, with a chosen glb size.

    conv:  C = tokens (contraction),  M = cols (out_feat),  Q = rows (in_feat)
           Weights = C x M = dY   (held),  Inputs = C x Q = X,  Outputs = dW
    The contraction (tokens) is tiled into T pieces at DRAM so a dY slab + the
    full X + dW fit in glb; T = ceil(needed / glb).  Tests whether WS-wgrad
    DRAM traffic blows up as SRAM shrinks.
    """
    b = 2
    x_bytes = tokens * rows * b
    dw_bytes = rows * cols * b
    glb_bytes = glb_kb * 1024
    # a dY slab is (tokens/T) x cols; want slab + X + dW <= glb
    T = 1
    while (tokens // max(1, T)) * cols * b + x_bytes + dw_bytes > glb_bytes and T < tokens // A:
        T *= 2
    ct_dram = T
    ct_spad = tokens // A // T
    mt = cols // A

    src = open(WS_ARCH).read()
    src = re.sub(r"meshX:\s*\d+,\s*meshY:\s*\d+", f"meshX: {A}, meshY: {A}", src)
    src = re.sub(
        r"spatial:\s*\n\s*permutation: \[C, M\]\s*\n\s*split: 1\s*\n\s*factors: \[.*?\]",
        ("spatial:\n        permutation: [C, M, R, S, N, P, Q]\n"
         f"        split: 1\n        factors: {_f(C=A, M=A)}"), src)
    src = src.replace(
        "    constraints:\n"
        "      dataspace: {bypass: [Inputs, Outputs], keep: [Weights]}\n"
        "      temporal: {permutation: [P, Q, R, S]}",
        "    constraints:\n"
        "      dataspace: {bypass: [Inputs, Outputs], keep: [Weights]}\n"
        "      temporal:\n        permutation: [C, M, P, Q, R, S, N]\n"
        f"        factors: {_f(C=ct_spad, M=mt)}")
    src = src.replace(
        "      read_bandwidth: 16\n      write_bandwidth: 16\n",
        "      read_bandwidth: 16\n      write_bandwidth: 16\n"
        "    constraints:\n"
        "      dataspace: {keep: [Inputs, Outputs], bypass: [Weights]}\n"
        "      temporal:\n        permutation: [Q, C, M, R, S, P, N]\n"
        f"        factors: {_f(Q=rows)}\n")
    # DRAM tiles the contraction T ways
    src = src.replace(
        '    name: DRAM\n    class: DRAM\n    attributes:\n'
        '      type: "LPDDR4"\n      width: 64\n      datawidth: 16\n',
        '    name: DRAM\n    class: DRAM\n    attributes:\n'
        '      type: "LPDDR4"\n      width: 64\n      datawidth: 16\n'
        f"    constraints:\n      temporal: {{factors: {_f(C=ct_dram)}}}\n")
    # glb capacity for this test
    depth = glb_kb * 1024 * 8 // GLB_WORD_BITS
    src = re.sub(r"(name: shared_glb.*?depth:\s*)\d+", rf"\g<1>{depth}",
                 src, flags=re.DOTALL)
    src = re.sub(r"(name: shared_glb.*?read_bandwidth:\s*)\d+", r"\g<1>65536",
                 src, flags=re.DOTALL)
    src = re.sub(r"(name: shared_glb.*?write_bandwidth:\s*)\d+", r"\g<1>65536",
                 src, flags=re.DOTALL)
    src = re.sub(r"(name: pe_spad.*?depth:\s*)\d+",
                 rf"\g<1>{max(192, ct_spad * mt)}", src, flags=re.DOTALL)
    return src, _problem(tokens, cols, rows), T


def _dram_pin(src):
    return src.replace(
        '    name: DRAM\n    class: DRAM\n    attributes:\n'
        '      type: "LPDDR4"\n      width: 64\n      datawidth: 16\n',
        '    name: DRAM\n    class: DRAM\n    attributes:\n'
        '      type: "LPDDR4"\n      width: 64\n      datawidth: 16\n'
        f"    constraints:\n      temporal: {{factors: {_f()}}}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--staircase", action="store_true",
                    help="also emit ws-wgrad configs over a glb sweep")
    args = ap.parse_args()

    made = []

    if args.staircase:
        rows, cols = 768, 3072          # mlp_up wgrad
        tokens, A = 4096, 128
        for glb_kb in (1024, 2048, 4096, 8192, 16384, 32768):
            d = os.path.join(DESIGNS, f"wsw_mlp_up_g{glb_kb}")
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)
            arch, prob, T = make_ws_wgrad(rows, cols, tokens, A, glb_kb)
            open(f"{d}/arch.yaml", "w").write(arch)
            open(f"{d}/problem.yaml", "w").write(prob)
            made.append(f"wsw_mlp_up_g{glb_kb}  (contraction tiled x{T})")
        for m in made:
            print(m)
        return

    for name, (K, N) in FWD_SHAPES.items():
        for A in args.arrays:
            if K % A or N % A:
                continue
            d = os.path.join(DESIGNS, f"ws_{name}_a{A}")
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)
            arch, prob = make_ws(K, N, TL_TOKENS, A)
            open(f"{d}/arch.yaml", "w").write(arch)
            open(f"{d}/problem.yaml", "w").write(prob)
            made.append(f"ws_{name}_a{A}")

    for name, (rows, cols) in WGRAD_SHAPES.items():
        for A in args.arrays:
            if rows % A or cols % A:
                continue
            d = os.path.join(DESIGNS, f"os_{name}_a{A}")
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)
            arch, prob = make_os_wgrad(rows, cols, TL_TOKENS, A)
            open(f"{d}/arch.yaml", "w").write(arch)
            open(f"{d}/problem.yaml", "w").write(prob)
            made.append(f"os_{name}_a{A}")

    for m in made:
        print(m)
    print(f"\n{len(made)} configs -> validate/bench/example_designs/")


if __name__ == "__main__":
    main()
