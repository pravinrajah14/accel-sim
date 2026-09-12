"""Compare simulator/ cycle + traffic estimates to Timeloop's, report error.

Two families of pinned (size-1 mapspace) Timeloop configs
(see gen_timeloop_configs.py):

  ws_<shape>_a<A>       forward GEMM, weight-stationary
  os_wgrad_<shape>_a<A> weight-gradient GEMM, output-stationary

Both stage the streamed operands in the global buffer, so each config's
mapping is the minimal-traffic one at that shape.  We check:

  * Cycles  vs  matmul_cycles(..., dataflow)   (dataflow still matters ~1% for cycles)
  * DRAM bytes (sum of per-tensor Scalar reads + updates x 2) vs
    gemm_traffic(...).total_bytes   (dataflow-independent)

Writes results/validation/validation.csv.
"""

import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator.compute import Hardware, matmul_cycles, ideal_mac_cycles  # noqa: E402
from simulator.memory import Memory, gemm_traffic  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DESIGNS = os.path.join(HERE, "bench", "example_designs")
RESULT_CSV = os.path.join(os.path.dirname(HERE), "results", "validation", "validation.csv")
WORD_BYTES = 2

TOK = 1024
FWD = {"proj_768x768": (768, 768), "mlp_up_768x3072": (768, 3072),
       "mlp_down_3072x768": (3072, 768)}
WGRAD = {"wgrad_proj_768x768": (768, 768), "wgrad_mlp_up_768x3072": (768, 3072),
         "wgrad_mlp_down_3072x768": (3072, 768)}

WS_RE = re.compile(r"^ws_(?P<shape>.+)_a(?P<arr>\d+)$")
OS_RE = re.compile(r"^os_(?P<shape>wgrad_.+)_a(?P<arr>\d+)$")
WSW_RE = re.compile(r"^wsw_mlp_up_g(?P<glb>\d+)$")   # ws-wgrad staircase
CYCLE_RE = re.compile(r"^\s*Cycles\s*:\s*(\d+)", re.MULTILINE)


def _cycles(stats):
    m = CYCLE_RE.search(open(stats).read())
    return int(m.group(1)) if m else None


def _dram_bytes(stats):
    """Sum per-tensor DRAM Scalar reads + updates, x word size."""
    txt = open(stats).read()
    seg = txt.split("=== DRAM ===", 1)
    if len(seg) < 2:
        return None
    seg = re.split(r"^Level \d+", seg[1], 1, re.MULTILINE)[0]
    words = 0
    for kind in ("reads", "updates"):
        for m in re.finditer(rf"Scalar {kind} \(per-instance\)\s*:\s*(\d+)", seg):
            words += int(m.group(1))
    return words * WORD_BYTES


def main():
    rows = []
    for stats in sorted(glob.glob(os.path.join(
            DESIGNS, "*", "outputs", "*", "timeloop-mapper.stats.txt"))):
        base = os.path.basename(stats.split(os.sep + "outputs" + os.sep)[0])
        tl_cyc = _cycles(stats)
        tl_bytes = _dram_bytes(stats)
        if tl_cyc is None:
            continue

        mw = WSW_RE.match(base)
        m = OS_RE.match(base)
        if mw:
            arr, fam, label, df = 128, "wsw-stair", f"g{int(mw['glb'])//1024}MiB", "ws"
            M, N, K = 768, 3072, 4096          # mlp_up wgrad, tokens=4096
            mem = Memory(sram_bytes=int(mw["glb"]) * 1024, dram_bandwidth_bytes_s=1e12)
        elif m and m["shape"] in WGRAD:
            r, c = WGRAD[m["shape"]]
            arr, fam, label, df = int(m["arr"]), "os-wgrad", m["shape"], "os"
            M, N, K = r, c, TOK
            mem = Memory(sram_bytes=1 << 30, dram_bandwidth_bytes_s=1e12)
        else:
            m = WS_RE.match(base)
            if not m or m["shape"] not in FWD:
                continue
            K, N = FWD[m["shape"]]
            arr, fam, label, df = int(m["arr"]), "ws-fwd", m["shape"], "ws"
            M, N, K = TOK, N, K
            mem = Memory(sram_bytes=1 << 30, dram_bandwidth_bytes_s=1e12)

        hw = Hardware(rows=arr, cols=arr, clock_hz=1e9)
        our_cyc = matmul_cycles(M, N, K, hw, df)
        our_bytes = gemm_traffic(M, N, K, hw, mem).total_bytes
        rows.append(dict(
            family=fam, shape=label, array=arr, tokens=TOK,
            tl_cycles=tl_cyc, model_cycles=our_cyc,
            cyc_err=round(100.0 * (our_cyc - tl_cyc) / tl_cyc, 2),
            tl_dram_bytes=tl_bytes, model_dram_bytes=int(our_bytes),
            dram_err=(round(100.0 * (our_bytes - tl_bytes) / tl_bytes, 2)
                      if tl_bytes else None)))

    if not rows:
        print("no Timeloop results. Run validate/run_timeloop.sh first.")
        return

    os.makedirs(os.path.dirname(RESULT_CSV), exist_ok=True)
    with open(RESULT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'family':9s} {'shape':22s} {'arr':>4s}  "
          f"{'cyc err':>8s}  {'DRAM MB (tl / model)':>22s}  {'err':>7s}")
    print("-" * 82)
    for r in sorted(rows, key=lambda x: (x["family"], x["shape"], x["array"])):
        tl_mb = r["tl_dram_bytes"] / 1e6 if r["tl_dram_bytes"] else 0
        md_mb = r["model_dram_bytes"] / 1e6
        de = f"{r['dram_err']:+.1f}%" if r["dram_err"] is not None else "  -  "
        print(f"{r['family']:9s} {r['shape']:22s} {r['array']:>4d}  "
              f"{r['cyc_err']:>+7.2f}%  {tl_mb:>9.2f} / {md_mb:<9.2f}  {de:>7s}")
    print("-" * 82)
    for fam in ("ws-fwd", "os-wgrad", "wsw-stair"):
        fr = [r for r in rows if r["family"] == fam]
        if not fr:
            continue
        cyc = sum(abs(r["cyc_err"]) for r in fr) / len(fr)
        dr = [abs(r["dram_err"]) for r in fr if r["dram_err"] is not None]
        drm = f"{sum(dr)/len(dr):.1f}%" if dr else "n/a"
        print(f"{fam:9s}  cycles MAPE {cyc:.1f}%   DRAM MAPE {drm}   (n={len(fr)})")
    print(f"wrote {RESULT_CSV}")


if __name__ == "__main__":
    main()
