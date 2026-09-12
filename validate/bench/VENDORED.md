# Vendored Timeloop example design

This directory is a **trimmed copy** of
[`Accelergy-Project/timeloop-accelergy-exercises`](https://github.com/Accelergy-Project/timeloop-accelergy-exercises),
used as the reference architecture for the forward-compute cross-check in
`../compare.py`.

- **Upstream commit:** `2d5510807128e9bd5f1cad0705cdf2ec4612fd4e` (2025-04-09)
- **Path:** `workspace/example_designs/`
- **License:** MIT (Joel Emer, 2019) — see `LICENSE.upstream`

## What was kept

- `example_designs/simple_weight_stationary/arch.yaml` — the weight-stationary
  systolic-array architecture
- `example_designs/simple_output_stationary/arch.yaml` — the output-stationary
  variant (used for the `wgrad` cross-check)
- `example_designs/_components/`, `example_designs/_include/` — compound
  component definitions + default problem/mapper
- `example_designs/top.yaml.jinja2` — the spec assembler
- `run_example_designs.py`, `util_functions.py` — the driver

## What was removed

All other example architectures (eyeriss, simba, sparseloop, …), every
`ref_outputs/` tree, and every bundled layer-shape set. The other example
designs and the DNN layer shapes are not needed for this cross-check.

## Local modifications

- `example_designs/_include/mapper.yaml` — search settings tuned for a
  size-1 (fully pinned) mapspace.
- `example_designs/{ws,os_wgrad}_<shape>_a<array>/` dirs are **generated** by
  `../gen_timeloop_configs.py` (git-ignored).
