# Work log — 2026-09-18

- Installed the pinned project environment with mise-managed Python 3.10.
- Confirmed CUDA access on an RTX 3060 with 11.7 GiB VRAM.
- Reproduced the standard-load OOM. The model weights loaded, then Transformers' allocator warmup requested an additional 7.83 GiB.
- Added the opt-in `--skip-allocator-warmup` CLI option and result metadata marker.
- Found a concurrent `llama-server` using 8.46 GiB; after it was stopped, the direct example and one user decision completed successfully.
- Validation: `python -m pytest -q` (16 passed), `results/raw/SHA256SUMS` (all OK), and `benchmarks/verify_published.py` (69 summary claims verified).
