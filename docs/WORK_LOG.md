# Work log — 2026-09-18

- Installed the pinned project environment with mise-managed Python 3.10.
- Confirmed CUDA access on an RTX 3060 with 11.7 GiB VRAM.
- Reproduced the standard-load OOM. The model weights loaded, then Transformers' allocator warmup requested an additional 7.83 GiB.
- Added the opt-in `--skip-allocator-warmup` CLI option and result metadata marker.
- Found a concurrent `llama-server` using 8.46 GiB; after it was stopped, the direct example and one user decision completed successfully.
- Validation: `python -m pytest -q` (16 passed), `results/raw/SHA256SUMS` (all OK), and `benchmarks/verify_published.py` (69 summary claims verified).
- Confirmed the local llama.cpp router exposes `/apply-template`, `/tokenize`, and `/completion` with `n_probs`, `logit_bias`, and `post_sampling_probs`.
- Added experimental `--backend llama-server --server URL --mode direct` support. It validates option-label tokens through the server, applies equal positive logit bias, and normalizes the returned option probabilities.
- Ran all three owned example rows against the resident `lfm2.5-2.6b` GGUF model; every returned distribution summed to 1.0.
- Validation after the backend addition: `python -m pytest -q` (19 passed), `results/raw/SHA256SUMS` (all OK), and `benchmarks/verify_published.py` (69 summary claims verified).
