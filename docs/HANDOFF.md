# Handoff — 12 GiB GPU use

For short RTX 3060 workloads, activate `.venv310`, ensure competing GPU processes are stopped, and add `--skip-allocator-warmup` to `openjev-score`. The 4B BF16 model loads successfully in this mode; the output includes `"allocator_warmup": "skipped"` for auditability.

The browser demo was not usable on the target Linux/Chromium setup because the browser did not expose WebGPU `shader-f16`. This does not affect the Python CUDA path.

Do not use compatibility-mode timing as a comparison with the published RTX 3090 measurements. The model-loading behavior differs and the target GPU has different capacity and performance.

For a resident llama.cpp model, use `openjev-score --backend llama-server --server http://127.0.0.1:8080 --mode direct --model MODEL_ALIAS`. Only direct mode is supported. The backend calls the server's tokenizer and template APIs, so labels must be single tokens on that exact server model. It uses equal logit bias plus post-sampling option probabilities; outputs are GGUF/backend-dependent and must not be compared with the published Transformers measurements.
