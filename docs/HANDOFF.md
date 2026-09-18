# Handoff — 12 GiB GPU use

For short RTX 3060 workloads, activate `.venv310`, ensure competing GPU processes are stopped, and add `--skip-allocator-warmup` to `openjev-score`. The 4B BF16 model loads successfully in this mode; the output includes `"allocator_warmup": "skipped"` for auditability.

The browser demo was not usable on the target Linux/Chromium setup because the browser did not expose WebGPU `shader-f16`. This does not affect the Python CUDA path.

Do not use compatibility-mode timing as a comparison with the published RTX 3090 measurements. The model-loading behavior differs and the target GPU has different capacity and performance.
