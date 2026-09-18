# Work plan — 12 GiB GPU compatibility

Status: complete (2026-09-18)

1. Identify the RTX 3060 model-loading failure. Complete: Transformers allocator warmup required an additional allocation beyond 12 GiB.
2. Add an opt-in compatibility path without changing the published default. Complete: `--skip-allocator-warmup` skips only that startup optimization and records the choice in result metadata.
3. Validate and run on the target machine. Complete: the RTX 3060 produced `my-results.jsonl` from a 121-token direct decision.
4. Publish the documented change. Complete: README and reproduction guide include scope and command.
