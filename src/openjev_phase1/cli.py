"""Create-only JSONL command line scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import load_causal_model, validate_row
from .direct import score as direct_score
from .llama_server import LlamaServerClient, score as llama_server_score
from .reranker import score as reranker_score
from .serial import SerialPrefixScorer
from .shared import score_shared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "serial", "shared", "reranker"), required=True)
    parser.add_argument("--backend", choices=("transformers", "llama-server"), default="transformers")
    parser.add_argument("--server", help="llama-server base URL, required with --backend llama-server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--skip-allocator-warmup",
        action="store_true",
        help="Skip Transformers' load-time CUDA allocator warmup for memory-constrained GPUs.",
    )
    args = parser.parse_args()
    if args.output.exists() or args.max_tokens < 1:
        parser.error("Output must be new and max-tokens must be positive")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("Input is empty")
    for row in rows:
        validate_row(row)
    if args.backend == "llama-server":
        if args.mode != "direct":
            parser.error("llama-server currently supports only --mode direct")
        if not args.server:
            parser.error("--server is required with --backend llama-server")
        if args.skip_allocator_warmup:
            parser.error("--skip-allocator-warmup applies only to --backend transformers")
        client = LlamaServerClient(args.server, args.model)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as destination:
            for row in rows:
                destination.write(json.dumps(llama_server_score(client, row, args.max_tokens), allow_nan=False) + "\n")
                destination.flush()
        return
    if not args.revision:
        parser.error("--revision is required with --backend transformers")
    model, tokenizer, metadata = load_causal_model(
        args.model,
        args.revision,
        skip_allocator_warmup=args.skip_allocator_warmup,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        if args.mode == "shared":
            results, timing = score_shared(model, tokenizer, rows, metadata, args.max_tokens)
            for result in results:
                destination.write(json.dumps({**result, "shared_timing": timing}, allow_nan=False) + "\n")
        elif args.mode == "serial":
            scorer = SerialPrefixScorer(model, tokenizer, metadata, args.max_tokens)
            for row in rows:
                destination.write(json.dumps(scorer.score(row), allow_nan=False) + "\n")
                destination.flush()
        else:
            scorer = direct_score if args.mode == "direct" else reranker_score
            for row in rows:
                destination.write(json.dumps(scorer(model, tokenizer, row, metadata, args.max_tokens), allow_nan=False) + "\n")
                destination.flush()


if __name__ == "__main__":
    main()
