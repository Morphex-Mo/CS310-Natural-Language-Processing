#!/usr/bin/env python3
"""Memory-safe token counting for large corpora with a HuggingFace tokenizers JSON."""

import argparse
import json
import time
from pathlib import Path

from tokenizers import Tokenizer


def iter_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield line


def main():
    parser = argparse.ArgumentParser(description="Count tokens with streaming + batching")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer JSON")
    parser.add_argument("--input", required=True, help="Path to corpus text file")
    parser.add_argument("--batch_size", type=int, default=2000, help="Encoding batch size")
    parser.add_argument("--progress_every", type=int, default=200000, help="Progress interval in lines")
    parser.add_argument("--max_lines", type=int, default=0, help="Optional cap for quick test (0 = full file)")
    parser.add_argument("--output_json", default="token_stats.json", help="Where to write summary JSON")
    args = parser.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    corpus = Path(args.input)
    if not corpus.exists():
        raise FileNotFoundError(f"Input file not found: {corpus}")

    start = time.time()
    total_line_count = 0
    nonempty_line_count = 0
    char_count = 0
    token_count = 0
    batch = []

    for raw_line in iter_lines(corpus):
        total_line_count += 1
        char_count += len(raw_line)

        line = raw_line.strip()
        if line:
            nonempty_line_count += 1
            batch.append(line)

        if len(batch) >= args.batch_size:
            encs = tok.encode_batch(batch)
            token_count += sum(len(e.ids) for e in encs)
            batch.clear()

        if args.progress_every > 0 and total_line_count % args.progress_every == 0:
            elapsed = time.time() - start
            print(f"progress lines={total_line_count:,} tokens={token_count:,} elapsed={elapsed:.1f}s")

        if args.max_lines > 0 and total_line_count >= args.max_lines:
            break

    if batch:
        encs = tok.encode_batch(batch)
        token_count += sum(len(e.ids) for e in encs)

    elapsed = time.time() - start
    summary = {
        "input": str(corpus),
        "tokenizer": args.tokenizer,
        "total_lines": total_line_count,
        "nonempty_lines": nonempty_line_count,
        "nonspace_chars": char_count,
        "tokens": token_count,
        "elapsed_sec": round(elapsed, 3),
        "full_scan": args.max_lines == 0,
    }

    Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
