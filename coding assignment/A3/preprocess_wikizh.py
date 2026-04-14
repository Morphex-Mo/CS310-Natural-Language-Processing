#!/usr/bin/env python3
"""Extract and preprocess CLUECorpus2020 Chinese Wikipedia JSONL files."""

import argparse
import json
from pathlib import Path


def iter_source_files(input_path: Path):
    if input_path.is_file():
        yield input_path
        return
    for file_path in sorted(input_path.rglob("*")):
        if file_path.is_file():
            yield file_path


def extract_text_from_jsonl(file_path: Path):
    kept = 0
    skipped = 0
    lines = []

    with file_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                skipped += 1
                continue

            title = str(obj.get("title", "")).strip()
            text = str(obj.get("text", "")).strip()
            merged = " ".join(part for part in [title, text] if part)
            if merged:
                lines.append(merged)
                kept += 1

    return lines, kept, skipped


def main():
    parser = argparse.ArgumentParser(description="Extract title+text from CLUE wiki JSONL data")
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--output", default="wikizh.txt", help="Output concatenated text file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_kept = 0
    total_skipped = 0

    with output_path.open("w", encoding="utf-8") as out:
        for src in iter_source_files(input_path):
            total_files += 1
            lines, kept, skipped = extract_text_from_jsonl(src)
            for line in lines:
                out.write(line + "\n")
            total_kept += kept
            total_skipped += skipped

    print(f"Processed files: {total_files}")
    print(f"Kept records: {total_kept}")
    print(f"Skipped invalid JSON lines: {total_skipped}")
    print(f"Saved extracted corpus to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
