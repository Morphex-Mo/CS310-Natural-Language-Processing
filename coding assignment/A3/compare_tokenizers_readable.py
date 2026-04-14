#!/usr/bin/env python3
"""Readable tokenizer comparison for Chinese text.

This does not change your original compare_tokenizers.py.
It explains GPT-2 byte-level tokens without showing replacement chars.
"""

from tokenizers import Tokenizer
from transformers import AutoTokenizer


def safe_gpt2_piece(tok, tid):
    """Show raw GPT-2 token text to avoid replacement-char decoding artifacts."""
    return tok.convert_ids_to_tokens([tid])[0]


def main():
    text = "太阳照常升起。"

    tokenizer_from_scratch = Tokenizer.from_file("wikizh_tokenizer_whitespace.json")
    tokenizer_gpt2_original = AutoTokenizer.from_pretrained("gpt2")

    ids_new = tokenizer_from_scratch.encode(text).ids
    pieces_new = [tokenizer_from_scratch.decode([tid]) for tid in ids_new]

    ids_gpt2 = tokenizer_gpt2_original.encode(text)
    pieces_gpt2 = [safe_gpt2_piece(tokenizer_gpt2_original, tid) for tid in ids_gpt2]

    print("Input:", text)
    print("Trained wikizh_tokenizer:", ids_new, pieces_new)
    print("Original GPT-2 ids:", ids_gpt2)
    print("Original GPT-2 raw tokens:", pieces_gpt2)
    print(
        "Note: GPT-2 is byte-level; many Chinese bytes are split across tokens, "
        "so standalone token decoding can look garbled."
    )


if __name__ == "__main__":
    main()
