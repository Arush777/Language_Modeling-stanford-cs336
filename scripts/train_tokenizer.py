from __future__ import annotations

import argparse
import json
from pathlib import Path

from cs336_basics.train_bpe import train_bpe


def save_vocab_merges(out_dir: Path, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Store vocab bytes as hex so it's always JSON-safe
    vocab_json = {str(i): b.hex() for i, b in vocab.items()}
    (out_dir / "vocab.json").write_text(json.dumps(vocab_json), encoding="utf-8")

    # Store merges as hex pairs, one per line
    with (out_dir / "merges.txt").open("w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a.hex()} {b.hex()}\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--special", action="append", default=["<|endoftext|>"])
    args = p.parse_args()

    vocab, merges = train_bpe(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=args.special,
    )

    print("vocab_size:", len(vocab))
    print("num_merges:", len(merges))
    save_vocab_merges(args.out, vocab, merges)
    print("saved to:", args.out)


if __name__ == "__main__":
    main()
