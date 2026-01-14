from __future__ import annotations

import argparse
import json
from array import array
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer


def load_vocab_merges(tok_dir: Path) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab_json = json.loads((tok_dir / "vocab.json").read_text(encoding="utf-8"))
    vocab = {int(k): bytes.fromhex(v_hex) for k, v_hex in vocab_json.items()}

    merges: list[tuple[bytes, bytes]] = []
    for line in (tok_dir / "merges.txt").read_text(encoding="utf-8").splitlines():
        a_hex, b_hex = line.split()
        merges.append((bytes.fromhex(a_hex), bytes.fromhex(b_hex)))

    return vocab, merges


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tok", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--special", action="append", default=["<|endoftext|>"])
    p.add_argument("--flush", type=int, default=1_000_000)
    args = p.parse_args()

    vocab, merges = load_vocab_merges(args.tok)
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=args.special)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # uint32 is safe for any reasonable vocab size
    buf = array("I")
    total = 0

    with args.input.open("r", encoding="utf-8", errors="replace") as fin, args.output.open("wb") as fout:
        for tid in tokenizer.encode_iterable(fin):
            buf.append(tid)
            total += 1
            if len(buf) >= args.flush:
                buf.tofile(fout)
                buf = array("I")
        if buf:
            buf.tofile(fout)

    meta = {"dtype": "uint32", "count": total}
    (args.output.with_suffix(args.output.suffix + ".meta.json")).write_text(
        json.dumps(meta), encoding="utf-8"
    )
    print("wrote", total, "token ids to", args.output)


if __name__ == "__main__":
    main()
