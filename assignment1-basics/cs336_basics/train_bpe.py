from __future__ import annotations

from collections import Counter, defaultdict
import os
import regex

# Same GPT-2 pre-tokenization regex you used in the tokenizer.
GPT2_PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PRETOK_RE = regex.compile(GPT2_PAT)


def _compile_special_re(special_tokens: list[str]) -> regex.Pattern | None:
    if not special_tokens:
        return None
    # Longest-first handles overlapping specials deterministically.
    special_alt = "|".join(regex.escape(s) for s in sorted(special_tokens, key=len, reverse=True))
    return regex.compile(special_alt)


def _iter_chunks_split_on_specials(text: str, special_re: regex.Pattern | None) -> list[str]:
    """
    Return a list of chunks where special tokens appear as their own elements,
    and non-special spans remain intact.
    """
    if special_re is None:
        return [text]

    out: list[str] = []
    pos = 0
    for m in special_re.finditer(text):
        if m.start() > pos:
            out.append(text[pos:m.start()])
        out.append(m.group(0))  # the special token itself
        pos = m.end()
    if pos < len(text):
        out.append(text[pos:])
    return out


def _word_from_piece(piece: str, is_special: bool) -> tuple[bytes, ...]:
    if is_special:
        # Represent the entire special token as ONE symbol so it never merges with anything.
        return (piece.encode("utf-8"),)
    b = piece.encode("utf-8")
    return tuple(bytes([x]) for x in b)  # start from single-byte symbols


def _pairs_list(word: tuple[bytes, ...]) -> list[tuple[bytes, bytes]]:
    return [(word[i], word[i + 1]) for i in range(len(word) - 1)]


def _merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    a, b = pair
    out: list[bytes] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def _init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    # Reference vocab puts special tokens first; do the same.
    vocab: dict[int, bytes] = {}
    idx = 0
    for s in special_tokens:
        vocab[idx] = s.encode("utf-8")
        idx += 1
    for b in range(256):
        vocab[idx] = bytes([b])
        idx += 1
    return vocab


def _build_word_counts(
    input_path: str | os.PathLike,
    special_tokens: list[str],
) -> Counter[tuple[bytes, ...]]:
    """
    1) Read corpus
    2) Split around special tokens so merges can't cross them
    3) GPT-2 pretokenize each non-special span
    4) Convert each pretoken into tuple-of-byte-symbols and count
    """
    special_re = _compile_special_re(special_tokens)
    special_set = set(special_tokens)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    word_counts: Counter[tuple[bytes, ...]] = Counter()

    for chunk in _iter_chunks_split_on_specials(text, special_re):
        if not chunk:
            continue

        if chunk in special_set:
            word_counts[_word_from_piece(chunk, is_special=True)] += 1
            continue

        for m in _PRETOK_RE.finditer(chunk):
            piece = m.group(0)
            if piece:
                word_counts[_word_from_piece(piece, is_special=False)] += 1

    return word_counts


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # ✅ Your bug: you wrote vocab_size - 256 - special_tokens
    # It must be vocab_size - 256 - len(special_tokens)
    num_merges = vocab_size - 256 - len(special_tokens)
    if num_merges < 0:
        raise ValueError("vocab_size too small for 256 bytes + special tokens")

    vocab = _init_vocab(special_tokens)
    word_counts = _build_word_counts(input_path, special_tokens)

    # Track pair frequencies and which word-types contain each pair (for speed)
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: defaultdict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

    def add_word_contrib(word: tuple[bytes, ...], freq: int) -> None:
        pairs = _pairs_list(word)
        for p in pairs:
            pair_counts[p] += freq
        for p in set(pairs):
            pair_to_words[p].add(word)

    def remove_word_contrib(word: tuple[bytes, ...], freq: int) -> None:
        pairs = _pairs_list(word)
        for p in pairs:
            pair_counts[p] -= freq
            if pair_counts[p] <= 0:
                pair_counts.pop(p, None)
        for p in set(pairs):
            s = pair_to_words.get(p)
            if s is None:
                continue
            s.discard(word)
            if not s:
                pair_to_words.pop(p, None)

    # Initialize counts
    for w, c in word_counts.items():
        if len(w) >= 2:
            add_word_contrib(w, c)

    merges_out: list[tuple[bytes, bytes]] = []

    for _ in range(num_merges):
        if not pair_counts:
            break

        # ✅ Deterministic tie-break: (count, pair) and choose max
        best_pair = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        merges_out.append(best_pair)

        # New token bytes = concatenation
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        affected = list(pair_to_words.get(best_pair, ()))
        if not affected:
            pair_counts.pop(best_pair, None)
            continue

        new_entries: Counter[tuple[bytes, ...]] = Counter()

        # Update only affected word-types
        for w in affected:
            freq = word_counts.get(w)
            if not freq:
                continue
            remove_word_contrib(w, freq)
            word_counts.pop(w, None)

            merged_w = _merge_word(w, best_pair)
            new_entries[merged_w] += freq

        for new_w, freq in new_entries.items():
            prev = word_counts.get(new_w, 0)
            word_counts[new_w] = prev + freq
            if len(new_w) >= 2:
                add_word_contrib(new_w, freq)

    return vocab, merges_out
