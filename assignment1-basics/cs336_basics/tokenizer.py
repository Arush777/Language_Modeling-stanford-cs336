from __future__ import annotations

from collections.abc import Iterable, Iterator
import regex


class Tokenizer:
    """
    GPT-2 style BPE tokenizer:
      - regex pre-tokenization
      - byte-level BPE merges (provided as bytes pairs)
      - special tokens that bypass BPE
    """

    GPT2_PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self,vocab: dict[int, bytes],merges: list[tuple[bytes, bytes]],special_tokens: list[str] | None = None):
        
        self.id_to_bytes=dict(vocab)
        self.bytes_to_id={b:i for i,b in self.id_to_bytes.items()}
        # merge rank: earlier in merges list = lower rank number = higher priority
        self.merge_ranks = {pair: r for r, pair in enumerate(merges)} #{pair:next_bit}(r is index number) therefore #merge_ranks[a,b]=index is what we are doing hence lowest rank is smallest index in karpathy merge list was 72,154=255 here we have byte strings b'H',b'l' in pairs etc.
        #list[tuple[bytes, bytes]]) -> dict[tuple[bytes, bytes], int]
        
        self.cache = {}
        
        self._pretok_re = regex.compile(self.GPT2_PAT)


        self.special_tokens = list(special_tokens or [])
        self.special_set = set(self.special_tokens)

        # map special token string -> id (specials are stored in vocab as UTF-8 bytes)
        self.special_to_id = {}
        for s in self.special_tokens:
            b = s.encode("utf-8")
            if b in self.bytes_to_id:
                self.special_to_id[s] = self.bytes_to_id[b]

        # One combined regex:
        # - Special tokens first (longest-first so overlapping specials work)
        # - Then GPT-2 pattern
        if self.special_tokens:
            special_alt = "|".join(
                regex.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True)
            )
            self._special_re=regex.compile(special_alt)
            self._max_special_len = max(len(s) for s in self.special_tokens)
        else:
            self._special_re = None
            self._max_special_len = 0

        self._bpe_cache: dict[bytes, tuple[bytes, ...]] = {}

    def _bpe(self, token_bytes: bytes) -> tuple[bytes, ...]:
        """
        Apply BPE merges to a single pre-token piece (in bytes).
        Returns the final list of merged byte-strings.
        """
        cached = self._bpe_cache.get(token_bytes)
        if cached is not None:
            return cached

        # Start from single bytes
        symbols = [bytes([b]) for b in token_bytes]
        if len(symbols) <= 1:
            out = tuple(symbols)
            self._bpe_cache[token_bytes] = out
            return out

        while True:
            best_rank = None
            best_pair = None

            # find best mergeable adjacent pair (lowest rank)
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            # merge all occurrences of best_pair in a single pass
            new_symbols: list[bytes] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                    new_symbols.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

            if len(symbols) <= 1:
                break

        out = tuple(symbols)
        self._bpe_cache[token_bytes] = out
        return out

    def _encode_normal_span(self, span: str) -> list[int]:
        ids = []
        for m in self._pretok_re.finditer(span):
            piece = m.group(0)
            b = piece.encode("utf-8")
            for bb in self._bpe(b):
                ids.append(self.bytes_to_id[bb])
        return ids


    def encode(self, text: str) -> list[int]:
        if self._special_re is None:
            return self._encode_normal_span(text)

        ids = []
        pos = 0
        for m in self._special_re.finditer(text):
            #print("SPECIAL MATCH:", m.group(0), m.start(), m.end())
            ids.extend(self._encode_normal_span(text[pos:m.start()]))

            special = m.group(0)
            ids.append(self.special_to_id[special])

            pos = m.end()

        ids.extend(self._encode_normal_span(text[pos:]))
        return ids

    

    def decode(self, ids: list[int]) -> str:
        tokens=b"".join(self.id_to_bytes[i] for i in ids)
        text=tokens.decode("utf-8",errors="replace")
        return text

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Stream encoding that matches encoding the concatenation of all chunks.
        Key idea: keep a rolling buffer and only emit tokens when we're confident
        they can’t change with more future characters.
        """
        # If this looks like a file object, prefer .read() in big chunks
        if hasattr(iterable, "read"):
            read = getattr(iterable, "read")
            yield from self._encode_stream_reader(read)
            return

        # Otherwise treat as generic iterable of strings
        buffer = ""
        lookahead = max(1, self._max_special_len - 1)

        for chunk in iterable:
            buffer += chunk
            yield_from, buffer = self._encode_buffer_prefix(buffer, lookahead)
            for _id in ids:
                yield _id
        # flush remainder
        for _id in self.encode(buffer):
            yield _id

    def _encode_stream_reader(self, read_fn) -> Iterator[int]:
        buffer = ""
        lookahead = max(1, self._max_special_len - 1)

        while True:
            chunk = read_fn(65536)
            if chunk == "":
                break

            buffer += chunk
            ids,buffer=self._encode_buffer_prefix(buffer,lookahead)
            for _id in ids:
                yield _id
                
        for _id in self.encode(buffer):
            yield _id

    def _encode_buffer_prefix(self, buffer: str, lookahead: int) -> tuple[list[int], str]:
        """
        Encode as much of buffer as is safe, leaving a suffix (lookahead) to avoid
        splitting:
          - inside a special token
          - inside a regex token that might grow with more input
        """
        safe_end = len(buffer) - lookahead
        if safe_end <= 0:
            return [], buffer

        out: list[int] = []
        
        last_end = 0

        def consume_normal(span_start: int, span_end: int) -> None:
            nonlocal last_end
            for m in self._pretok_re.finditer(buffer[span_start:span_end]):
                end = span_start + m.end()
                if end <= safe_end:
                    last_end = end
                else:
                    break

        if self._special_re is None:
            consume_normal(0,len(buffer))
        else :
            pos=0
            for sm in self._special_re.finditer(buffer):
                consume_normal(pos,sm.start())
                if sm.start() >=safe_end:
                    break
                if sm.end()<=safe_end:
                    last_end=sm.end()
                    pos=sm.end()
                else:
                    break
            else:
                consume_normal(pos,len(buffer))
        if last_end==0:
            return [],buffer
        prefix=buffer[:last_end]
        return self.encode(prefix),buffer[last_end:]