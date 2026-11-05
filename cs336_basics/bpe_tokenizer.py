import heapq
import json
import os
from collections import deque
from collections.abc import Iterable, Iterator
from itertools import chain, groupby
from typing import Self

import regex as re

from .bpe import _PRETOKEN_PATTERN


_PRETOKEN_CHARACTER_PATTERN = re.compile(
    r"(?P<letter>\p{L})|(?P<number>\p{N})|(?P<whitespace>\s)|(?P<punctuation>[^\s\p{L}\p{N}])"
)


def _iter_regex_pretokens(fragments: Iterable[str]) -> Iterator[str]:
    """Recognize the training regex's ordinary pre-tokens in O(N) total time.

    Call this on ONE ordinary-text region after special-token recognition;
    exhausting fragments means that region has ended (at a special token or EOF).
    Input fragment boundaries themselves have no tokenization significance.

    Each character is classified once with the same Unicode regex engine as
    training. At most three classified characters are held for lookahead; runs
    accumulate in a list and are joined only when finalized. Time is O(N + F)
    for N characters across F fragments (including empty ones). Working memory
    is O(L), plus the current input fragment, for longest pre-token/run length L.
    This state machine is specific to _PRETOKEN_PATTERN; keep them in sync.
    """
    def characters() -> Iterator[tuple[str, str]]:
        for fragment in fragments:
            for character in fragment:
                match = _PRETOKEN_CHARACTER_PATTERN.fullmatch(character)
                assert match is not None and match.lastgroup is not None
                yield character, match.lastgroup

    source = characters()
    pending: deque[tuple[str, str]] = deque()
    exhausted = False

    def peek(offset: int = 0) -> tuple[str, str] | None:
        nonlocal exhausted
        while len(pending) <= offset and not exhausted:
            try:
                pending.append(next(source))
            except StopIteration:
                exhausted = True
        return pending[offset] if offset < len(pending) else None

    contraction_endings = {"l": "l", "v": "e", "r": "e"}
    while (first := peek()) is not None:
        # The contraction alternative has priority only at the match's start.
        # An apostrophe within a punctuation run (including " '") is ordinary.
        if first[0] == "'":
            second = peek(1)
            width = 0
            if second is not None:
                if second[0] in "sdmt":
                    width = 2
                elif second[0] in contraction_endings:
                    third = peek(2)
                    if third is not None and third[0] == contraction_endings[second[0]]:
                        width = 3
            if width:
                yield "".join(pending.popleft()[0] for _ in range(width))
                continue

        parts: list[str] = []
        # Only an ASCII space can prefix a letter/number/punctuation run.
        if first[0] == " ":
            second = peek(1)
            if second is not None and second[1] != "whitespace":
                parts.append(pending.popleft()[0])
                first = second

        kind = first[1]
        while (current := peek()) is not None and current[1] == kind:
            parts.append(pending.popleft()[0])

        if kind == "whitespace" and current is not None and len(parts) > 1:
            # Before a non-whitespace character, \s+(?!\S) backs off exactly
            # one whitespace character. That last character starts the next
            # match, possibly as the ASCII-space prefix of a non-whitespace run.
            pending.appendleft((parts.pop(), "whitespace"))
        yield "".join(parts)


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        '''
        Construct a tokenizer from a given vocabulary, list of merges, 
        and (optionally) a list of special tokens. This function should accept
        the following parameters:
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        '''
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        if any(not token for token in self.special_tokens):
            raise ValueError("Special tokens must be nonempty strings")

        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        next_id = max(self.vocab, default=-1) + 1
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.token_to_id:
                self.vocab[next_id] = token_bytes
                self.token_to_id[token_bytes] = next_id
                next_id += 1

        self.merge_ranks: dict[tuple[bytes, bytes], int] = {}
        for rank, pair in enumerate(self.merges):
            self.merge_ranks.setdefault(pair, rank)
        # Reuse merged byte strings instead of copying them on every occurrence.
        self._merged_tokens = [left + right for left, right in self.merges]
        self._byte_tokens = tuple(bytes([value]) for value in range(256))
        self._special_pattern = (
            re.compile("|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)))
            if self.special_tokens else None
        )
        # The streaming matcher uses a trie to recognize literal special tokens
        # across fragments and delay shorter matches while a longer one is possible.
        self._special_trie: list[dict[str, int]] = [{}]
        self._special_terminals: dict[int, str] = {}
        for token in self.special_tokens:
            node = 0
            for character in token:
                if character not in self._special_trie[node]:
                    self._special_trie[node][character] = len(self._special_trie)
                    self._special_trie.append({})
                node = self._special_trie[node][character]
            self._special_terminals[node] = token
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Self:
        '''
        Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the 
        same format that your BPE training code output) and (optionally) a list of special tokens.
        This method should accept the following additional parameters:
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        '''
        with open(vocab_filepath, encoding="utf-8") as vocab_file:
            vocab = {int(token_id): bytes.fromhex(token_hex) for token_id, token_hex in json.load(vocab_file).items()}
        with open(merges_filepath, encoding="utf-8") as merges_file:
            merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in json.load(merges_file)]
        return cls(vocab, merges, special_tokens)
        
    def encode(self, text: str) -> list[int]:
        '''
        Encode an input text into a sequence of token IDs.
        '''
        ids: list[int] = []
        boundaries = self._special_pattern.finditer(text) if self._special_pattern is not None else ()
        start = 0
        for boundary in boundaries:
            for match in _PRETOKEN_PATTERN.finditer(text, start, boundary.start()):
                ids.extend(self._encode_pretoken(match.group().encode("utf-8")))
            ids.append(self.token_to_id[boundary.group().encode("utf-8")])
            start = boundary.end()
        for match in _PRETOKEN_PATTERN.finditer(text, start):
            ids.extend(self._encode_pretoken(match.group().encode("utf-8")))
        return ids

    def _encode_pretoken(self, pretoken: bytes) -> list[int]:
        """Apply ranked merges to one pre-token using stable node indices."""
        tokens = [self._byte_tokens[value] for value in pretoken]
        length = len(tokens)
        if length == 0:
            return []
        previous = [index - 1 for index in range(length)]
        following = [index + 1 for index in range(length)]
        following[-1] = -1

        # Position breaks rank ties, so overlapping occurrences merge left first.
        candidates: list[tuple[int, int, int]] = []
        for left in range(length - 1):
            rank = self.merge_ranks.get((tokens[left], tokens[left + 1]))
            if rank is not None:
                candidates.append((rank, left, left + 1))
        heapq.heapify(candidates)

        while candidates:
            rank, left, right = heapq.heappop(candidates)
            # A neighboring merge may have consumed a node or changed its value.
            if following[left] != right or (tokens[left], tokens[right]) != self.merges[rank]:
                continue

            before, after = previous[left], following[right]
            tokens[left] = self._merged_tokens[rank]
            tokens[right] = b""
            following[left] = after
            previous[right] = following[right] = -1
            if after != -1:
                previous[after] = left

            # Only the two edges adjacent to the merged node can become eligible.
            for first, second in ((before, left), (left, after)):
                if first != -1 and second != -1:
                    new_rank = self.merge_ranks.get((tokens[first], tokens[second]))
                    if new_rank is not None:
                        heapq.heappush(candidates, (new_rank, first, second))

        ids = []
        index = 0
        while index != -1:
            ids.append(self.token_to_id[tokens[index]])
            index = following[index]
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode concatenated fragments, preserving all token boundaries.

        Only the current input fragment, unresolved pre-token, special-token
        lookahead, and that pre-token's BPE state/output are retained. Very long
        pre-tokens can still require substantial memory. No corpus-wide output
        list or cache is accumulated; all streaming state belongs to this call.
        """
        if not self.special_tokens:
            for pretoken in _iter_regex_pretokens(iterable):
                yield from self._encode_pretoken(pretoken.encode("utf-8"))
            return

        # groupby supplies each ordinary region lazily. Reaching its end tells
        # the regex recognizer to flush, even if the special abuts a word/space.
        parts = self._iter_special_parts(iterable)
        for is_special, region in groupby(parts, key=lambda part: part[0]):
            if is_special:
                for _, special in region:
                    yield self.token_to_id[special.encode("utf-8")]
            else:
                for pretoken in _iter_regex_pretokens(text for _, text in region):
                    yield from self._encode_pretoken(pretoken.encode("utf-8"))

    def _iter_special_parts(self, fragments: Iterable[str]) -> Iterator[tuple[bool, str]]:
        """Yield ordinary characters or whole special tokens, earliest/longest first.

        Lookahead is bounded by maximum special-token length S. Failed prefixes
        are replayed so a match starting inside them is not lost. Worst-case
        matching time is O(N*S + F), linear in input size for fixed specials;
        this simple trie matcher is not an Aho-Corasick automaton.
        """
        source = chain.from_iterable(fragments)
        pending: deque[str] = deque()
        while True:
            node = 0
            candidate: list[str] = []
            best: str | None = None
            while True:
                character = pending.popleft() if pending else next(source, None)
                if character is None:
                    break
                candidate.append(character)
                child = self._special_trie[node].get(character)
                if child is None:
                    break
                node = child
                if node in self._special_terminals:
                    best = self._special_terminals[node]
                if not self._special_trie[node]:
                    break

            if not candidate:
                return
            consumed = len(best) if best is not None else 1
            pending.extendleft(reversed(candidate[consumed:]))
            yield (True, best) if best is not None else (False, candidate[0])
   
    def decode(self, ids: list[int]) -> str:
        '''
        Decode a sequence of token IDs into text.
        '''
        pass
