import heapq
import json
import os
from collections.abc import Iterable, Iterator
from typing import Self

import regex as re

from .bpe import _PRETOKEN_PATTERN


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
        '''
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        '''
        pass
   
    def decode(self, ids: list[int]) -> str:
        '''
        Decode a sequence of token IDs into text.
        '''
        pass
