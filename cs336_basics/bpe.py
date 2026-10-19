import heapq
import mmap
import multiprocessing
import os
import re as standard_re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import regex as re


_PRETOKEN_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
type Pair = tuple[bytes, bytes]


class _HeapEntry:
    """Reverse both frequency and byte-pair ordering for heapq's min-heap."""

    __slots__ = ("frequency", "pair")

    def __init__(self, frequency: int, pair: Pair):
        self.frequency = frequency
        self.pair = pair

    def __lt__(self, other: "_HeapEntry") -> bool:
        if self.frequency != other.frequency:
            return self.frequency > other.frequency
        return self.pair > other.pair


def _count_pretokens(input_path: str, start: int, end: int, special_tokens: list[str]) -> Counter[bytes]:
    with open(input_path, "rb") as corpus:
        corpus.seek(start)
        content = corpus.read(end - start).decode("utf-8")

    counts: Counter[bytes] = Counter()
    if special_tokens:
        special_pattern = re.compile(
            "|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
        )
        boundaries = special_pattern.finditer(content)
    else:
        boundaries = ()
    segment_start = 0
    for boundary in boundaries:
        for match in _PRETOKEN_PATTERN.finditer(content, segment_start, boundary.start()):
            counts[match.group().encode("utf-8")] += 1
        segment_start = boundary.end()
    for match in _PRETOKEN_PATTERN.finditer(content, segment_start):
        counts[match.group().encode("utf-8")] += 1
    return counts


def _count_chunk(task: tuple[str, int, int, list[str], str | None]) -> Counter[bytes]:
    input_path, start, end, special_tokens, profile_dir = task
    if profile_dir is None:
        return _count_pretokens(input_path, start, end, special_tokens)

    import cProfile

    profiler = cProfile.Profile()
    counts = profiler.runcall(_count_pretokens, input_path, start, end, special_tokens)
    profiler.dump_stats(str(Path(profile_dir) / f"worker-{os.getpid()}-{start}.prof"))
    return counts


def _chunk_boundaries(input_path: str, num_workers: int, special_tokens: list[str]) -> list[int]:
    """Split only at complete special-token matches, including overlapping literals.

    mmap lets the boundary scan operate on bytes without copying the full corpus.
    With no special tokens (or no matches), keep one chunk rather than split a
    UTF-8 character or regex pre-token. Uneven documents can yield fewer chunks.
    """
    size = os.path.getsize(input_path)
    if num_workers == 1 or not special_tokens or size == 0:
        return [0, size]

    ordered_tokens = sorted(special_tokens, key=lambda token: len(token), reverse=True)
    escaped_tokens: list[bytes] = [standard_re.escape(token.encode("utf-8")) for token in ordered_tokens]
    pattern = standard_re.compile(b"|".join(escaped_tokens))
    boundaries = [0]
    target = 1
    with open(input_path, "rb") as corpus, mmap.mmap(corpus.fileno(), 0, access=mmap.ACCESS_READ) as content:
        for match in pattern.finditer(content):
            position = match.start()
            if position * num_workers < size * target:
                continue
            boundaries.append(position)
            while target < num_workers and position * num_workers >= size * target:
                target += 1
            if target == num_workers:
                break
    boundaries.append(size)
    return boundaries


def single_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    *,
    num_workers: int = 1,
    profile_dir: str | os.PathLike | None = None,
    timings: dict[str, float] | None = None,
    **kwargs,
) -> tuple[dict[int, bytes], list[Pair]]:
    """Train BPE with sequential merges and optionally parallel pre-tokenization.

    Each distinct pre-token has one mutable sequence, weighted by its corpus
    frequency. Stable array positions form a doubly linked list; a pair's
    occurrence set contains the positions of its left nodes. Heap entries are
    validated against the current weighted pair counts when popped.

    Each worker reads its assigned UTF-8 chunk into memory. Without special-token
    boundaries, pre-tokenization stays sequential. profile_dir optionally records
    a cProfile file per parallel chunk (sequential work is captured by the caller's
    profiler); timings receives stage wall times in seconds.
    """
    started = perf_counter()
    if num_workers < 1:
        raise ValueError("num_workers must be positive")
    special_tokens = list(dict.fromkeys(special_tokens))
    if any(not token for token in special_tokens):
        raise ValueError("Special tokens must be nonempty strings")
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size must accommodate all 256 bytes and special tokens")

    byte_tokens = [bytes([value]) for value in range(256)]
    vocab = dict(enumerate(byte_tokens))
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    if len(vocab) == vocab_size:
        return vocab, []

    if profile_dir is not None:
        Path(profile_dir).mkdir(parents=True, exist_ok=True)
    input_path = os.fspath(input_path)
    boundaries = _chunk_boundaries(input_path, num_workers, special_tokens)
    tasks = [
        (input_path, start, end, special_tokens, os.fspath(profile_dir) if profile_dir is not None else None)
        for start, end in zip(boundaries, boundaries[1:])
    ]
    boundary_finished = perf_counter()
    pretoken_count: Counter[bytes] = Counter()
    if len(tasks) == 1:
        # Keep sequential work inside the caller's profile, rather than nesting
        # a second cProfile instance (which Python does not support).
        pretoken_count = _count_pretokens(input_path, tasks[0][1], tasks[0][2], special_tokens)
    else:
        with ProcessPoolExecutor(max_workers=len(tasks), mp_context=multiprocessing.get_context("spawn")) as pool:
            for counts in pool.map(_count_chunk, tasks):
                pretoken_count.update(counts)
        del counts
    pretokenization_finished = perf_counter()

    # Indices never shift. Empty bytes mark consumed nodes; live tokens are nonempty.
    tokens: list[bytes] = []
    previous: list[int] = []
    following: list[int] = []
    weights: list[int] = []
    pair_count: dict[Pair, int] = defaultdict(int)
    pair_occurrences: dict[Pair, set[int]] = defaultdict(set)

    for pretoken, frequency in pretoken_count.items():
        offset = len(tokens)
        size = len(pretoken)
        for position, value in enumerate(pretoken):
            node = offset + position
            tokens.append(byte_tokens[value])
            previous.append(node - 1 if position else -1)
            following.append(node + 1 if position + 1 < size else -1)
            weights.append(frequency)
            if position + 1 < size:
                pair = (byte_tokens[value], byte_tokens[pretoken[position + 1]])
                pair_count[pair] += frequency
                pair_occurrences[pair].add(node)
    del pretoken_count

    heap = [_HeapEntry(frequency, pair) for pair, frequency in pair_count.items()]
    heapq.heapify(heap)
    merges: list[Pair] = []
    changed: set[Pair] = set()
    initialization_finished = perf_counter()

    def remove_edge(left: int) -> None:
        right = following[left]
        pair = (tokens[left], tokens[right])
        occurrences = pair_occurrences[pair]
        occurrences.remove(left)
        pair_count[pair] -= weights[left]
        if not occurrences:
            del pair_occurrences[pair]
            del pair_count[pair]
        changed.add(pair)

    def add_edge(left: int) -> None:
        right = following[left]
        pair = (tokens[left], tokens[right])
        pair_occurrences[pair].add(left)
        pair_count[pair] += weights[left]
        changed.add(pair)

    while len(vocab) < vocab_size and heap:
        entry = heapq.heappop(heap)
        pair = entry.pair
        if entry.frequency != pair_count.get(pair, 0):
            continue

        merged_token = pair[0] + pair[1]
        # Stable node indices preserve left-to-right order within each pre-token.
        # Snapshot before mutating the occurrence sets, including overlaps.
        occurrences = sorted(pair_occurrences[pair])
        changed.clear()
        for left in occurrences:
            right = following[left]
            if tokens[left] != pair[0] or right == -1 or tokens[right] != pair[1]:
                continue

            before = previous[left]
            after = following[right]
            if before != -1:
                remove_edge(before)
            remove_edge(left)
            if after != -1:
                remove_edge(right)

            tokens[left] = merged_token
            tokens[right] = b""
            following[left] = after
            previous[right] = following[right] = -1
            if after != -1:
                previous[after] = left

            if before != -1:
                add_edge(before)
            if after != -1:
                add_edge(left)

        vocab[len(vocab)] = merged_token
        merges.append(pair)
        # Batch heap refreshes until the merge round ends. Both increases and
        # decreases get fresh entries; zero-frequency pairs need no entry.
        for changed_pair in changed:
            frequency = pair_count.get(changed_pair, 0)
            if frequency:
                heapq.heappush(heap, _HeapEntry(frequency, changed_pair))

    if timings is not None:
        finished = perf_counter()
        timings.update(
            boundaries=boundary_finished - started,
            pretokenization=pretokenization_finished - boundary_finished,
            initialization=initialization_finished - pretokenization_finished,
            merging=finished - initialization_finished,
            total=finished - started,
            chunks=float(len(tasks)),
        )
    return vocab, merges
