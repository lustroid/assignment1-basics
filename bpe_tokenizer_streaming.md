# BPE Streaming Encoder: Implementation and Analysis

## 1. Overview

I implemented a single-process, memory-efficient streaming BPE encoder:

`Tokenizer.encode_iterable(iterable: Iterable[str]) -> Iterator[int]`

**Implementation:** `bpe_tokenizer.py`

The encoder processes text fragments incrementally and yields individual token IDs without materializing the entire corpus.

Its primary correctness requirement is **fragment independence**:

```python
list(tokenizer.encode_iterable(fragments)) == tokenizer.encode("".join(fragments))
```

The implementation separates special-token recognition, pre-tokenization, BPE merging, and lazy output generation.

## 2. Architecture

```text
              Iterable[str]
                    |
                    v
          _iter_special_parts()
             (Trie matcher)
                    |
                    v
           itertools.groupby()
                    |
             +------+------+
             |             |
             v             v
        Ordinary Text   Special Token
             |             |
             v             |
     _iter_regex_pretokens()
             |             |
             v             |
       _encode_pretoken()
             |             |
             v             v
         yield from    Yield special ID
             |             |
             +------+------+
                    |
                    v
               Iterator[int]
```

### 2.1 Special-Token Recognition

`_iter_special_parts()` uses a Trie constructed during tokenizer initialization.

- `chain.from_iterable()` reads characters across arbitrary fragment boundaries.
- Trie traversal recognizes shared special-token prefixes.
- `best` tracks the longest completed match at the earliest matching position.
- A `pending` deque replays over-read characters, preventing failed prefix matches from hiding later matches.
- EOF resolves incomplete prefixes and any shorter completed match.

Special tokens bypass ordinary regex pre-tokenization and BPE merging. Each special token establishes a hard boundary.

`itertools.groupby()` lazily separates ordinary regions from specials without materializing entire regions.

### 2.2 Incremental Pre-tokenization

`_iter_regex_pretokens()` implements a specialized streaming recognizer equivalent to the training regex:

```regex
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

The key principle is:

> Finalize a pre-token only when future input cannot change its identity or boundaries.

Rather than repeatedly applying regex matching to a growing buffer, my implementation:

- Classifies Unicode characters as letters, numbers, punctuation, or whitespace.
- Uses bounded lookahead to recognize contractions.
- Preserves regex branch priority and whitespace backtracking.
- Accumulates characters until their pre-token boundaries are guaranteed.
- Finalizes remaining text upon EOF or a special-token boundary.

This avoids potentially quadratic rescanning and provides **$O(N+F)$** boundary recognition for $N$ characters across $F$ fragments.

### 2.3 BPE Encoding and Generator

Each complete pre-token is UTF-8 encoded and passed to `_encode_pretoken()`.

My BPE implementation uses:

- A merge-rank hashmap for priority lookup.
- A min-heap for selecting eligible merges.
- Stable-index arrays with predecessor/successor links.
- Lazy validation to discard stale heap candidates.

The two interfaces have different output granularities:

```text
_encode_pretoken() -> list[int]
encode_iterable()  -> Iterator[int]
```

`yield from` exposes each pre-token's encoded IDs individually. The generator suspends between yields and can release completed intermediate results.

**Streaming occurs at the pre-token level, while output is exposed at the token-ID level.**

## 3. Data Structures

| Data structure | Purpose |
|---|---|
| Special-token Trie | Recognize literal specials and shared prefixes |
| Terminal-node hashmap | Identify completed special tokens |
| `candidate` list | Track the current Trie matching attempt |
| `pending` deque | Replay over-read characters |
| Regex lookahead deque | Retain a bounded number of classified characters |
| Character-run list | Accumulate the current unfinished pre-token |
| `chain` / `groupby` | Flatten fragments and separate regions lazily |
| Merge-rank hashmap | Retrieve learned BPE priorities |
| Min-heap | Select the highest-priority eligible merge |
| Neighbor arrays | Support $O(1)$ local adjacency updates |
| Per-pre-token ID list | Temporarily store the current encoding result |

The Trie and merge tables are persistent tokenizer metadata. Streaming buffers and BPE state are local to each encoding invocation.

## 4. Complexity Analysis

Let:

- $N$: Total input Unicode characters.
- $F$: Number of input fragments.
- $S$: Maximum special-token length.
- $L_i$: UTF-8 byte length of ordinary pre-token $i$.
- $T$: Sum of configured special-token lengths.

### Time Complexity

| Operation | Complexity |
|---|---|
| Trie construction | $O(T)$, once |
| Special-token matching | $O(NS+F)$ |
| Ordinary pre-token recognition | $O(N+F)$ |
| BPE encoding per pre-token | Expected $O(L_i\log(1+L_i))$ |
| Yielding IDs | $O(1)$ per ID |

Overall, excluding initialization:

$$
\boxed{
O\left(
N\max(1,S)+F+
\sum_i L_i\log(1+L_i)
\right)
}
$$

This assumes expected constant-time hashmap operations and bounded token-key operation costs for a fixed model.

For fixed $S$, special-token matching is linear in input size. The BPE complexity depends on UTF-8 byte length rather than Unicode character count.

### Memory Complexity

Let $C$ be the largest supplied fragment and $L$ the longest unresolved ordinary pre-token/run in bytes.

Additional working memory is approximately:

$$
\boxed{O(C+S+L)}
$$

This excludes persistent vocabulary/merge metadata and memory retained by the caller.

**Streaming does not guarantee constant memory.** Extremely large fragments or pre-tokens can still require substantial RAM.

## 5. Correctness Invariants

| Invariant | Requirement |
|---|---|
| Input preservation | No characters are lost, duplicated, or reordered |
| Special-token precedence | Earliest match wins; longest match at that position takes priority |
| Regex equivalence | Streaming pre-tokens match the original regex |
| Fragment independence | Different fragment boundaries produce identical IDs |
| Early finalization | Future input cannot alter emitted pre-tokens |
| Output ordering | Token IDs preserve original text order |
| Laziness | The encoder does not require complete input exhaustion before yielding available results |

The central end-to-end invariant is:

$$
\boxed{
\begin{aligned}
&\operatorname{list}(\mathtt{encode\_iterable}(\mathit{fragments}))
\\ &\quad=\operatorname{encode}(\operatorname{concat}(\mathit{fragments}))
\end{aligned}
}
$$

## 6. Design Trade-offs

### Custom Lexer vs. Original Regex

My streaming lexer is manually specialized for `_PRETOKEN_PATTERN`.

**Advantage:** Linear-time incremental boundary detection without repeated full-buffer regex matching.

**Disadvantage:** Changes to the training regex do not automatically update the streaming lexer.

I considered automatic regex-to-state-machine generation but rejected it because the additional complexity is not justified for Optimus_Prime.

Instead, I treat the original regex as the **authoritative specification** and use differential testing to detect divergence.

### Streaming vs. Parallelism

The current implementation uses single-process streaming.

Parallel encoding is a separate optimization involving worker scheduling, interprocess communication, output ordering, and backpressure.

I deliberately excluded it from the current implementation.

## 7. Verification

The reported test suite covers:

- Arbitrary fragmentation and Unicode.
- Contractions and whitespace rules.
- Overlapping and incomplete special-token matches.
- Special tokens crossing fragment boundaries.
- Independent and interleaved iterators.
- Infinite-input laziness and memory usage.
- Long character runs (200,000 characters).
- Equivalence with eager encoding and GPT-2 reference IDs.

**Result: 77 tests passed.**

Differential testing is especially important for detecting divergence between `_PRETOKEN_PATTERN` and `_iter_regex_pretokens()`.

Passing tests provides evidence of correctness, not a formal equivalence proof.

## 8. Performance Validation

Measured on `tests/fixtures/tinystories_sample_5M.txt` (5,242,880 bytes), using the repository GPT-2 fixture model: 50,257 vocabulary entries, 50,000 merges, and `<|endoftext|>` enabled. These results concern this fixture/model, not a full TinyStories or OWT corpus.

### Throughput, memory, and fragment sizes

Each case ran three times in fresh processes, sequentially in shuffled order. Times below are **unprofiled medians**; peak RSS is the maximum across those trials. Timing includes UTF-8 file reading and the same bounded token-ID checksum sink, but excludes process startup, model loading, and warmup. Fragment sizes count Unicode characters; MB/s uses decimal bytes.

| Mode | Median seconds | MB/s | Peak RSS (MiB) |
|---|---:|---:|---:|
| encode() | 3.252 | 1.612 | 76.75 |
| encode_iterable(), 1 chars | 8.573 | 0.612 | 53.20 |
| encode_iterable(), 64 chars | 8.007 | 0.655 | 53.18 |
| encode_iterable(), 4,096 chars | 7.893 | 0.664 | 53.38 |
| encode_iterable(), 65,536 chars | 8.544 | 0.614 | 53.35 |

At 4,096 characters per fragment, streaming took **2.43× as long**, with approximately **23.38 MiB (30.5%) less peak RSS**. All trials, including the separate profiled runs, emitted **1,289,382 IDs with identical SHA-256 hashes**.

Single-character fragments took about 8.6% longer than 4,096-character fragments. The 64-character result was close to 4,096; 65,536-character trials varied from 7.990 to 8.580 seconds. There is no evidence here that increasing fragment size monotonically improves throughput or that 4,096 is universally optimal.

RSS includes the interpreter and tokenizer metadata. Streaming did not exceed its approximately 53 MiB model-loading high-water mark; **this does not mean encoding allocates no memory**. Eager encoding increased its pre-measurement high-water mark by a median 23.43 MiB. Streaming IDs were drained rather than collected. Filesystem cache state was not controlled, and this single corpus size does not establish a memory scaling curve.

### Profile findings

Separate cProfile runs took 8.926 seconds for eager encoding and 30.420 seconds for streaming with 4,096-character fragments. Instrumentation overhead is substantial and differs between paths, so these runs are excluded from throughput comparisons.

| Function | Eager cumulative seconds | Streaming cumulative seconds |
|---|---:|---:|
| `_encode_pretoken` | 7.278 | 8.364 |
| `_iter_regex_pretokens` | — | 20.862 |
| `_iter_special_parts` | — | 5.431 |

**Cumulative times overlap:** `_iter_special_parts` runs while the regex recognizer requests input, so its time is already included in the recognizer's total. Do not add these two rows.

Streaming self times further identify Python overhead: `_iter_special_parts` 3.544 s, `peek` 3.268 s, `_iter_regex_pretokens` 2.879 s, character `fullmatch` 2.255 s, and `characters` 2.005 s. Both modes called `_encode_pretoken` 1,263,131 times. The evidence points to character classification, special matching, lookahead, and generator coordination as the main added cost—not additional BPE work. This does not predict the speedup of any proposed optimization without another benchmark.

### Reproduction and artifacts

`cs336_basics/profile_tokenizer.py` accepts the same hex JSON artifacts as `Tokenizer.from_files`. For the current converted fixture model:

```bash
.venv/bin/python -m cs336_basics.profile_tokenizer \
  tests/fixtures/tinystories_sample_5M.txt \
  --vocab /tmp/bpe-tokenizer-validation/model/vocab.json \
  --merges /tmp/bpe-tokenizer-validation/model/merges.json \
  --fragment-chars 1 64 4096 65536 --repeats 3 --profile \
  --output-dir /tmp/bpe-tokenizer-validation
```

The temporary model files were converted from `tests/fixtures/gpt2_vocab.json` and `gpt2_merges.txt`; they are not the trainer's previously generated model. Replace these paths with persistent trainer artifacts for your own model. Raw GPT-2 fixture files use a different format and cannot be passed directly.

Each invocation creates a unique output directory containing `report.json`, raw trial data, `.prof` files, readable cumulative/self-time tables, and full-precision function statistics. `--profile` adds separate eager and streaming profiles (4,096-character fragments by default); `--profile-fragment-chars` changes that size. Use `--special-token` repeatedly for custom specials, or `--no-special-tokens` to disable them.

The measured report and profiles are preserved in [benchmarks/tokenizer_fixture](benchmarks/tokenizer_fixture/), including input/model/source hashes and environment details. Inspect a saved profile interactively with:

```bash
.venv/bin/python -m pstats benchmarks/tokenizer_fixture/stream-4096.prof
```

Three profiler CLI tests pass, covering Unicode/newline preservation, empty input, output equivalence, separate profiling runs, and invalid fragment sizes.

### Larger training-corpus experiment: 100 MiB

A follow-up used 1, 10, and 100 MiB prefixes of the actual TinyStories training file, with the same model and 4,096-character streaming fragments. Each mode ran twice in fresh processes; all output counts and hashes matched.

| Input | Eager seconds | Streaming seconds | Eager peak RSS (MiB) | Streaming peak RSS (MiB) |
|---|---:|---:|---:|---:|
| 1 MiB | 0.652 | 1.652 | 55.00 | 53.29 |
| 10 MiB | 6.642 | 16.381 | 105.05 | 53.15 |
| 100 MiB | 64.731 | 161.321 | 685.24 | 53.23 |

At 100 MiB, streaming took 2.49× as long with 92.2% less total peak RSS. The decoded Python string alone occupied 399.84 MiB, explaining part of eager memory growth beyond input-file size. Streaming stayed near its 53 MiB tokenizer/interpreter baseline.

Isolated, unprofiled 100 MiB stage runs measured special matching at **20.921 s streaming vs. 0.083 s native regex**, and ordinary pre-token recognition at **66.001 s vs. 10.440 s**. Both comparisons produced matching hashes. Separate cProfile runs on the 10 MiB prefix identify per-character Python operations as the added cost. Diagnostic stages preload text and are excluded from encoding-memory measurements; their sink costs differ from end-to-end encoding.

There were **25,254,741 pre-token occurrences but only 21,809 unique byte strings**: 99.9136% were repeats. This is an upper bound on avoidable BPE calls with an unlimited cache, not a measured overall speedup or bounded-cache hit rate. Pre-tokenization and special matching would still be required.

See [the full experiment report](benchmarks/tokenizer_100m/README.md) for methodology, raw reports/profiles, repetition coverage, and reproduction commands. The reusable diagnostic is `cs336_basics/profile_tokenizer_stages.py`.

## 9. Conclusion

My streaming BPE encoder combines:

1. Trie-based special-token recognition.
2. Linear-time incremental pre-tokenization.
3. Heap-based ranked BPE merging.
4. Stable-index sequence representation.
5. Lazy token-ID generation.

The implementation avoids corpus-wide input/output materialization while preserving tokenization across arbitrary fragments.

The principal maintenance trade-off is keeping the custom lexer consistent with the authoritative regex through differential testing.

**Next step:** Benchmark a bounded pre-token encoding cache and front-end overhead reductions, then check whether the gains generalize to more diverse corpora.