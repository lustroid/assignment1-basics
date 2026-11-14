# OpenWebText training profile: eight workers, 128 MiB

## Configuration and reproducibility

The full 11,920,511,059-byte training corpus was processed with eight workers,
128 MiB target chunks, a 32,000-entry vocabulary, and `<|endoftext|>`.
CPU profiling and five-second memory sampling were enabled. Outputs are in
`artifacts/tokenizers/openwebtext-32k-w8-c128-profile/` (ignored by Git).

```bash
.venv/bin/python -m cs336_basics.profile_bpe \
  data/owt_train.txt \
  --num-workers 8 --chunk-size-mib 128 \
  --vocab-size 32000 --special-token '<|endoftext|>' \
  --profile --memory-interval 5 \
  --output-dir artifacts/tokenizers/openwebtext-32k-w8-c128-profile
```

The invocation was wrapped in an external guard checking available system RAM
every five seconds and stopping below 3 GiB. It did not trigger; minimum
observed available RAM was about 14.5 GiB. The command above does not include
that guard. See `command.txt`, `benchmark.log`, and `memory-guard.json`.

## Stage wall times

| Stage | Unprofiled baseline | Profiled run |
|---|---:|---:|
| Chunk boundaries | 8.81 s | 8.98 s |
| Pre-tokenization, 89 chunks | 447.70 s | 823.37 s |
| Merge initialization | 39.61 s | 135.34 s |
| Merging | 253.30 s | 385.04 s |
| Total wall time, including cleanup | 755.25 s | 1357.60 s |

Profiling increased total time by about 1.80x and pre-tokenization time by
1.84x. These are single runs, with cache state, background work, and temperature
uncontrolled. Use the unprofiled run for throughput, not profiled wall times.
The profiled run's sampled peak process-tree RSS was about 5.19 GiB during
pre-tokenization and 12.16 GiB overall. Sampling can miss peaks, and RSS may
double-count shared pages. The configured interval is not a guarantee of exact
sample spacing, especially while Python is busy.

## Worker pre-tokenization breakdown

All 89 worker profiles were aggregated. Their total is 6406.57 summed elapsed
profile seconds across concurrent workers, not 6406.57 seconds of wall time or
pure CPU execution. Percentages below partition self time, avoiding overlapping
cumulative times. Profiling overhead can distort the relative shares.

| Operation | Summed self seconds | Share | Calls |
|---|---:|---:|---:|
| `_count_pretokens` body | 4749.73 | 74.14% | 89 |
| Match `.group()` | 831.82 | 12.98% | 2,471,753,092 |
| String `.encode()` | 767.20 | 11.98% | 2,471,753,092 |
| UTF-8 `.decode()` | 40.59 | 0.63% | 89 |
| File `.read()` | 7.46 | 0.12% | 89 |
| Counter `__missing__` | 4.76 | 0.07% | 35,042,610 |
| Regex `.finditer()` setup | 2.78 | 0.04% | 2,399,575 |

The corpus produced 2,471,753,092 pre-token occurrences and 6,601,892 unique
pre-tokens. Even small per-occurrence operations are repeated billions of times.

The 74.14% body bucket includes advancing regex iterators, dictionary hashing
and counter reads/writes, allocations, and Python loop overhead. These costs
are not individually resolved. In particular:

- `.finditer()` setup time is not total regex matching time: matching happens
  as its iterator advances within the loop.
- `Counter.__missing__` measures only missing-key handling, not all counting.
- Special-token iterator advancement also contributes to the unresolved body;
  this run does not isolate the entire special-token detection stage.

Read and UTF-8 decode together account for under 1% of measured worker self
time. That argues against file-read calls dominating this run, but does not
establish cold-disk behavior or distinguish compute from memory-bandwidth
limits. Repeated full-corpus runs may benefit from the filesystem cache.

Per-chunk profiled durations ranged from 52.27 to 102.67 seconds. Different
content and scheduling can affect this range; it alone does not prove a
load-balancing problem. Worker profiles cover `_count_pretokens`, excluding
startup, dispatch, and serializing/returning their result counters.

## Parent-profile limitation and diagnostic

The full parent call tree is internally inconsistent: `single_bpe` has only
0.047 s cumulative profile time despite 1352.73 s of measured stage time.
It also includes background sampler and executor thread frames. Do not use its
nesting, wait totals, or cumulative percentages to attribute IPC, aggregation,
or sampler cost. Preserve it as diagnostic evidence, not a trusted breakdown.

A small controlled diagnostic used the 5 MiB TinyStories fixture, two workers,
1 MiB chunks, a 300-entry vocabulary, and profiling in both cases:

| Memory sampling | Stage total | `single_bpe` profile cumulative | Sampler frames |
|---|---:|---:|---|
| Disabled | 1.2664 s | 1.2665 s | Absent |
| Enabled, 0.1 s target interval | 1.2034 s | 0.0021 s | Present |

This reproduces the attribution problem with the in-process sampling thread
enabled in this Python 3.13.15 environment. It does not establish the runtime's
underlying cause. Raw diagnostic reports and profiles are preserved under
`diagnostic-no-sampler/`, `diagnostic-with-sampler/`, and
`parent-profile-diagnostic.json` in the output directory. A future parent
profiling run should disable in-process memory sampling and retain external
memory monitoring. No trainer or profiler implementation was changed here.

## Artifacts, validation, and next investigation

The output directory contains all per-chunk profiles and `parent.prof` under
the profile subdirectory, the profiler's readable summaries, plus:

- `workers-aggregate.prof` for interactive inspection with `python -m pstats`.
- `worker-functions.json` and `parent-functions.json` for full-precision rows.
- `worker-chunks.json` for chunk offsets, sizes, worker PIDs, and elapsed times.
- `report.json`, memory samples, the command/log, and the guard outcome.
- `validation.json`: byte-identical vocabulary and merge files versus the
  unprofiled baseline, correct sizes, and six Unicode/special-token round-trip
  and streaming/eager smoke cases passed.

The result signature is
`a1bff1312f2441a3d5167d7dea33523a59018cb8fcd20bf417b577fed5ff187a`.
Raw profiles remain local and need a separate backup; this Markdown summary
is outside the ignored artifacts directory so it can be tracked.

The function-level profile above leaves the main loop unresolved. The follow-up
below measures inside that loop without changing the trainer implementation.

## Follow-up: isolating the main-loop bottleneck

### Method and scope

Eight document-aligned samples were taken from evenly spread chunk offsets
across OpenWebText. Each targets 16 MiB, ending at the next complete special
token boundary. Together they contain 134,273,920 bytes and 27,796,181 ordinary
pre-token occurrences. This is a representative sampling experiment, not a
measurement of each operation over the entire corpus.

Eight processes ran each measurement three times in shuffled order after a
warmup. Text and ordinary-region spans were prepared before timing, excluding
disk I/O, decoding, and special-token search from ordinary-loop measurements.
Special-token scanning was timed independently. Results are median sums of
worker elapsed times, not overall wall time.

Two independent methods were used:

1. Progressive passes: regex iteration only; add match extraction; add UTF-8
   byte conversion; add the original counter increment. Each pass includes
   the same occurrence counter. Adjacent pass differences estimate the added
   cost of each operation; allocation lifetimes and cache behavior also change.
2. Sparse in-loop timers: explicitly time `next()`, `.group()`, `.encode()`,
   counter lookup, and counter assignment plus integer addition for roughly one
   in 1,024 matches. Three different sampling phases yielded 81,438 observations.
   Both raw and approximate timer-overhead-corrected values are preserved.

The sampled loop was about 6.2% slower than the progressive full loop. In a
separate paired check against the unchanged `_count_pretokens` function,
diagnostic-loop overhead ranged from 2.4% to 12.1%, with a 4.4% paired median.
The original-function timings include its warm file read, decode, and special
search; diagnostic preparation is excluded. This is a fidelity check, not an
exact timer-overhead correction. All counters matched the original function,
and every repeated full/sampled pass produced identical count hashes.

### Progressive-pass results

| Pass | Median summed worker seconds |
|---|---:|
| Special-token search alone | 0.150 |
| Ordinary regex iteration | 18.385 |
| Regex + match extraction | 22.229 |
| Regex + extraction + byte conversion | 24.282 |
| Full loop including counting | 36.416 |

| Estimated contribution to the full ordinary loop | Seconds | Share |
|---|---:|---:|
| Regex iterator advancement and common loop control | 18.385 | 50.5% |
| Added match extraction | 3.844 | 10.6% |
| Added UTF-8 byte conversion | 2.054 | 5.6% |
| Added counter/hash/allocation work | 12.134 | 33.3% |

### Sparse operation timings

The following percentages normalize only the five timed operations. They omit
some loop bookkeeping, and are not exact percentages of production wall time.

| Operation | Raw timer share | Timer-corrected share |
|---|---:|---:|
| Regex iterator `next()` | 40.7% | 44.9% |
| Match `.group()` | 14.4% | 13.2% |
| String `.encode()` | 12.0% | 10.4% |
| Counter lookup | 15.6% | 14.7% |
| Counter assignment + integer addition | 17.4% | 16.8% |

Counter lookup plus update is about 31.5% of corrected timed operations, close
to the progressive-pass estimate of 33.3%. Both methods rank regex iterator
advancement first, counter work second, and extraction/conversion behind them.
Regex iteration includes searching, Unicode-category handling, and match
construction; this experiment does not split those native-regex internals.
Likewise, counter timings include hashing, lookup/update, allocation and memory
access, rather than isolating each of those lower-level costs.

The per-sample progressive estimates ranked regex above added counting in
seven of eight samples; counting was larger in one. The main-loop workload is
therefore mixed, not universally dominated by one operation for every document.
Measured worker process CPU time was almost equal to worker elapsed time in
these preloaded passes, consistent with active CPU-side work rather than file
I/O waits. This does not distinguish execution throughput from memory latency
or bandwidth limits.

### Conclusion and reproducibility

For these spread-out samples, **regex iterator advancement is the largest
main-loop component, roughly 45-50%; counter work is second, roughly 32-33%.**
This resolves the earlier function-level profile's central ambiguity. The
method agreement supports this ranking, not nanosecond precision or an exact
full-corpus percentage. Cache state, order, temperature, instrumentation, and
text composition affect the estimates. Any optimization still needs validation
against unchanged counts and unprofiled elapsed time.

The measurement-only harness, exact offsets, source hash, repeated trials,
operation timings, count hashes, logs, and original-function checks are saved
under `artifacts/tokenizers/pretoken-loop-diagnostic/` (ignored by Git):

```bash
.venv/bin/python artifacts/tokenizers/pretoken-loop-diagnostic/diagnostic.py
.venv/bin/python artifacts/tokenizers/pretoken-loop-diagnostic/check_original.py
```

The first harness reads the prior full-run `worker-chunks.json` to choose its
sample offsets. Back up these local scripts and artifacts separately; this
Markdown preserves the methodology and conclusions in a trackable file.
