# Trained BPE tokenizers

Generated outputs live in `artifacts/tokenizers/`, which is ignored by Git.
Back up this directory separately if you want to avoid retraining.

Run these commands from the repository root, one at a time:

```bash
.venv/bin/python -m cs336_basics.profile_bpe \
  data/TinyStoriesV2-GPT4-train.txt \
  --num-workers 4 --chunk-size-mib 32 \
  --vocab-size 10000 --special-token '<|endoftext|>' \
  --memory-interval 5 \
  --output-dir artifacts/tokenizers/tinystories-10k

.venv/bin/python -m cs336_basics.profile_bpe \
  data/owt_train.txt \
  --num-workers 4 --chunk-size-mib 32 \
  --vocab-size 32000 --special-token '<|endoftext|>' \
  --memory-interval 5 \
  --output-dir artifacts/tokenizers/openwebtext-32k
```

Both use the full training corpus and include all 256 byte tokens plus
`<|endoftext|>`. Four workers and 32 MiB target chunks limit concurrent
pre-tokenization memory on this laptop. The chunk size is a soft limit:
document boundaries can make a chunk larger.

Each output directory contains `vocab.json` (IDs mapped to hex-encoded bytes),
`merges.json` (ordered pairs of hex-encoded bytes), `longest_tokens.txt`,
and `report.json`. The report records stage timing, vocabulary and merge counts,
trainer source hash, output signature, environment, and input path and size.
The saved `input-sha256.json` additionally identifies the exact corpus contents.
Memory sampling adds monitoring overhead; these are not controlled benchmarks.

Reload the paired vocabulary and merges with `Tokenizer.from_files`, supplying
`["<|endoftext|>"]` as the special-token configuration. Raw GPT-2 test fixtures
are in a different format and are not substitutes for these trained artifacts.

## Eight-worker OpenWebText comparison

```bash
.venv/bin/python -m cs336_basics.profile_bpe \
  data/owt_train.txt \
  --num-workers 8 --chunk-size-mib 128 \
  --vocab-size 32000 --special-token '<|endoftext|>' \
  --memory-interval 5 \
  --output-dir artifacts/tokenizers/openwebtext-32k-w8-c128
```

This run preserves the four-worker artifacts. CPU profiling is disabled in
both runs, and both use five-second memory sampling. It compares two complete
configurations, not worker count in isolation. These are single runs; filesystem
cache, background activity, and laptop temperature are not controlled.

The recorded eight-worker run uses an external five-second memory guard that
stops its process tree below 3 GiB available system RAM. `memory-guard.json`
records its outcome, and `command.txt` and `benchmark.log` preserve the invocation
and output. The command above alone does not install that guard.

### Observed comparison on this laptop

| Measurement | 4 workers / 32 MiB | 8 workers / 128 MiB |
|---|---:|---:|
| Total wall time | 924.32 s | 755.25 s |
| Pre-tokenization | 618.86 s | 447.70 s |
| Merge initialization | 37.28 s | 39.61 s |
| Merging | 252.56 s | 253.30 s |
| Sampled pre-tokenization peak tree RSS | 1.41 GiB | 5.29 GiB |
| Sampled overall peak tree RSS | 11.78 GiB | 11.73 GiB |

The eight-worker configuration reduced total wall time by 18.3% and
pre-tokenization time by 27.7%. The memory guard observed at least 13.87 GiB
available RAM throughout its samples and did not stop the run. RSS samples can
miss short peaks and double-count shared pages; see the reports for PSS and
sampling limitations.

Both runs produced byte-identical vocabulary and merge JSON files, with 32,000
vocabulary entries and 31,743 merges. Reload, Unicode and special-token round
trips, and streaming/eager equivalence passed six smoke cases. The eight-worker
directory contains `validation.json` and `comparison.json` with the checks and
derived timing ratios. This single comparison does not establish the optimal
worker count or chunk size.

## Eight workers: 128 MiB versus 512 MiB

The follow-up keeps eight workers, the full corpus, 32K vocabulary, special
token, disabled CPU profiling, and five-second memory sampling unchanged:

```bash
.venv/bin/python -m cs336_basics.profile_bpe \
  data/owt_train.txt \
  --num-workers 8 --chunk-size-mib 512 \
  --vocab-size 32000 --special-token '<|endoftext|>' \
  --memory-interval 5 \
  --output-dir artifacts/tokenizers/openwebtext-32k-w8-c512
```

| Measurement | 128 MiB | 512 MiB |
|---|---:|---:|
| Chunks | 89 | 23 |
| Total wall time | 755.25 s | 759.17 s |
| Pre-tokenization | 447.70 s | 477.22 s |
| Merge initialization | 39.61 s | 38.95 s |
| Merging | 253.30 s | 230.36 s |
| Sampled pre-tokenization peak tree RSS | 5.29 GiB | 17.39 GiB |
| Sampled overall peak tree RSS | 11.73 GiB | 17.39 GiB |

The 512 MiB run took 6.6% longer in pre-tokenization and 0.5% longer overall.
Its faster sequential merge stage does not demonstrate a chunk-size benefit:
the learned outputs and merge statistics are identical, and each configuration
was measured only once. Larger chunks provided no observed throughput benefit
and used substantially more memory. This alone does not prove a compute or I/O
bottleneck; load imbalance, memory pressure, and run-to-run variation remain
possible contributors.

The same external guard observed at least 9.00 GiB available RAM and did not
stop the run. As above, the command alone does not include the guard. The new
directory preserves its command, log, guard outcome, report, comparison, and
validation. Vocabulary and merge JSON files are byte-identical to the 128 MiB
run; all six reload/round-trip/streaming smoke cases passed. Keep 128 MiB as the
current measured default among these two configurations.

## Detailed training profiling

See [BPE_TRAINING_PROFILE.md](BPE_TRAINING_PROFILE.md) for the full eight-worker,
128 MiB profiling run, worker-operation breakdown, reproduction command, and
artifact inventory. It also documents a reproduced parent-profile attribution
problem when in-process memory sampling is enabled; the worker profiles and
independent stage wall times are the usable measurements from that run.
