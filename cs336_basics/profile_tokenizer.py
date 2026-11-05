"""Compare eager/streaming encoding in isolated processes; optionally run cProfile.

Accepts the hex-encoded vocab.json and merges.json saved by profile_bpe.
Timing includes UTF-8 file reading and a common bounded checksum sink, but excludes
model loading, warmup, process startup, and profile/report serialization.
"""

import argparse
import cProfile
import gc
import hashlib
import json
import platform
import pstats
import random
import resource
import statistics
import struct
import subprocess
import sys
import tempfile
from itertools import islice
from pathlib import Path
from time import perf_counter

import psutil
import regex

from .bpe_tokenizer import Tokenizer


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _consume(ids):
    """Hash little-endian uint32 IDs in bounded batches, without retaining output."""
    iterator = iter(ids)
    digest = hashlib.sha256()
    count = 0
    while batch := list(islice(iterator, 4096)):
        digest.update(struct.pack(f"<{len(batch)}I", *batch))
        count += len(batch)
    return count, digest.hexdigest()


def _worker(job):
    tokenizer = Tokenizer.from_files(job["vocab"], job["merges"], job["special_tokens"])
    # Warm both paths equally before timing, without loading the corpus.
    tokenizer.encode("Hello, world!\n")
    list(tokenizer.encode_iterable(["Hello, ", "world!\n"]))
    gc.collect()
    rss_scale = 1 if sys.platform == "darwin" else 1024
    baseline_rss = psutil.Process().memory_info().rss
    baseline_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_scale
    profiler = cProfile.Profile() if job["profile_prefix"] else None
    started = perf_counter()
    if profiler is not None:
        profiler.enable()
    with open(job["input"], encoding="utf-8", newline="") as source:
        if job["fragment_chars"] is None:
            result = _consume(tokenizer.encode(source.read()))
        else:
            fragments = iter(lambda: source.read(job["fragment_chars"]), "")
            result = _consume(tokenizer.encode_iterable(fragments))
    if profiler is not None:
        profiler.disable()
    elapsed = perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_scale
    row = {
        "case": job["case"], "fragment_chars": job["fragment_chars"],
        "profiled": profiler is not None, "wall_seconds": elapsed,
        "mb_per_second": job["input_bytes"] / 1_000_000 / elapsed,
        "tokens_per_second": result[0] / elapsed,
        "token_count": result[0], "token_ids_sha256": result[1],
        "baseline_rss_bytes": baseline_rss, "baseline_peak_rss_bytes": baseline_peak,
        "peak_rss_bytes": peak, "peak_growth_bytes": max(0, peak - baseline_peak),
        "vocab_size": len(tokenizer.vocab), "merge_count": len(tokenizer.merges),
    }
    if profiler is not None:
        prefix = job["profile_prefix"]
        profiler.dump_stats(prefix + ".prof")
        with open(prefix + ".txt", "w", encoding="utf-8") as output:
            stats = pstats.Stats(profiler, stream=output)
            stats.strip_dirs().sort_stats("cumulative").print_stats(30)
            stats.sort_stats("tottime").print_stats(30)
        # Save all function rows so later analysis need not parse formatted text.
        stats = pstats.Stats(profiler)
        functions = [
            {"file": file, "line": line, "function": name, "primitive_calls": cc,
             "calls": nc, "self_seconds": tt, "cumulative_seconds": ct}
            # pstats exposes this full-precision table at runtime; its stub omits it.
            for (file, line, name), (cc, nc, tt, ct, _) in stats.stats.items()  # ty: ignore[unresolved-attribute]
        ]
        Path(prefix + "-functions.json").write_text(json.dumps(functions, indent=2) + "\n")
        row["profile_path"] = prefix + ".prof"
    return row


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker-json":
        print(json.dumps(_worker(json.loads(sys.argv[2]))))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fragment-chars", type=int, nargs="+", default=[1, 64, 4096, 65536])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--special-token", action="append", default=None)
    parser.add_argument("--no-special-tokens", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Extra profiled runs: eager and one streaming size")
    parser.add_argument("--profile-fragment-chars", type=int, default=4096)
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 1 for size in args.fragment_chars) or args.profile_fragment_chars < 1:
        parser.error("Repeat counts and fragment sizes must be positive")
    if args.no_special_tokens and args.special_token:
        parser.error("--no-special-tokens cannot be combined with --special-token")
    for path in (args.input, args.vocab, args.merges):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=args.output_dir)).resolve()
    sizes = list(dict.fromkeys(args.fragment_chars))
    base = {
        "input": str(args.input.resolve()), "input_bytes": args.input.stat().st_size,
        "vocab": str(args.vocab.resolve()), "merges": str(args.merges.resolve()),
        "special_tokens": [] if args.no_special_tokens else args.special_token or ["<|endoftext|>"],
    }
    cases = [("eager", None)] + [(f"stream-{size}", size) for size in sizes]
    rows = []

    def run(case, size, profile=False):
        job = dict(base, case=case, fragment_chars=size,
                   profile_prefix=str(run_dir / case) if profile else None)
        result = subprocess.run(
            [sys.executable, "-m", "cs336_basics.profile_tokenizer", "--worker-json", json.dumps(job)],
            check=True, capture_output=True, text=True,
        )
        row = json.loads(result.stdout)
        rows.append(row)
        print(f"{case}{' [profiled]' if profile else ''}: {row['wall_seconds']:.3f}s, "
              f"{row['mb_per_second']:.3f} MB/s, peak RSS {row['peak_rss_bytes'] / 2**20:.2f} MiB", flush=True)
        (run_dir / "trials.json").write_text(json.dumps(rows, indent=2) + "\n")
        if (row["token_count"], row["token_ids_sha256"]) != (rows[0]["token_count"], rows[0]["token_ids_sha256"]):
            raise RuntimeError(f"Token output mismatch: {case}; details in {run_dir}")

    rng = random.Random(0)
    for _ in range(args.repeats):
        order = cases.copy()
        rng.shuffle(order)
        for case, size in order:
            run(case, size)
    if args.profile:
        run("eager", None, True)
        run(f"stream-{args.profile_fragment_chars}", args.profile_fragment_chars, True)

    summary = []
    for case, size in cases:
        trials = [row for row in rows if row["case"] == case and not row["profiled"]]
        median = statistics.median(row["wall_seconds"] for row in trials)
        summary.append({
            "case": case, "fragment_chars": size, "trials": len(trials),
            "median_seconds": median, "min_seconds": min(row["wall_seconds"] for row in trials),
            "max_seconds": max(row["wall_seconds"] for row in trials),
            "median_mb_per_second": base["input_bytes"] / 1_000_000 / median,
            "max_peak_rss_bytes": max(row["peak_rss_bytes"] for row in trials),
            "median_peak_growth_bytes": statistics.median(row["peak_growth_bytes"] for row in trials),
        })
    report = {
        **base, "input_sha256": _sha256(args.input), "vocab_sha256": _sha256(args.vocab),
        "merges_sha256": _sha256(args.merges), "python": sys.version, "platform": platform.platform(),
        "regex_version": regex.__version__, "tokenizer_source_sha256": _sha256(Path(__file__).with_name("bpe_tokenizer.py")),
        "token_count": rows[0]["token_count"], "token_ids_sha256": rows[0]["token_ids_sha256"],
        "vocab_size": rows[0]["vocab_size"], "merge_count": rows[0]["merge_count"],
        "results_match": True, "summary": summary, "trials": rows,
        "timing_note": "Sequential fresh processes, shuffled case order, warm filesystem cache not controlled. "
                       "Includes UTF-8 file reading and bounded ID checksum sink; excludes model loading and warmup. "
                       "MB/s uses decimal bytes. Fragment sizes are Unicode characters. Profiled runs excluded from medians.",
        "memory_note": "OS lifetime peak RSS includes interpreter/model loading; each trial uses a fresh process. "
                       "Peak growth is increase over pre-measurement high-water mark, not exact allocated memory. "
                       "Streaming output is drained; collecting IDs would add output memory.",
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {run_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
