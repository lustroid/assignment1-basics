"""Benchmark BPE and optionally profile both the parent and pre-token workers.

Run with: python -m cs336_basics.profile_bpe INPUT --output-dir /tmp/bpe-run
Use a separate invocation per worker count for independent memory measurements.
"""

import argparse
import cProfile
import hashlib
import json
import pstats
import resource
import sys
import tempfile
from pathlib import Path
from time import perf_counter

from .bpe import single_bpe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--special-token", action="append", default=None)
    parser.add_argument("--profile", action="store_true", help="Enable cProfile (adds runtime overhead)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = Path(tempfile.mkdtemp(prefix="profiles-", dir=args.output_dir)) if args.profile else None
    timings: dict[str, float] = {}
    profiler = cProfile.Profile() if args.profile else None
    started = perf_counter()
    try:
        if profiler is not None:
            profiler.enable()
        vocab, merges = single_bpe(
            args.input_path,
            args.vocab_size,
            args.special_token if args.special_token is not None else ["<|endoftext|>"],
            num_workers=args.num_workers,
            profile_dir=profile_dir,
            timings=timings,
        )
    finally:
        if profiler is not None:
            profiler.disable()
        elapsed = perf_counter() - started

    # Linux reports ru_maxrss in KiB; macOS reports bytes. Child usage on Linux
    # records the largest individual child's peak, not a simultaneous tree sum.
    rss_scale = 1 if sys.platform == "darwin" else 1024
    parent_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_scale
    child_peak_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * rss_scale
    signature = json.dumps(
        {
            "vocab": [(key, value.hex()) for key, value in vocab.items()],
            "merges": [(a.hex(), b.hex()) for a, b in merges],
        },
        separators=(",", ":"),
    ).encode("utf-8")
    report = {
        "input_path": str(args.input_path.resolve()),
        "input_bytes": args.input_path.stat().st_size,
        "requested_workers": args.num_workers,
        "vocabulary_size": len(vocab),
        "merge_count": len(merges),
        "result_sha256": hashlib.sha256(signature).hexdigest(),
        "profiling_enabled": args.profile,
        "profile_directory": str(profile_dir) if profile_dir is not None else None,
        "wall_seconds": elapsed,
        "stage_seconds": timings,
        "parent_peak_rss_bytes": parent_peak_rss,
        "largest_child_peak_rss_bytes": child_peak_rss,
        "memory_note": "OS process high-water marks, including interpreter overhead; child peak is not aggregate memory.",
    }
    report_text = json.dumps(report, indent=2)
    (args.output_dir / "report.json").write_text(report_text + "\n", encoding="utf-8")
    print(report_text)
    if profiler is not None:
        assert profile_dir is not None
        profiler.dump_stats(str(profile_dir / "parent.prof"))
        for label, files in [
            ("Parent", [profile_dir / "parent.prof"]),
            ("Pre-tokenization workers (summed CPU profiles)", sorted(profile_dir.glob("worker-*.prof"))),
        ]:
            if not files:
                continue
            with (args.output_dir / f"{label.split()[0].lower()}-profile.txt").open("w", encoding="utf-8") as output:
                stats = pstats.Stats(*(str(path) for path in files), stream=output)
                stats.strip_dirs().sort_stats("cumulative").print_stats(25)


if __name__ == "__main__":
    main()
