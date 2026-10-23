"""Benchmark BPE and optionally profile both the parent and pre-token workers.

Run with: python -m cs336_basics.profile_bpe INPUT --output-dir /tmp/bpe-run
Use a separate invocation per worker count for independent memory measurements.
Each run saves vocab.json and merges.json with hex-encoded bytes, plus a
longest_tokens.txt inspection summary, in the output directory.
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

from .bpe import Pair, single_bpe


def _save_training_artifacts(output_dir: Path, vocab: dict[int, bytes], merges: list[Pair]) -> dict[str, object]:
    """Save exact token bytes and merge order, with a readable longest-token summary."""
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.json"
    summary_path = output_dir / "longest_tokens.txt"
    vocab_path.write_text(
        json.dumps({str(key): value.hex() for key, value in sorted(vocab.items())}, indent=2) + "\n",
        encoding="utf-8",
    )
    merges_path.write_text(
        json.dumps([(left.hex(), right.hex()) for left, right in merges], indent=2) + "\n",
        encoding="utf-8",
    )

    longest_length = max((len(token) for token in vocab.values()), default=0)
    longest_ids = sorted(key for key, token in vocab.items() if len(token) == longest_length)
    lines = [f"Longest token byte length: {longest_length}", ""]
    for token_id in longest_ids:
        token = vocab[token_id]
        try:
            readable = repr(token.decode("utf-8"))
        except UnicodeDecodeError:
            readable = "<invalid UTF-8>"
        lines.append(f"id={token_id} bytes={len(token)} raw={token!r} utf8={readable}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "artifact_paths": {
            "vocab": str(vocab_path.resolve()),
            "merges": str(merges_path.resolve()),
            "longest_tokens": str(summary_path.resolve()),
        },
        "longest_token_byte_length": longest_length,
        "longest_token_ids": longest_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--chunk-size-mib", type=int, default=128, help="Target input bytes per chunk in MiB (default: 128)"
    )
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
            chunk_size_bytes=args.chunk_size_mib * 1024 * 1024,
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
    # Training timing, memory counters, and profiling have already stopped.
    artifacts = _save_training_artifacts(args.output_dir, vocab, merges)
    report = {
        "input_path": str(args.input_path.resolve()),
        "input_bytes": args.input_path.stat().st_size,
        "requested_workers": args.num_workers,
        "chunk_size_bytes": args.chunk_size_mib * 1024 * 1024,
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
        **artifacts,
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
