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
import platform
import pstats
import resource
import sys
import tempfile
import threading
from pathlib import Path
from time import perf_counter

import psutil

from .bpe import Pair, single_bpe


class _MemorySampler:
    """Sample live parent/descendant memory; do not sum lifetime high-water marks."""

    def __init__(self, interval: float, output_dir: Path):
        self.interval = interval
        self.process = psutil.Process()
        self.started = perf_counter()
        self.stage = "startup"
        self.events = []
        self.samples = []
        self.path = output_dir / "memory-samples.jsonl"
        self.output = self.path.open("w", encoding="utf-8")
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def mark(self, stage):
        self.events.append({"stage": stage, "seconds": perf_counter() - self.started})
        self.stage = stage

    def sample(self):
        stage = self.stage
        parent_rss = 0
        children_rss = 0
        total_pss = 0
        pss_available = True
        child_count = 0
        try:
            processes = [self.process, *self.process.children(recursive=True)]
        except psutil.Error:
            return
        for process in processes:
            try:
                info = process.memory_full_info()
            except psutil.Error:
                continue
            if process.pid == self.process.pid:
                parent_rss = info.rss
            else:
                children_rss += info.rss
                child_count += 1
            pss = getattr(info, "pss", None)
            if pss is None:
                pss_available = False
            else:
                total_pss += pss
        self.samples.append({
            "seconds": perf_counter() - self.started,
            "stage": stage if stage == self.stage else "transition",
            "parent_rss_bytes": parent_rss, "children_rss_bytes": children_rss,
            "tree_rss_bytes": parent_rss + children_rss,
            "tree_pss_bytes": total_pss if pss_available else None,
            "children": child_count, "system_available_bytes": psutil.virtual_memory().available,
        })
        # Preserve diagnostic evidence even if an external memory guard stops a run.
        self.output.write(json.dumps(self.samples[-1]) + "\n")
        self.output.flush()

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def start(self):
        self.sample()
        self.thread.start()

    def finish(self, output_dir):
        self.stop_event.set()
        self.thread.join()
        self.sample()
        self.output.close()
        stages = {}
        for stage in dict.fromkeys(row["stage"] for row in self.samples):
            rows = [row for row in self.samples if row["stage"] == stage]
            stages[stage] = {
                "samples": len(rows),
                **{key: max((row[key] for row in rows if row[key] is not None), default=None)
                   for key in ["parent_rss_bytes", "children_rss_bytes", "tree_rss_bytes", "tree_pss_bytes"]},
            }
        peak = max(self.samples, key=lambda row: row["tree_rss_bytes"])
        return {
            "interval_seconds": self.interval, "sample_count": len(self.samples),
            "samples_path": str(self.path.resolve()), "stage_events": self.events,
            "stages": stages, "peak_tree_rss_sample": peak,
            "peak_tree_pss_bytes": max((row["tree_pss_bytes"] for row in self.samples
                                        if row["tree_pss_bytes"] is not None), default=None),
            "note": "Sampled parent plus live descendants (including process-pool helpers). RSS can double-count "
                    "shared pages; PSS apportions them. Sequential process reads are not an atomic snapshot. "
                    "Sampling may miss short peaks; callbacks mark phases but do not force samples. "
                    "Resident mmap pages count, including reclaimable file-backed pages during boundary scanning.",
        }


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
    parser.add_argument("--memory-interval", type=float, default=0,
                        help="Sample parent/worker RSS and PSS every N seconds; 0 disables sampling")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.memory_interval < 0:
        parser.error("--memory-interval must be nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = Path(tempfile.mkdtemp(prefix="profiles-", dir=args.output_dir)) if args.profile else None
    timings: dict[str, float] = {}
    statistics: dict[str, int] = {}
    sampler = _MemorySampler(args.memory_interval, args.output_dir) if args.memory_interval else None
    if sampler is not None:
        sampler.start()
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
            statistics=statistics,
            stage_callback=sampler.mark if sampler is not None else None,
        )
    finally:
        if profiler is not None:
            profiler.disable()
        elapsed = perf_counter() - started
        memory = sampler.finish(args.output_dir) if sampler is not None else None

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
        "training_statistics": statistics,
        "sampled_memory": memory,
        "python": sys.version,
        "platform": platform.platform(),
        "trainer_source_sha256": hashlib.sha256(Path(__file__).with_name("bpe.py").read_bytes()).hexdigest(),
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
            ("Pre-tokenization workers (summed elapsed profiles)", sorted(profile_dir.glob("worker-*.prof"))),
        ]:
            if not files:
                continue
            with (args.output_dir / f"{label.split()[0].lower()}-profile.txt").open("w", encoding="utf-8") as output:
                stats = pstats.Stats(*(str(path) for path in files), stream=output)
                stats.strip_dirs().sort_stats("cumulative").print_stats(25)
                stats.sort_stats("tottime").print_stats(25)


if __name__ == "__main__":
    main()
