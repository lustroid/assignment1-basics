"""Isolate tokenizer front-end costs and measure pre-token repetition.

These diagnostic workers preload text and ordinary-region spans. Their memory
usage is NOT a streaming-memory measurement. Use profile_tokenizer for that.
"""

import argparse
import cProfile
import hashlib
import json
import pstats
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

from .bpe import _PRETOKEN_PATTERN
from .bpe_tokenizer import Tokenizer, _iter_regex_pretokens
from .profile_tokenizer import _sha256


def _regions(tokenizer, text):
    start = 0
    spans = []
    if tokenizer._special_pattern is not None:
        for match in tokenizer._special_pattern.finditer(text):
            spans.append((start, match.start()))
            start = match.end()
    spans.append((start, len(text)))
    return spans


def _pretokens(text, spans, size, streaming):
    for start, end in spans:
        if streaming:
            fragments = (text[offset:min(offset + size, end)] for offset in range(start, end, size))
            yield from _iter_regex_pretokens(fragments)
        else:
            for match in _PRETOKEN_PATTERN.finditer(text, start, end):
                yield match.group()


def _special_digest(tokenizer, text, size, streaming):
    def matches():
        if streaming:
            fragments = (text[offset:offset + size] for offset in range(0, len(text), size))
            position = 0
            for special, part in tokenizer._iter_special_parts(fragments):
                if special:
                    yield position, part
                position += len(part)
            assert position == len(text)
        elif tokenizer._special_pattern is not None:
            for match in tokenizer._special_pattern.finditer(text):
                yield match.start(), match.group()

    digest = hashlib.sha256()
    count = 0
    for position, token in matches():
        digest.update(struct.pack("<QQ", position, len(token)))
        digest.update(token.encode("utf-8"))
        count += 1
    return {"count": count, "sha256": digest.hexdigest()}


def _pretoken_digest(pretokens):
    digest = hashlib.sha256()
    count = 0
    for token in pretokens:
        data = token.encode("utf-8")
        digest.update(struct.pack("<Q", len(data)))
        digest.update(data)
        count += 1
    return {"count": count, "sha256": digest.hexdigest()}


def _repetition(counts):
    total = counts.total()
    unique = len(counts)
    common = counts.most_common()
    return {
        "occurrences": total, "unique_pretokens": unique,
        "duplicate_occurrences": total - unique,
        "duplicate_fraction": (total - unique) / total if total else 0,
        "mean_occurrences_per_unique": total / unique if unique else 0,
        "singleton_types": sum(count == 1 for count in counts.values()),
        "max_pretoken_bytes": max(map(len, counts), default=0),
        "top_type_coverage": {
            str(k): sum(count for _, count in common[:k]) / total if total else 0
            for k in [10, 100, 1000, 10000]
        },
        "top_20": [{"bytes_repr": repr(token), "count": count} for token, count in common[:20]],
        "note": "Duplicate fraction is an unlimited-cache upper bound on avoidable BPE calls, "
                "not a measured cache hit rate, memory requirement, or end-to-end speedup.",
    }


def _worker(job):
    tokenizer = Tokenizer.from_files(job["vocab"], job["merges"], job["special_tokens"])
    with open(job["input"], encoding="utf-8", newline="") as source:
        text = source.read()
    spans = _regions(tokenizer, text)
    mode, size = job["mode"], job["fragment_chars"]
    profile = cProfile.Profile() if job["profile"] else None
    if profile is not None:
        profile.enable()
    started = perf_counter()
    if mode.startswith("special-"):
        result = _special_digest(tokenizer, text, size, mode == "special-stream")
    elif mode.startswith("pretoken-"):
        result = _pretoken_digest(_pretokens(text, spans, size, mode == "pretoken-stream"))
    else:
        counts = Counter(token.encode("utf-8") for token in _pretokens(text, spans, size, False))
        result = {}
    elapsed = perf_counter() - started
    if profile is not None:
        profile.disable()
        prefix = Path(job["output_dir"]) / mode
        profile.dump_stats(str(prefix) + ".prof")
        with open(str(prefix) + ".txt", "w") as output:
            stats = pstats.Stats(profile, stream=output)
            stats.strip_dirs().sort_stats("cumulative").print_stats(30)
            stats.sort_stats("tottime").print_stats(30)
    if mode == "repetition":
        result = _repetition(counts)
    return {
        "mode": mode, "profiled": profile is not None, "seconds": elapsed,
        "input_characters": len(text), "loaded_text_object_bytes": sys.getsizeof(text), **result,
    }


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker-json":
        print(json.dumps(_worker(json.loads(sys.argv[2]))))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fragment-chars", type=int, default=4096)
    parser.add_argument("--special-token", action="append")
    modes = ["special-regex", "special-stream", "pretoken-regex", "pretoken-stream", "repetition"]
    parser.add_argument("--modes", nargs="+", choices=modes, default=modes)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.fragment_chars < 1:
        parser.error("Fragment size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    job = {
        "input": str(args.input.resolve()), "vocab": str(args.vocab.resolve()),
        "merges": str(args.merges.resolve()), "special_tokens": args.special_token or ["<|endoftext|>"],
        "fragment_chars": args.fragment_chars, "profile": args.profile,
        "output_dir": str(args.output_dir.resolve()),
    }
    rows = []
    for mode in args.modes:
        completed = subprocess.run(
            [sys.executable, "-m", "cs336_basics.profile_tokenizer_stages", "--worker-json", json.dumps(dict(job, mode=mode))],
            capture_output=True, text=True, check=True,
        )
        row = json.loads(completed.stdout)
        rows.append(row)
        print(json.dumps(row), flush=True)
        report = {
            **job, "input_bytes": args.input.stat().st_size, "input_sha256": _sha256(args.input),
            "vocab_sha256": _sha256(args.vocab), "merges_sha256": _sha256(args.merges),
            "python": sys.version,
            "measurements": rows,
            "method": "One trial per stage in fresh sequential processes. Text and ordinary-region spans "
                      "are prepared before timing. Special stages compare special positions/strings. "
                      "Pre-token stages compare length-delimited UTF-8 tokens over identical ordinary regions, "
                      "excluding special matching and BPE. Timings include their respective checksum sinks. "
                      "Repetition counting is a separate diagnostic allocation, excluded from memory benchmarks.",
        }
        (args.output_dir / "stages.json").write_text(json.dumps(report, indent=2) + "\n")
    validated = []
    for prefix in ["special", "pretoken"]:
        matching = [row for row in rows if row["mode"].startswith(prefix + "-")]
        if len(matching) == 2:
            if (matching[0]["count"], matching[0]["sha256"]) != (matching[1]["count"], matching[1]["sha256"]):
                raise RuntimeError(f"{prefix} output mismatch")
            validated.append(prefix)
    repetition = next((row for row in rows if row["mode"] == "repetition"), None)
    pretoken = next((row for row in rows if row["mode"].startswith("pretoken-")), None)
    if repetition is not None and pretoken is not None:
        assert repetition["occurrences"] == pretoken["count"]
    report["validated_stage_pairs"] = validated
    (args.output_dir / "stages.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
