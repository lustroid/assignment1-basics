import json
import pstats
import subprocess
import sys

import pytest

from cs336_basics.bpe_tokenizer import Tokenizer
from cs336_basics.profile_bpe import _save_training_artifacts
from cs336_basics.profile_tokenizer import _consume


@pytest.mark.parametrize("text", ["", "aaaé\r\n<s> aaa<s>\n"])
def test_benchmark_cli_checks_outputs_and_separates_profiles(tmp_path, text):
    vocab = {index: bytes([index]) for index in range(256)}
    vocab[256] = b"aa"
    merges = [(b"a", b"a")]
    _save_training_artifacts(tmp_path, vocab, merges)
    corpus = tmp_path / "input.txt"
    corpus.write_bytes(text.encode())
    output = tmp_path / "benchmark"
    subprocess.run([
        sys.executable, "-m", "cs336_basics.profile_tokenizer", str(corpus),
        "--vocab", str(tmp_path / "vocab.json"), "--merges", str(tmp_path / "merges.json"),
        "--output-dir", str(output), "--fragment-chars", "1", "8", "--repeats", "2",
        "--special-token", "<s>", "--profile", "--profile-fragment-chars", "8",
    ], check=True, capture_output=True, text=True)
    report_path, = output.glob("run-*/report.json")
    report = json.loads(report_path.read_text())
    expected_count, expected_hash = _consume(Tokenizer(vocab, merges, ["<s>"]).encode(text))
    assert report["results_match"]
    assert report["input_bytes"] == len(text.encode())
    assert report["token_count"] == expected_count
    assert report["token_ids_sha256"] == expected_hash
    assert len(report["trials"]) == 8
    assert len(report["summary"]) == 3
    assert all(row["trials"] == 2 for row in report["summary"])
    for row in report["trials"]:
        assert row["token_count"] == expected_count
        assert row["token_ids_sha256"] == expected_hash
        assert row["peak_rss_bytes"] >= row["baseline_peak_rss_bytes"]
        if row["profiled"]:
            stats = pstats.Stats(row["profile_path"])
            assert any(name == "_encode_pretoken" for _, _, name in stats.stats) or not text


def test_invalid_fragment_size_rejected(tmp_path):
    result = subprocess.run([
        sys.executable, "-m", "cs336_basics.profile_tokenizer", "unused",
        "--vocab", "unused", "--merges", "unused", "--output-dir", str(tmp_path),
        "--fragment-chars", "0",
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "must be positive" in result.stderr
    assert not list(tmp_path.iterdir())
