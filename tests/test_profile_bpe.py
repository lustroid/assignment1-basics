import hashlib
import json
import pstats
import subprocess
import sys

import pytest

from cs336_basics.bpe import single_bpe
from cs336_basics.profile_bpe import _save_training_artifacts


def _load_artifacts(directory):
    vocab = {
        int(key): bytes.fromhex(value) for key, value in json.loads((directory / "vocab.json").read_text()).items()
    }
    merges = [
        (bytes.fromhex(left), bytes.fromhex(right))
        for left, right in json.loads((directory / "merges.json").read_text())
    ]
    return vocab, merges


def test_artifacts_preserve_bytes_merge_order_and_longest_ties(tmp_path):
    vocab = {9: b"abc", 2: b"\xff\x00\xfe", 7: "牛".encode(), 5: b"<X>", 0: b"\x00"}
    merges = [(b"\xff", b"\x00"), (b"\xff\x00", b"\xfe")]
    metadata = _save_training_artifacts(tmp_path, vocab, merges)
    assert _load_artifacts(tmp_path) == (vocab, merges)
    assert metadata["longest_token_byte_length"] == 3
    assert metadata["longest_token_ids"] == [2, 5, 7, 9]
    summary = (tmp_path / "longest_tokens.txt").read_text(encoding="utf-8")
    assert "raw=b'\\xff\\x00\\xfe' utf8=<invalid UTF-8>" in summary
    assert "utf8='牛'" in summary
    assert [int(line.split()[0].split("=")[1]) for line in summary.splitlines() if line.startswith("id=")] == [
        2,
        5,
        7,
        9,
    ]


def test_artifacts_replace_existing_outputs_and_support_no_merges(tmp_path):
    _save_training_artifacts(tmp_path, {0: b"previous"}, [(b"pre", b"vious")])
    _save_training_artifacts(tmp_path, {0: b"\x00"}, [])
    assert _load_artifacts(tmp_path) == ({0: b"\x00"}, [])
    assert "previous" not in (tmp_path / "longest_tokens.txt").read_text()


@pytest.mark.parametrize("profile", [False, True])
def test_command_saves_training_outputs(tmp_path, profile):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("aaa 牛 café<|endoftext|>aaa 牛 café<|endoftext|>ababab" * 25_000, encoding="utf-8")
    output = tmp_path / "outputs"
    command = [
        sys.executable,
        "-m",
        "cs336_basics.profile_bpe",
        str(corpus),
        "--num-workers",
        "2",
        "--chunk-size-mib",
        "1",
        "--vocab-size",
        "280",
        "--output-dir",
        str(output),
    ]
    if profile:
        command.append("--profile")
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    report = json.loads(result.stdout)
    assert report == json.loads((output / "report.json").read_text())
    vocab, merges = single_bpe(corpus, 280, ["<|endoftext|>"])
    assert _load_artifacts(output) == (vocab, merges)
    # Preserve the existing hash encoding, independently of artifact formatting.
    signature = json.dumps(
        {
            "vocab": [(key, value.hex()) for key, value in vocab.items()],
            "merges": [(a.hex(), b.hex()) for a, b in merges],
        },
        separators=(",", ":"),
    ).encode("utf-8")
    assert report["result_sha256"] == hashlib.sha256(signature).hexdigest()
    assert report["longest_token_byte_length"] == max(map(len, vocab.values()))
    assert report["chunk_size_bytes"] == 1024 * 1024
    assert report["stage_seconds"]["chunks"] >= 2
    assert report["artifact_paths"]["vocab"] == str(output / "vocab.json")
    if profile:
        stats = pstats.Stats(str(next(output.glob("profiles-*/parent.prof"))))
        assert not any(key[2] == "_save_training_artifacts" for key in stats.stats)
        assert list(output.glob("profiles-*/worker-*.prof"))
