import json
import subprocess
import sys

import pytest

from cs336_basics.bpe_tokenizer import Tokenizer
from cs336_basics.profile_bpe import _save_training_artifacts
from cs336_basics.profile_tokenizer_stages import _special_digest


@pytest.mark.parametrize("text", ["", "aa aa<s>aa aa"])
def test_stage_cli_and_repetition(tmp_path, text):
    _save_training_artifacts(tmp_path, {i: bytes([i]) for i in range(256)}, [])
    corpus = tmp_path / "input.txt"
    corpus.write_text(text)
    output = tmp_path / "out"
    subprocess.run([
        sys.executable, "-m", "cs336_basics.profile_tokenizer_stages", str(corpus),
        "--vocab", str(tmp_path / "vocab.json"), "--merges", str(tmp_path / "merges.json"),
        "--output-dir", str(output), "--special-token", "<s>", "--fragment-chars", "1", "--profile",
    ], check=True, capture_output=True, text=True)
    report = json.loads((output / "stages.json").read_text())
    rows = {row["mode"]: row for row in report["measurements"]}
    for prefix in ["special", "pretoken"]:
        assert rows[prefix + "-regex"]["sha256"] == rows[prefix + "-stream"]["sha256"]
    assert rows["special-stream"]["count"] == (1 if text else 0)
    assert rows["pretoken-stream"]["count"] == (4 if text else 0)
    assert rows["repetition"]["unique_pretokens"] == (2 if text else 0)
    assert rows["repetition"]["duplicate_fraction"] == (0.5 if text else 0)
    assert rows["repetition"]["top_type_coverage"]["10"] == (1 if text else 0)
    assert len(list(output.glob("*.prof"))) == 5


def test_special_diagnostic_tracks_positions_and_overlaps():
    tokenizer = Tokenizer({i: bytes([i]) for i in range(256)}, [], ["abcd", "b", "🙂", "🙂🙂"])
    text = "abcx🙂🙂b🙂abcd"
    native = _special_digest(tokenizer, text, 1, False)
    streamed = _special_digest(tokenizer, text, 1, True)
    assert native == streamed
    assert native["count"] == 5
