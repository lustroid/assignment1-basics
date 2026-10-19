"""Differential checks for the indexed trainer against full pair recounts."""

import random
from collections import Counter

import pytest
import regex

from cs336_basics.bpe import single_bpe


def _recount_reference(text, vocab_size, special_tokens):
    special_tokens = list(dict.fromkeys(special_tokens))
    vocab = {value: bytes([value]) for value in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    delimiter = "|".join(regex.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
    segments = regex.split(delimiter, text) if special_tokens else [text]
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    words = [
        [bytes([byte]) for byte in match.group().encode("utf-8")]
        for segment in segments
        for match in regex.finditer(pattern, segment)
    ]
    merges = []
    while len(vocab) < vocab_size:
        counts = Counter(pair for word in words for pair in zip(word, word[1:]))
        if not counts:
            break
        pair = max(counts, key=lambda candidate: (counts[candidate], candidate))
        vocab[len(vocab)] = pair[0] + pair[1]
        merges.append(pair)
        for index, word in enumerate(words):
            updated = []
            position = 0
            while position < len(word):
                if position + 1 < len(word) and (word[position], word[position + 1]) == pair:
                    updated.append(pair[0] + pair[1])
                    position += 2
                else:
                    updated.append(word[position])
                    position += 1
            words[index] = updated
    return vocab, merges


@pytest.mark.parametrize(
    "text,special_tokens",
    [
        ("", []),
        ("aaaaaaa aaa aaaa aaaaaaa", []),
        ("abababab ababa babab aba aba", []),
        ("aa zz bb yy", []),
        ("牛😀 café\r\n牛😀 café\n", []),
        ("aa<|endoftext|>aa<|endoftext|><|endoftext|>bb", ["<|endoftext|>"]),
        ("aaa.*aaa[END]aaa", [".*", "[END]"]),
        ("aaa<X><X>bbb<X>ccc", ["<X>", "<X><X>"]),
        ("<X><X><X>", ["<X>", "<X>"]),
    ],
)
def test_indexed_matches_recount(tmp_path, text, special_tokens):
    path = tmp_path / "corpus.txt"
    path.write_bytes(text.encode("utf-8"))
    assert single_bpe(path, 320, special_tokens) == _recount_reference(text, 320, special_tokens)


def test_indexed_matches_randomized_recounts(tmp_path):
    rng = random.Random(336)
    path = tmp_path / "corpus.txt"
    for _ in range(100):
        words = ["".join(rng.choices("abcé牛", k=rng.randrange(1, 16))) for _ in range(12)]
        text = "".join(rng.choice(words) + rng.choice([" ", "\n", "<X>"]) for _ in range(30))
        vocab_size = rng.randrange(257, 330)
        path.write_bytes(text.encode("utf-8"))
        assert single_bpe(path, vocab_size, ["<X>"]) == _recount_reference(text, vocab_size, ["<X>"])


def test_initial_vocabulary_and_no_merges(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("aaaa", encoding="utf-8")
    vocab, merges = single_bpe(path, 257, ["<X>"])
    assert vocab == {**{value: bytes([value]) for value in range(256)}, 256: b"<X>"}
    assert merges == []


def test_invalid_vocabulary_and_empty_special_token(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("aaaa", encoding="utf-8")
    with pytest.raises(ValueError, match="accommodate"):
        single_bpe(path, 256, ["<X>"])
    with pytest.raises(ValueError, match="nonempty"):
        single_bpe(path, 300, [""])
    with pytest.raises(ValueError, match="num_workers"):
        single_bpe(path, 300, [], num_workers=0)


@pytest.mark.parametrize(
    "text,special_tokens",
    [
        ("牛😀 café\n<|endoftext|>aaaa\r\n<|endoftext|>ababab" * 10, ["<|endoftext|>"]),
        ("aaa<X><X>bbb<X>牛😀ccc<X><X>ddd" * 10, ["<X>", "<X><X>"]),
        ("aaa.*bbb[END]ccc.*牛😀" * 10, [".*", "[END]"]),
        ("aaa 牛😀 bbb", []),
        ("aaa 牛😀 bbb", ["<X>"]),
        ("", ["<X>"]),
    ],
)
def test_parallel_matches_recount(tmp_path, text, special_tokens):
    path = tmp_path / "corpus.txt"
    path.write_bytes(text.encode("utf-8"))
    assert single_bpe(path, 320, special_tokens, num_workers=3) == _recount_reference(text, 320, special_tokens)
