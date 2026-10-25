"""Encoding checks independent of the unfinished decoding/streaming methods."""

import random

import pytest
import tiktoken

from cs336_basics.bpe_tokenizer import Tokenizer
from cs336_basics.profile_bpe import _save_training_artifacts

from .common import FIXTURES_PATH
from .test_tokenizer import MERGES_PATH, VOCAB_PATH, get_tokenizer_from_vocab_merges_path


def byte_vocab(merges=()):
    vocab = {value: bytes([value]) for value in range(256)}
    for left, right in merges:
        if left + right not in vocab.values():
            vocab[len(vocab)] = left + right
    return vocab


def scan_encode(data, merges, vocab):
    """Independent oracle: rescan the whole sequence for each highest-priority pair."""
    tokens = [bytes([value]) for value in data]
    while True:
        for pair in merges:
            position = next(
                (index for index in range(len(tokens) - 1) if tuple(tokens[index:index + 2]) == pair), None
            )
            if position is not None:
                tokens[position:position + 2] = [pair[0] + pair[1]]
                break
        else:
            break
    inverse = {token: token_id for token_id, token in vocab.items()}
    return [inverse[token] for token in tokens]


def test_special_tokens_preserve_ids_and_inputs():
    vocab = byte_vocab()
    vocab[500] = b"<s>"
    original = vocab.copy()
    specials = ["<s>", "<s><s>", "[.*]", "<s>"]
    tokenizer = Tokenizer(vocab, [], specials)
    assert tokenizer.encode("<s><s><s>[.*]") == [501, 500, 502]
    assert vocab == original
    assert specials == ["<s>", "<s><s>", "[.*]", "<s>"]
    assert tokenizer.encode("") == []


def test_empty_special_token_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        Tokenizer(byte_vocab(), [], [""])


def test_from_files_roundtrip_and_subclass(tmp_path):
    merges = [(b"\xc3", b"\xa9"), (b"a", b"a")]
    vocab = byte_vocab(merges)
    _save_training_artifacts(tmp_path, vocab, merges)

    class ChildTokenizer(Tokenizer):
        pass

    tokenizer = ChildTokenizer.from_files(tmp_path / "vocab.json", tmp_path / "merges.json", ["<s>"])
    assert isinstance(tokenizer, ChildTokenizer)
    assert tokenizer.merges == merges
    assert all(tokenizer.vocab[key] == value for key, value in vocab.items())
    assert tokenizer.encode("éaaa<s>") == [256, 257, 97, 258]


def test_rank_and_stale_candidates():
    # (b,c) wins despite its later position; the old (a,b) entry becomes stale.
    merges = [(b"b", b"c"), (b"a", b"b"), (b"a", b"bc"), (b"abc", b"d")]
    vocab = byte_vocab(merges)
    tokenizer = Tokenizer(vocab, merges)
    assert tokenizer.encode("abcdabc") == [259, 258]


def test_overlap_merges_left_to_right():
    merges = [(b"a", b"a"), (b"aa", b"a"), (b"aa", b"aa")]
    vocab = byte_vocab(merges)
    tokenizer = Tokenizer(vocab, merges)
    for length in range(1, 100):
        text = "a" * length
        assert tokenizer.encode(text) == scan_encode(text.encode(), merges, vocab)


def test_no_merge_across_pretokens_or_special_tokens():
    merges = [(b"a", b"!"), (b"a", b"b")]
    tokenizer = Tokenizer(byte_vocab(merges), merges, ["<s>"])
    assert tokenizer.encode("a!") == [97, 33]
    assert tokenizer.encode("a<s>b") == [97, 258, 98]


def test_randomized_against_full_scans():
    rng = random.Random(42)
    for _ in range(100):
        symbols = [b"a", b"b", b"c"]
        merges = []
        for _ in range(30):
            pair = (rng.choice(symbols), rng.choice(symbols))
            if pair not in merges:
                merges.append(pair)
                if pair[0] + pair[1] not in symbols:
                    symbols.append(pair[0] + pair[1])
        vocab = byte_vocab(merges)
        tokenizer = Tokenizer(vocab, merges)
        for _ in range(10):
            text = "".join(rng.choices("abc", k=rng.randrange(100)))
            assert tokenizer.encode(text) == scan_encode(text.encode(), merges, vocab)


@pytest.fixture(scope="module")
def gpt2_tokenizers():
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH, ["<|endoftext|>"])
    return tokenizer, tiktoken.get_encoding("gpt2")


@pytest.mark.parametrize("text", [
    "", "hello", "Hello, world! It's 2026.", "é中文🙂𐍈", "   \n\t  hello\r\n  ",
    "<|endoftext|><|endoftext|>hello<|endoftext|>", "a" * 1000,
])
def test_encode_matches_gpt2(gpt2_tokenizers, text):
    tokenizer, reference = gpt2_tokenizers
    assert tokenizer.encode(text) == reference.encode(text, allowed_special={"<|endoftext|>"})


@pytest.mark.parametrize("filename", ["tinystories_sample.txt", "address.txt"])
def test_fixture_encode_matches_gpt2(gpt2_tokenizers, filename):
    tokenizer, reference = gpt2_tokenizers
    text = (FIXTURES_PATH / filename).read_text(encoding="utf-8")
    assert tokenizer.encode(text) == reference.encode(text, allowed_special={"<|endoftext|>"})
