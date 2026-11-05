"""Check streaming token IDs without depending on the unfinished decode method."""

import random
import tracemalloc
from itertools import islice, repeat

import pytest
import tiktoken

from cs336_basics.bpe_tokenizer import Tokenizer

from .common import FIXTURES_PATH
from .test_bpe_tokenizer_encoding import byte_vocab
from .test_tokenizer import MERGES_PATH, VOCAB_PATH, get_tokenizer_from_vocab_merges_path


def small_tokenizer(specials):
    merges = [(b"a", b"a"), (b"aa", b"a"), (b"b", b"c"), (b" ", b"a"), (b"'", b"r"), (b"'r", b"e")]
    return Tokenizer(byte_vocab(merges), merges, specials)


@pytest.mark.parametrize("specials,text", [
    ([], "aaa we're 牛🙂\n\t  abc"),
    (["<s>", "<s><s>"], "aaa<s><s>abc<s>  aaa"),
    (["abcd", "b"], "abcx abcd abc"),
    (["ab", "abcd", "bc"], "abcx abcd ab abc"),
    (["[.*]", "🙂牛", "\n\n", "aaa"], "a[.*]a🙂牛\n\naaaaa"),
    (["'re"], "a'reaaa're"),
    (["<s>"], "aaa  <s>aaa<s"),
    (["ab", "ab "], "ab ab"),
    (["\n", "\n\n"], "aaa\n\naaa\n"),
])
def test_arbitrary_fragment_boundaries(specials, text):
    tokenizer = small_tokenizer(specials)
    expected = tokenizer.encode(text)
    assert list(tokenizer.encode_iterable(iter(text))) == expected
    for split in range(len(text) + 1):
        fragments = iter(["", text[:split], "", text[split:], ""])
        assert list(tokenizer.encode_iterable(fragments)) == expected


@pytest.mark.parametrize("specials", [[], ["<s>"]])
def test_empty_stream(specials):
    tokenizer = small_tokenizer(specials)
    assert list(tokenizer.encode_iterable(iter([]))) == []
    assert list(tokenizer.encode_iterable(["", ""])) == []


def test_randomized_specials_and_fragmentation():
    rng = random.Random(45)
    alphabet = "abc 'r\n🙂"
    for _ in range(300):
        specials = ["".join(rng.choices(alphabet, k=rng.randrange(1, 9))) for _ in range(5)]
        specials.extend([specials[0][:1], specials[0] + specials[1]])
        tokenizer = small_tokenizer(specials)
        text = "".join(rng.choices(alphabet, k=80)) + "".join(specials)
        expected = tokenizer.encode(text)
        assert list(tokenizer.encode_iterable(iter(text))) == expected
        cuts = sorted({0, len(text), *(rng.randrange(len(text) + 1) for _ in range(20))})
        assert list(tokenizer.encode_iterable(text[a:b] for a, b in zip(cuts, cuts[1:]))) == expected


@pytest.mark.parametrize("specials", [[], ["<s>", "<s><s>"]])
def test_lazy_and_independent_iterators(specials):
    tokenizer = small_tokenizer(specials)

    def fragments():
        yield "aaa aaa aaa "
        raise AssertionError("Consumed more input before producing available IDs")

    first = tokenizer.encode_iterable(fragments())
    second = tokenizer.encode_iterable(iter("abc aaa"))
    assert next(first) == tokenizer.encode("aaa")[0]
    assert list(second) == tokenizer.encode("abc aaa")
    first.close()
    # An infinite source must be usable without waiting for EOF.
    stream = tokenizer.encode_iterable(repeat("aaa "))
    assert len(list(islice(stream, 20))) == 20
    stream.close()


def test_longest_special_waits_for_confirmation():
    tokenizer = small_tokenizer(["<s>", "<s><s>"])
    assert list(tokenizer.encode_iterable(["<s>", "<s>"])) == tokenizer.encode("<s><s>")
    assert list(tokenizer.encode_iterable(["<s>", "<s"])) == tokenizer.encode("<s><s")


@pytest.fixture(scope="module")
def gpt2_tokenizers():
    return (
        get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH, ["<|endoftext|>"]),
        tiktoken.get_encoding("gpt2"),
    )


@pytest.mark.parametrize("filename", [
    "tinystories_sample.txt", "address.txt", "german.txt",
    "special_token_trailing_newlines.txt", "special_token_double_newlines_non_whitespace.txt",
])
def test_file_and_character_stream_match_gpt2(gpt2_tokenizers, filename):
    tokenizer, reference = gpt2_tokenizers
    path = FIXTURES_PATH / filename
    text = path.read_text(encoding="utf-8")
    expected = reference.encode(text, allowed_special={"<|endoftext|>"})
    with path.open(encoding="utf-8") as source:
        assert list(tokenizer.encode_iterable(source)) == expected
    assert list(tokenizer.encode_iterable(iter(text))) == expected


@pytest.mark.parametrize("specials", [[], ["<s>", "<s><s>"]])
def test_working_memory_does_not_grow_with_corpus(specials):
    tokenizer = small_tokenizer(specials)
    fragment = "aaa <s> abc\n"
    expected_per_fragment = len(tokenizer.encode(fragment))

    def measure(repetitions):
        tracemalloc.start()
        try:
            count = sum(1 for _ in tokenizer.encode_iterable(repeat(fragment, repetitions)))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert count == expected_per_fragment * repetitions
        return peak

    small_peak = measure(100)
    large_peak = measure(10_000)
    assert large_peak < small_peak + 100_000
