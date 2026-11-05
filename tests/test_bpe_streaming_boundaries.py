"""The streaming regex layer must agree with matching complete ordinary text."""

import random
from itertools import product

import pytest

from cs336_basics.bpe import _PRETOKEN_PATTERN
from cs336_basics.bpe_tokenizer import _iter_regex_pretokens


def whole_text_pretokens(text):
    return [match.group() for match in _PRETOKEN_PATTERN.finditer(text)]


@pytest.mark.parametrize("text", [
    "", "Hello transformer world", "we're I'll they've", "'r", "'re", "'ve", "'ll",
    "'sometext 'd 'm 't", "a'!r", " 're", "  a", "\n\t a", "\r\n\n", "a\n\n b",
    "!?🙂!!", "123٤٥Ⅻ", "牛é\u0301𐍈🙂 end", "\u00a0\u2003 x\x1c", "a  ",
])
def test_every_two_piece_split_and_character_fragments(text):
    expected = whole_text_pretokens(text)
    for split in range(len(text) + 1):
        assert list(_iter_regex_pretokens([text[:split], text[split:]])) == expected
    assert list(_iter_regex_pretokens(iter(text))) == expected


@pytest.mark.parametrize("text", ["a're!", "'llx", "'ve?", "a  b", "\n\n x", " 're"])
def test_every_partition(text):
    expected = whole_text_pretokens(text)
    for mask in range(1 << (len(text) - 1)):
        cuts = [0] + [index for index in range(1, len(text)) if mask & (1 << (index - 1))] + [len(text)]
        fragments = [text[start:end] for start, end in zip(cuts, cuts[1:])]
        assert list(_iter_regex_pretokens(fragments)) == expected


def test_randomized_fragmentation():
    rng = random.Random(43)
    alphabet = " abclrvemstd'!?012\t\r\n牛é🙂\u0301\u00a0\u2003\x1c"
    for _ in range(500):
        text = "".join(rng.choices(alphabet, k=rng.randrange(200)))
        fragments = []
        start = 0
        while start < len(text):
            end = start + rng.randrange(1, 15)
            fragments.extend([text[start:end], ""])
            start = end
        expected = whole_text_pretokens(text)
        assert list(_iter_regex_pretokens(fragments)) == expected
        assert list(_iter_regex_pretokens(iter(text))) == expected


def test_emits_before_consuming_entire_input():
    def fragments():
        yield "one two three "
        raise AssertionError("Read ahead before yielding a confirmed pre-token")

    pretokens = _iter_regex_pretokens(fragments())
    assert next(pretokens) == "one"
    pretokens.close()


def test_waits_for_ambiguous_contraction():
    # Emitting the apostrophe from the first fragment would be premature.
    assert list(_iter_regex_pretokens(["a'r", "e!"])) == ["a", "'re", "!"]


def test_long_unfinished_pretoken():
    text = "a" * 10_000
    assert list(_iter_regex_pretokens([text[:5000], text[5000:]])) == [text]


def test_confirmed_special_boundary_ends_ordinary_region():
    # The special-token layer ends each region separately; it must not feed
    # text from both sides into one continuous ordinary-text stream.
    before = list(_iter_regex_pretokens(["hel", "lo "]))
    after = list(_iter_regex_pretokens(["wor", "ld"]))
    assert before == ["hello", " "]
    assert after == ["world"]


def test_exhaustive_short_inputs():
    # Include competing contraction, whitespace, and optional-space branches.
    for length in range(6):
        for characters in product("'lre \ta1!", repeat=length):
            text = "".join(characters)
            assert list(_iter_regex_pretokens(iter(text))) == whole_text_pretokens(text)


def test_random_unicode_matches_regex_categories():
    rng = random.Random(44)
    for _ in range(300):
        text = "".join(chr(rng.randrange(0x110000)) for _ in range(100))
        assert list(_iter_regex_pretokens(iter(text))) == whole_text_pretokens(text)


@pytest.mark.timeout(10)
@pytest.mark.parametrize("character", ["a", "1", "!", " ", "\t"])
def test_long_run_in_single_character_fragments(character):
    # Repeatedly rescanning this growing suffix would require quadratic work.
    text = character * 200_000 + "x"
    assert list(_iter_regex_pretokens(iter(text))) == whole_text_pretokens(text)


def test_yields_as_soon_as_boundary_is_confirmed():
    def fragments():
        yield "hel"
        yield "lo "
        raise AssertionError("The space already confirms the end of 'hello'")

    pretokens = _iter_regex_pretokens(fragments())
    assert next(pretokens) == "hello"
    pretokens.close()
