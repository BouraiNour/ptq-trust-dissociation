import pytest
from ptq_trust.parsing import parse_answer, split_sentences, margin, format_question, build_messages


@pytest.mark.parametrize("text,letter,method", [
    ("Step one. Step two. Answer: B", "B", "primary"),
    ("Reasoning here. **Answer: D**", "D", "primary"),
    ("Reasoning here. Answer:** A", "A", "primary"),
    ("Reasoning here. Answer: (C)", "C", "primary"),
    ("Hmm. Answer: A. Actually wait. Answer: C", "C", "primary"),      # last match wins
])
def test_primary_parse(text, letter, method):
    l, loose, m, _ = parse_answer(text)
    assert (l, loose, m) == (letter, letter, method)


@pytest.mark.parametrize("text", [
    "The answer depends on the vignette.",        # v1 regex returned 'D' here
    "The correct answer choice is C because X.",  # v1 regex returned 'C' here
    "This answer cannot be A",
    "the answers are mixed",
    "Truncated reasoning that never reaches an ans",
])
def test_no_false_positives(text):
    assert parse_answer(text)[0] is None
    assert parse_answer(text)[2] in ("none", "loose")


def test_loose_fallback_flagged_not_primary():
    l, loose, m, rat = parse_answer("Point one is here. Therefore, the answer is B.")
    assert l is None and loose == "B" and m == "loose"
    assert rat == "Point one is here."                  # sentence containing the match is removed


def test_rationale_cut_at_answer():
    assert parse_answer("First point. Second point. Answer: B")[3] == "First point. Second point."


def test_split_merges_short_fragments_and_never_drops_text():
    got = split_sentences("Hi. Yes. This is a long enough sentence. No. Another long enough sentence here.")
    assert got == ["Hi. Yes. This is a long enough sentence. No.", "Another long enough sentence here."]
    assert split_sentences("") == []


def test_split_keeps_all_words():
    t = "Ok. Then a long one follows here. Fine."
    assert " ".join(split_sentences(t)).split() == t.split()


def test_margin():
    assert margin([1, 5, 3, 2], 1) == 2
    assert margin([1, 5, 3, 2], 0) == -4


def test_prompt_format():
    ex = {"question": "Q?", "options": [{"key": "A", "value": "x"}, {"key": "B", "value": "y"}]}
    block = format_question(ex)
    assert block == "Q?\n\nOptions:\nA. x\nB. y"
    msg = build_messages(block)
    assert msg[0]["role"] == "user" and msg[0]["content"].endswith("one of A, B, C, D.")
