"""Tests for the metadata-translation helper in server.py.

_split_for_translation exists because a video description can run to several
thousand characters, and handing an LLM one giant string invites a truncated
or summarised reply instead of a translation. The properties that matter are
that nothing is dropped and that no chunk exceeds the limit.

The helper is extracted by source rather than imported: importing server.py
pulls in torch, fastapi and the whole pipeline, which is far more than this
pure function needs.
"""
import re
from pathlib import Path

import pytest

_SRC = (Path(__file__).resolve().parent.parent / "server.py").read_text(encoding="utf-8")
_FN = re.search(r"def _split_for_translation.*?\n    return chunks\n", _SRC, re.S)
assert _FN, "could not find _split_for_translation in server.py"
_ns: dict = {}
exec(_FN.group(0), _ns)
split = _ns["_split_for_translation"]


def test_empty_text_yields_no_chunks():
    assert split("") == []
    assert split("   ") == []
    assert split(None) == []


def test_short_text_is_one_chunk():
    assert split("Hello world") == ["Hello world"]


def test_text_at_the_limit_is_not_split():
    assert len(split("x" * 1500, 1500)) == 1


@pytest.mark.parametrize("text", [
    "A" * 900 + "\n\n" + "B" * 900,
    "\n\n".join(f"Para {i} " + "z" * 200 for i in range(20)),
    "\n".join("line " + "q" * 100 for _ in range(40)),
])
def test_no_chunk_exceeds_the_limit(text):
    assert all(len(c) <= 1500 for c in split(text, 1500))


def test_paragraphs_are_kept_whole_when_they_fit():
    # Splitting mid-sentence gives the model less to work with, so paragraph
    # boundaries are preferred over an arbitrary cut.
    text = "A" * 900 + "\n\n" + "B" * 900
    chunks = split(text, 1500)
    assert chunks == ["A" * 900, "B" * 900]


def test_no_words_are_lost():
    text = ("Welcome to the channel!\n\n" + "Some description text. " * 300
            + "\n\nSubscribe: http://example.com")
    chunks = split(text, 1500)
    assert set(text.split()) == set(" ".join(chunks).split())


def test_an_unbroken_monster_line_is_hard_cut_rather_than_dropped():
    # Last resort: a single 5000-char "word" has no boundary to split on, so
    # it is cut. It must still all be there.
    text = "y" * 5200
    chunks = split(text, 1500)
    assert all(len(c) <= 1500 for c in chunks)
    assert "".join(chunks) == text
