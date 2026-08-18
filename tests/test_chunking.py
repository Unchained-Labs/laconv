"""Sentence chunking: the difference between 3 s of silence and 300 ms of it.

Each case here is a thing that sounded wrong out loud, not a thing that looked
wrong in a corpus.
"""

from __future__ import annotations

from laconv.chunking import SentenceChunker, Transcript, chunk_text


def stream(text: str, size: int = 4) -> list[str]:
    """Emulate an LLM's token deltas -- boundaries land mid-word, as they do."""
    return [text[i : i + size] for i in range(0, len(text), size)]


def test_first_chunk_is_emitted_early():
    """The first chunk is what the human waits for in silence, so it gets a
    much lower threshold than the ones after it."""
    chunker = SentenceChunker(first_chunk_chars=8, min_chars=60)
    out: list[str] = []
    for delta in stream("It is sunny today. Fifteen degrees and clear all afternoon."):
        out += chunker.push(delta)
    assert out[0] == "It is sunny today."


# Thresholds are forced to 1 in the cases below so that each one tests the
# boundary *rule* and not the chunk-size heuristic layered on top of it.
EAGER = {"first_chunk_chars": 1, "min_chars": 1}


def test_abbreviations_do_not_split_mid_name():
    assert chunk_text("Dr. Klein is here. Send him in.", **EAGER) == [
        "Dr. Klein is here.",
        "Send him in.",
    ]


def test_decimals_and_numbered_lists_do_not_split():
    assert chunk_text("Pi is 3.14 roughly. Done.", **EAGER) == [
        "Pi is 3.14 roughly.",
        "Done.",
    ]
    # "1." opens a list item; cutting there would speak a bare "1."
    chunks = chunk_text("Steps: 1. buy milk 2. go home. That is all.", **EAGER)
    assert "1." not in chunks
    assert "2." not in chunks


def test_initialisms_survive():
    assert chunk_text("She works at the U.S. mint. Really.", **EAGER) == [
        "She works at the U.S. mint.",
        "Really.",
    ]


def test_question_and_exclamation_are_boundaries():
    assert chunk_text("Are you there? Good! Let us start.", **EAGER) == [
        "Are you there?",
        "Good!",
        "Let us start.",
    ]


def test_newline_is_a_boundary_because_list_items_are_speakable():
    chunks = chunk_text("Here is the list\nmilk\nbread\n", **EAGER)
    assert chunks == ["Here is the list", "milk", "bread"]


def test_a_wall_of_text_with_no_punctuation_still_gets_cut():
    """Otherwise a model that forgets punctuation produces one 4000-character
    TTS call and the human hears nothing for ten seconds."""
    text = "word " * 200
    chunks = chunk_text(text, max_chars=120, min_chars=40)
    assert len(chunks) > 1
    assert all(len(c) <= 130 for c in chunks)
    # Cut on a word boundary, not mid-word.
    assert all(not c.endswith("wor") for c in chunks)


def test_flush_emits_the_tail_however_short():
    chunker = SentenceChunker()
    assert chunker.push("Fine") == []
    assert chunker.flush() == ["Fine"]


def test_nothing_is_lost_or_duplicated():
    text = "First sentence here. Second one, slightly longer. And a third! Done?"
    chunks = chunk_text(text, **EAGER)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_empty_and_whitespace_streams_produce_no_chunks():
    chunker = SentenceChunker()
    assert chunker.push("   ") == []
    assert chunker.push("\n\n") == []
    assert chunker.flush() == []


def test_transcript_records_turns():
    transcript = Transcript()
    transcript.add("user", "hello")
    transcript.add("agent", "hi there", interrupted=True)
    assert len(transcript) == 2
    assert transcript.turns[1].interrupted is True
