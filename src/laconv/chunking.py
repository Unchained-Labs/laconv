"""Cutting an LLM token stream into speakable pieces.

This is where LaConv buys most of its perceived latency back. LeClanker streams
tokens; LaVoix synthesises whole strings. Waiting for `done` before calling TTS
means the human hears nothing until the model has finished thinking -- three
seconds of silence on a long answer. Cutting at the first sentence boundary
means audio starts after the first clause and the rest is synthesised while
that clause is playing.

The hard part is not splitting on ".". It is not splitting on the "." in
"Dr. Klein" or "3.14" or "e.g." and thereby speaking two fragments with a
seam in the middle of a word.

Rejected: a sentence tokenizer (nltk/pysbd/spacy). All of them are a model
download or a large dependency to solve a problem that, for *speech*, has a
much weaker requirement than for text: a wrong boundary costs a slightly odd
pause, not a wrong answer. A short abbreviation list plus "digit-dot-digit"
plus a minimum chunk length covers the cases that actually sound broken.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Terminators we will cut on. Newline counts: models emit lists and a list item
# is a speakable unit even without punctuation.
_TERMINATORS = ".!?…\n"
_CLAUSE = ",;:"

# Abbreviations whose trailing dot is not a sentence end. Short on purpose --
# every entry is a case that sounded wrong out loud, not a case that looked
# wrong in a corpus.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "st", "vs", "etc", "eg", "ie", "approx",
    "fig", "no", "vol", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
    "sept", "oct", "nov", "dec", "mon", "tue", "wed", "thu", "fri", "sat",
    "sun", "inc", "ltd", "co", "al", "am", "pm",
}

_TRAILING_WORD = re.compile(r"([A-Za-z]+)\.$")


def _ends_sentence(text: str, rest: str = "") -> bool:
    """True if `text` ends at a place we are willing to cut.

    `rest` is whatever has arrived after the candidate boundary. A streaming
    chunker often has none yet, and "wait for one more token" is always a
    cheaper mistake than "speak half a number".
    """
    if not text:
        return False
    last = text[-1]
    if last not in _TERMINATORS:
        return False
    if last != ".":
        return True

    if len(text) >= 2 and text[-2].isdigit():
        # A dot after a digit is three different things and only the follower
        # tells them apart: "3.14" (decimal), "1. buy milk" (list marker), and
        # "Sentence 0. Next one" (a real sentence that happens to end in a
        # number). Decide by what comes next; if nothing has arrived yet, wait.
        after = rest.lstrip()
        if not rest or not after:
            return False
        if rest[0].isdigit():
            return False  # decimal
        # List markers introduce the item, so the next word is lower-case.
        return after[0].isupper()

    match = _TRAILING_WORD.search(text)
    if match:
        word = match.group(1).lower()
        if word in _ABBREVIATIONS:
            return False
        # "U.S." / "e.g." / "J. R. R. Tolkien" -- a single letter before the
        # dot is an initial, not a one-letter sentence. Costs us a real
        # sentence that ends in a single-letter word, which in English is
        # "I." and essentially nothing else.
        if len(word) == 1:
            return False
    return True


@dataclass
class SentenceChunker:
    """Accumulate token deltas, emit speakable chunks.

    `min_chars` stops the first chunk being "Sure." -- a 5-character TTS call
    has the same round-trip cost as a 200-character one, so tiny chunks make
    the *stream* choppy while making the *first* sound arrive barely sooner.

    `first_chunk_chars` is separately (and much more aggressively) low: the
    very first chunk of a reply is the one the human is waiting on in silence,
    so it is worth an extra round trip to start talking sooner. Eight
    characters is roughly "Sure, ok." -- low enough that a short opener goes
    out immediately, high enough that a stray "1." does not become a chunk.

    `max_chars` is the escape hatch for a model that produces a wall of text
    with no punctuation at all: past that we cut at a clause boundary, then at
    a space, and only then mid-word.
    """

    min_chars: int = 60
    first_chunk_chars: int = 8
    max_chars: int = 300

    _pending: str = ""
    _emitted: int = 0
    _closed: bool = False

    @property
    def pending(self) -> str:
        return self._pending

    def push(self, delta: str) -> list[str]:
        """Add streamed text; return zero or more chunks ready to synthesise."""
        if self._closed:
            raise RuntimeError("chunker already flushed; make a new one per turn")
        self._pending += delta
        out: list[str] = []
        while True:
            chunk = self._try_cut()
            if chunk is None:
                return out
            # `_try_cut` can consume whitespace-only text and hand back "".
            # Progress is still made (pending shrank), so keep looping, but
            # never emit an empty chunk to a TTS engine.
            if chunk:
                out.append(chunk)

    def flush(self) -> list[str]:
        """End of stream: emit whatever is left, however short."""
        self._closed = True
        rest = self._pending.strip()
        self._pending = ""
        if rest:
            self._emitted += 1
            return [rest]
        return []

    # -- internals ----------------------------------------------------------

    def _threshold(self) -> int:
        return self.first_chunk_chars if self._emitted == 0 else self.min_chars

    def _try_cut(self) -> str | None:
        text = self._pending
        if not text.strip():
            return None
        threshold = self._threshold()

        if len(text) >= threshold:
            for i in range(threshold - 1, len(text)):
                if _ends_sentence(text[: i + 1], text[i + 1 :]):
                    return self._emit(i + 1)

        if len(text) >= self.max_chars:
            cut = self._forced_cut(text)
            if cut:
                return self._emit(cut)
        return None

    def _forced_cut(self, text: str) -> int:
        window = text[: self.max_chars]
        for i in range(len(window) - 1, self.min_chars, -1):
            if window[i] in _CLAUSE:
                return i + 1
        space = window.rfind(" ")
        if space > self.min_chars:
            return space + 1
        return self.max_chars  # give up and cut mid-word rather than stall

    def _emit(self, upto: int) -> str:
        chunk = self._pending[:upto].strip()
        self._pending = self._pending[upto:]
        if not chunk:
            return ""
        self._emitted += 1
        return chunk


def chunk_text(text: str, **kwargs: object) -> list[str]:
    """One-shot convenience: the chunks a whole reply would have produced."""
    chunker = SentenceChunker(**kwargs)  # type: ignore[arg-type]
    out = chunker.push(text)
    return [c for c in [*out, *chunker.flush()] if c]


@dataclass
class TranscriptTurn:
    """One exchange, kept so a session can hand history to a stateless agent."""

    role: str  # "user" | "agent"
    text: str
    interrupted: bool = False


@dataclass
class Transcript:
    turns: list[TranscriptTurn] = field(default_factory=list)

    def add(self, role: str, text: str, interrupted: bool = False) -> None:
        self.turns.append(TranscriptTurn(role, text, interrupted))

    def __len__(self) -> int:
        return len(self.turns)
