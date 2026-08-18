"""The three things LaConv needs from the outside world, and nothing more.

`Stt`, `Agent`, `Tts`. Backends live in `laconv.backends` (LaVoix for the first
and third, LeClanker for the second), but the session only ever sees these
protocols -- which is why `tests/test_session.py` can run a whole conversation,
barge-in included, with three fakes and no network.

They are async because a voice loop that blocks the event loop for a 400 ms
HTTP round trip stops feeding the VAD, and a VAD that misses frames misses the
interruption you are trying to detect. Backends that are only available
synchronously (`urllib`) wrap themselves in `asyncio.to_thread`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Utterance:
    """What the human said, as heard."""

    text: str
    language: str | None = None
    # Kept because "the STT returned empty string" and "the STT was not called"
    # are different bugs, and only one of them is the user's fault.
    duration_ms: float = 0.0
    raw: dict = field(default_factory=dict)


@runtime_checkable
class Stt(Protocol):
    async def transcribe(self, pcm: bytes, sample_rate: int) -> Utterance:
        """Mono 16-bit PCM in, text out.

        Utterance-granular, not streaming: see README, "Streaming, honestly".
        """
        ...


@runtime_checkable
class Agent(Protocol):
    def reply(self, text: str, *, session_id: str) -> AsyncIterator[str]:
        """Yield reply text deltas.

        Declared `def`, not `async def`, on purpose: implementations are async
        *generator* functions, and calling one returns the iterator directly
        rather than a coroutine that must be awaited first. The iterator
        must be cancellable at any yield point.

        Cancellation is the contract that makes barge-in work: LaConv cancels
        the asyncio task running this iterator, so a backend that swallows
        `CancelledError` in a `try/except Exception` will keep an interrupted
        agent talking. `backends/leclanker.py` has a comment about exactly this.
        """
        ...


@runtime_checkable
class Tts(Protocol):
    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        """Text in, WAV bytes out."""
        ...


@runtime_checkable
class Speaker(Protocol):
    """Where synthesised audio goes: a sound card, a websocket, /dev/null.

    Separate from `Tts` because barge-in has to stop *playback*, and stopping
    playback is a property of the sink, not of the synthesiser. A `Tts` that
    has already returned bytes cannot un-say them.
    """

    async def play(self, wav: bytes) -> None:
        """Play to completion, or raise CancelledError if interrupted."""
        ...

    async def stop(self) -> None:
        """Drop anything queued or playing, right now. Must be idempotent."""
        ...


class NullSpeaker:
    """Discards audio. The default, so a session is runnable in a test."""

    def __init__(self) -> None:
        self.played: list[bytes] = []
        self.stops = 0

    async def play(self, wav: bytes) -> None:
        self.played.append(wav)

    async def stop(self) -> None:
        self.stops += 1
