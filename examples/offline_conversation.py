#!/usr/bin/env python3
"""A full conversation, with a barge-in, on a machine with no microphone.

    python examples/offline_conversation.py

This is the runnable proof of the design claim: every timing decision LaConv
makes -- when speech started, when the sentence ended, when to stop the
speaker -- is exercised here with synthetic PCM and three fake engines. Swap
`FakeStt`/`FakeAgent`/`FakeTts` for `LavoixStt`/`LeClankerAgent`/`LavoixTts`
and the same script drives the real thing.
"""

from __future__ import annotations

import asyncio

from laconv import (
    AudioFormat,
    ConversationSession,
    SessionConfig,
    SessionEvent,
    Utterance,
    silence,
    tone,
)
from laconv.turn import BargeIn

FMT = AudioFormat()  # 16 kHz mono, 20 ms frames


class FakeStt:
    """Answers with a fixed sentence, whatever the audio was."""

    def __init__(self) -> None:
        self.said = ["what is the weather like", "actually, tell me about tomorrow"]

    async def transcribe(self, pcm: bytes, sample_rate: int) -> Utterance:  # noqa: ARG002
        await asyncio.sleep(0.15)  # a plausible STT round trip
        return Utterance(text=self.said.pop(0) if self.said else "thanks")


class FakeAgent:
    """A LeClanker stand-in: streams text deltas, slowly, and is cancellable."""

    async def reply(self, text: str, *, session_id: str):  # noqa: ARG002
        for part in [
            "It is sunny in Paris right now. ",
            "Fifteen degrees, with a light wind from the west. ",
            "Rain is expected after six in the evening. ",
            "Tomorrow looks similar but a little cooler.",
        ]:
            await asyncio.sleep(0.3)  # time to first token, then per sentence
            yield part


class FakeTts:
    """Returns a beep whose length is proportional to the text."""

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:  # noqa: ARG002
        await asyncio.sleep(0.1)
        return tone(len(text) * 10, FMT)


def show(event: SessionEvent) -> None:
    detail = " ".join(f"{k}={v}" for k, v in event.detail.items())
    body = f" {event.text!r}" if event.text else ""
    print(f"  [{event.state.value:9}] {event.type}{body} {detail}".rstrip())


async def main() -> None:
    session = ConversationSession(
        FakeStt(),
        FakeAgent(),
        FakeTts(),
        # `full` because in this simulation the "speaker" is not audible to the
        # "microphone" -- exactly the condition full barge-in requires. On real
        # open-air hardware without AEC you want the half_duplex default.
        config=SessionConfig(fmt=FMT, barge_in=BargeIn.FULL, session_id="demo"),
    )
    session.subscribe(show)

    print("1. the human asks a question")
    await session.push_audio(silence(400, FMT))  # let the VAD learn the room
    await session.push_audio(tone(900, FMT))  # ~a second of speech
    await session.push_audio(silence(900, FMT))  # the pause that ends the turn

    print("\n2. the agent starts answering, and the human talks over it")
    await asyncio.sleep(0.8)
    print(f"   (state before the interruption: {session.state.value})")
    await session.push_audio(tone(500, FMT))
    await session.push_audio(silence(900, FMT))
    await session.wait_for_turn()
    await session.close()

    print(
        f"\ncompleted turns: {session.machine.turns}   "
        f"interruptions: {session.machine.interruptions}"
    )
    for turn in session.transcript.turns:
        mark = " (cut off)" if turn.interrupted else ""
        print(f"  {turn.role:5}: {turn.text}{mark}")


if __name__ == "__main__":
    asyncio.run(main())
