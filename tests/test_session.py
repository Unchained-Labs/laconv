"""A whole conversation, barge-in included, with no microphone in the room.

This file is the design constraint made executable. If any of it needs audio
hardware, the architecture has gone wrong.
"""

from __future__ import annotations

import asyncio

import pytest

from laconv.audio import AudioFormat, silence, tone
from laconv.endpoint import Endpointer
from laconv.engines import Utterance
from laconv.session import ConversationSession, SessionConfig
from laconv.turn import BargeIn, State

FMT = AudioFormat(sample_rate=16000, frame_ms=20)


class FakeStt:
    def __init__(self, text: str = "hello there", delay: float = 0.0) -> None:
        self.text = text
        self.delay = delay
        self.calls: list[int] = []

    async def transcribe(self, pcm: bytes, sample_rate: int) -> Utterance:  # noqa: ARG002
        self.calls.append(len(pcm))
        if self.delay:
            await asyncio.sleep(self.delay)
        return Utterance(text=self.text)


class FakeAgent:
    def __init__(self, parts: list[str] | None = None, delay: float = 0.0) -> None:
        self.parts = parts or ["It is sunny. ", "Fifteen degrees and clear."]
        self.delay = delay
        self.prompts: list[str] = []
        self.cancelled = False
        self.finished = False

    async def reply(self, text: str, *, session_id: str):  # noqa: ARG002
        self.prompts.append(text)
        try:
            for part in self.parts:
                if self.delay:
                    await asyncio.sleep(self.delay)
                yield part
            self.finished = True
        finally:
            # An interrupted generator is torn down by one of two routes and a
            # real backend has to survive both: `CancelledError` if the task
            # died while the generator was awaiting, `GeneratorExit` from
            # `aclose()` if it was parked on a yield. Recording it in `finally`
            # covers both without pretending they are the same exception.
            if not self.finished:
                self.cancelled = True


class FakeTts:
    def __init__(self, delay: float = 0.0) -> None:
        self.texts: list[str] = []
        self.delay = delay

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:  # noqa: ARG002
        self.texts.append(text)
        if self.delay:
            await asyncio.sleep(self.delay)
        return b"RIFF" + text.encode()


class RecordingSpeaker:
    def __init__(self, delay: float = 0.0) -> None:
        self.played: list[bytes] = []
        self.stops = 0
        self.delay = delay

    async def play(self, wav: bytes) -> None:
        if self.delay:
            await asyncio.sleep(self.delay)
        self.played.append(wav)

    async def stop(self) -> None:
        self.stops += 1


def build(**kwargs):
    stt = kwargs.pop("stt", FakeStt())
    agent = kwargs.pop("agent", FakeAgent())
    tts = kwargs.pop("tts", FakeTts())
    speaker = kwargs.pop("speaker", RecordingSpeaker())
    endpointer = Endpointer(fmt=FMT, silence_ms=200, min_speech_ms=100, preroll_ms=40,
                            start_frames=2)
    config = SessionConfig(fmt=FMT, **kwargs)
    session = ConversationSession(stt, agent, tts, speaker, config, endpointer)
    return session, stt, agent, tts, speaker


async def speak(session: ConversationSession, ms: int = 400) -> None:
    """Utter `ms` of sound followed by enough silence to endpoint."""
    await session.push_audio(tone(ms, FMT))
    await session.push_audio(silence(400, FMT))


async def calibrate(session: ConversationSession) -> None:
    await session.push_audio(silence(300, FMT))


async def until(predicate, timeout: float = 2.0) -> None:
    """Wait for a condition instead of for a duration.

    Sleeping a fixed 80 ms and asserting the state is how async tests become
    flaky on a loaded CI box. Every wait below is a wait for the thing the
    test is actually about.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition never became true")


# -- the happy path ---------------------------------------------------------


async def test_one_full_turn_end_to_end():
    session, stt, agent, tts, speaker = build()
    events: list[str] = []
    session.subscribe(lambda e: events.append(e.type))

    await calibrate(session)
    await speak(session)
    await session.wait_for_turn()

    assert stt.calls, "STT never saw the utterance"
    assert agent.prompts == ["hello there"]
    assert tts.texts == ["It is sunny.", "Fifteen degrees and clear."]
    assert len(speaker.played) == 2
    assert session.state is State.IDLE
    assert session.machine.turns == 1
    assert "transcript" in events and "speaking" in events

    roles = [(t.role, t.text) for t in session.transcript.turns]
    assert roles == [
        ("user", "hello there"),
        ("agent", "It is sunny. Fifteen degrees and clear."),
    ]


async def test_audio_starts_before_the_agent_has_finished_thinking():
    """The whole latency argument for chunking, asserted.

    Not "TTS was called N times" -- that is a property of the chunk-size
    heuristic. The claim that matters is that the first synthesis happens while
    the model is still generating.
    """
    emitted: list[str] = []

    class NarratingAgent:
        prompts: list[str] = []

        async def reply(self, text: str, *, session_id: str):  # noqa: ARG002
            self.prompts.append(text)
            for part in [
                "The weather is fine today. ",
                "Fifteen degrees and clear. ",
                "Rain is expected after six.",
            ]:
                await asyncio.sleep(0.01)
                emitted.append(part)
                yield part

    class WatchingTts(FakeTts):
        def __init__(self) -> None:
            super().__init__()
            self.parts_emitted_at_call: list[int] = []

        async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
            self.parts_emitted_at_call.append(len(emitted))
            return await super().synthesize(text, voice=voice)

    tts = WatchingTts()
    session, _, _, _, _ = build(agent=NarratingAgent(), tts=tts)
    await calibrate(session)
    await speak(session)
    await session.wait_for_turn()

    assert len(tts.texts) > 1, "the reply was synthesised in one blocking call"
    assert tts.parts_emitted_at_call[0] < 3, "nothing was spoken until the agent finished"


async def test_empty_transcript_gives_the_floor_back_without_calling_the_agent():
    session, _, agent, tts, _ = build(stt=FakeStt(text="   "))
    await calibrate(session)
    await speak(session)
    await session.wait_for_turn()

    assert agent.prompts == []
    assert tts.texts == []
    assert session.state is State.IDLE


# -- barge-in ---------------------------------------------------------------


async def test_full_barge_in_stops_the_speaker_and_cancels_the_agent():
    session, _, agent, _, speaker = build(
        barge_in=BargeIn.FULL,
        agent=FakeAgent(parts=[f"Sentence {i}. " for i in range(20)], delay=0.02),
        speaker=RecordingSpeaker(delay=0.02),
    )
    await calibrate(session)
    await session.push_audio(tone(300, FMT))
    await session.push_audio(silence(400, FMT))

    # Let the agent get properly under way before interrupting.
    await until(lambda: len(speaker.played) >= 1)
    assert session.state is State.SPEAKING
    played_before = len(speaker.played)

    await session.push_audio(tone(200, FMT))  # the human talks over it
    assert session.state is State.LISTENING
    assert speaker.stops >= 1
    assert agent.finished is False
    assert agent.cancelled is True

    await asyncio.sleep(0.1)
    # Nothing new is played after the interruption -- that is barge-in.
    assert len(speaker.played) == played_before


async def test_an_interrupted_reply_is_recorded_as_interrupted():
    """A partial answer must not vanish from the transcript, and must not be
    recorded as if the agent had said the whole thing."""
    session, _, _, _, _ = build(
        barge_in=BargeIn.FULL,
        agent=FakeAgent(parts=[f"Sentence {i}. " for i in range(20)], delay=0.02),
        speaker=RecordingSpeaker(delay=0.02),
    )
    await calibrate(session)
    await speak(session, ms=300)
    await until(lambda: len(session.speaker.played) >= 1)
    assert session.state is State.SPEAKING

    await session.push_audio(tone(300, FMT))

    agent_turns = [t for t in session.transcript.turns if t.role == "agent"]
    assert len(agent_turns) == 1
    assert agent_turns[0].interrupted is True
    assert "Sentence 19." not in agent_turns[0].text


async def test_half_duplex_mutes_the_mic_instead_of_interrupting():
    session, _, agent, _, speaker = build(
        barge_in=BargeIn.HALF_DUPLEX,
        agent=FakeAgent(parts=[f"Sentence {i}. " for i in range(10)], delay=0.02),
        speaker=RecordingSpeaker(delay=0.01),
    )
    await calibrate(session)
    await speak(session, ms=300)
    await until(lambda: session.state is State.SPEAKING)
    assert session.muted is True

    await session.push_audio(tone(400, FMT))  # shouting at it changes nothing
    assert session.state is State.SPEAKING
    assert speaker.stops == 0

    await session.wait_for_turn()
    assert agent.finished is True
    assert session.muted is False


async def test_hold_mode_counts_the_interruption_but_keeps_talking():
    session, _, agent, _, speaker = build(
        barge_in=BargeIn.HOLD,
        agent=FakeAgent(parts=[f"Sentence {i}. " for i in range(10)], delay=0.02),
        speaker=RecordingSpeaker(delay=0.01),
    )
    await calibrate(session)
    await speak(session, ms=300)
    await until(lambda: session.state is State.SPEAKING)

    await session.push_audio(tone(300, FMT))
    assert session.state is State.SPEAKING
    assert speaker.stops == 0
    assert session.machine.interruptions >= 1
    await session.wait_for_turn()
    assert agent.finished is True


async def test_interrupting_a_slow_model_before_it_says_anything():
    """Half-duplex still allows interrupting during THINKING -- the mic is only
    gated while audio is actually playing."""
    session, _, agent, tts, _ = build(
        barge_in=BargeIn.HALF_DUPLEX,
        stt=FakeStt(delay=0.3),
    )
    await calibrate(session)
    await speak(session, ms=300)
    await until(lambda: session.state is State.THINKING)

    await session.push_audio(tone(200, FMT))
    assert session.state is State.LISTENING
    await asyncio.sleep(0.4)
    assert tts.texts == []  # the abandoned turn never spoke
    assert agent.prompts == []


async def test_a_stale_turn_cannot_write_over_the_new_one():
    """The generation fence: an interrupted turn that is mid-STT must not
    resurface and speak after the human has already said something else."""
    session, _, agent, tts, speaker = build(
        barge_in=BargeIn.FULL, stt=FakeStt(text="first", delay=0.25)
    )
    await calibrate(session)
    await speak(session, ms=300)
    await asyncio.sleep(0.02)

    await session.push_audio(tone(300, FMT))  # interrupt during STT
    await session.push_audio(silence(400, FMT))
    await session.wait_for_turn()
    await asyncio.sleep(0.35)

    assert agent.prompts == ["first"]  # exactly one turn reached the agent
    assert len(tts.texts) == 2
    assert speaker.played


# -- failure and lifecycle --------------------------------------------------


async def test_a_failing_agent_gives_the_floor_back():
    class BrokenAgent:
        async def reply(self, text: str, *, session_id: str):  # noqa: ARG002
            raise RuntimeError("model exploded")
            yield ""  # pragma: no cover - unreachable, keeps it a generator

    session, _, _, _, _ = build(agent=BrokenAgent())
    errors: list[str] = []
    session.subscribe(lambda e: errors.append(e.text) if e.type == "error" else None)

    await calibrate(session)
    await speak(session)
    await session.wait_for_turn()
    await asyncio.sleep(0.05)

    assert any("model exploded" in e for e in errors)
    assert session.state is State.IDLE  # ready for the next thing the human says


async def test_a_slow_agent_hits_the_first_response_timeout():
    session, _, _, _, _ = build(stt=FakeStt(delay=5), first_response_timeout_s=0.05)
    errors: list[str] = []
    session.subscribe(lambda e: errors.append(e.text) if e.type == "error" else None)

    await calibrate(session)
    await speak(session)
    await session.wait_for_turn()
    await asyncio.sleep(0.05)
    assert session.state is State.IDLE


async def test_a_broken_observer_cannot_kill_the_call():
    session, _, _, tts, _ = build()

    def explode(event):
        raise RuntimeError("the UI is on fire")

    session.subscribe(explode)
    await calibrate(session)
    await speak(session)
    await session.wait_for_turn()
    assert tts.texts  # the conversation carried on regardless


async def test_close_silences_everything_and_ignores_later_audio():
    session, _, agent, _, speaker = build(
        agent=FakeAgent(parts=[f"Sentence {i}. " for i in range(20)], delay=0.02),
        speaker=RecordingSpeaker(delay=0.02),
    )
    await calibrate(session)
    await speak(session, ms=300)
    await until(lambda: session.state is State.SPEAKING)

    await session.close()
    assert session.state is State.CLOSED
    assert speaker.stops >= 1

    played = len(speaker.played)
    await session.push_audio(tone(400, FMT))
    await asyncio.sleep(0.05)
    assert len(speaker.played) == played


async def test_partial_frames_are_reassembled_across_pushes():
    """Devices hand you 512-sample buffers; 20 ms at 16 kHz is 320 samples."""
    session, stt, _, _, _ = build()
    await calibrate(session)
    pcm = tone(400, FMT) + silence(400, FMT)
    step = 1024  # deliberately not a multiple of frame_bytes (640)
    for i in range(0, len(pcm), step):
        await session.push_audio(pcm[i : i + step])
    await session.wait_for_turn()
    assert stt.calls


# -- typed input ------------------------------------------------------------


async def test_inject_text_takes_the_same_path_as_speech():
    session, stt, agent, tts, _ = build()
    await session.inject_text("what is the time")
    await session.wait_for_turn()

    assert stt.calls == []  # STT skipped, but everything else identical
    assert agent.prompts == ["what is the time"]
    assert tts.texts
    assert session.state is State.IDLE


async def test_typing_while_the_agent_speaks_interrupts_it():
    session, _, agent, _, speaker = build(
        barge_in=BargeIn.FULL,
        agent=FakeAgent(parts=[f"Sentence {i}. " for i in range(20)], delay=0.02),
        speaker=RecordingSpeaker(delay=0.02),
    )
    await calibrate(session)
    await speak(session, ms=300)
    await until(lambda: session.state is State.SPEAKING)

    await session.inject_text("actually never mind")
    await session.wait_for_turn()
    assert agent.cancelled is True
    assert speaker.stops >= 1
    assert agent.prompts[-1] == "actually never mind"


async def test_inject_text_ignores_blanks():
    session, _, agent, _, _ = build()
    await session.inject_text("   ")
    await session.wait_for_turn()
    assert agent.prompts == []


@pytest.mark.parametrize("mode", list(BargeIn))
async def test_every_barge_in_mode_completes_a_plain_turn(mode):
    session, _, _, tts, _ = build(barge_in=mode)
    await calibrate(session)
    await speak(session)
    await session.wait_for_turn()
    assert tts.texts
    assert session.state is State.IDLE
