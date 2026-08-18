"""The impure shell: audio frames in, a conversation out.

Everything decision-shaped lives in turn.py / endpoint.py / chunking.py, all of
which are synchronous and pure. This module does the parts that can only be
done with a running event loop -- spawning the agent turn as a cancellable
task, racing it against the human's next utterance, and making sure that when
somebody interrupts, the speaker actually goes quiet.

The design constraint that shaped it: `tests/test_session.py` runs a full
conversation, barge-in included, with three fakes and no audio device. If you
add a feature here that cannot be tested that way, it is in the wrong module.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from laconv.audio import AudioFormat
from laconv.chunking import SentenceChunker, Transcript
from laconv.endpoint import Boundary, Endpointer
from laconv.engines import Agent, NullSpeaker, Speaker, Stt, Tts
from laconv.turn import Action, BargeIn, Event, State, TurnMachine

log = logging.getLogger("laconv.session")

Observer = Callable[["SessionEvent"], Any]


@dataclass
class SessionEvent:
    """Something a UI would want to draw. Never something a caller must handle."""

    type: str
    state: State
    text: str = ""
    detail: dict = field(default_factory=dict)


@dataclass
class SessionConfig:
    fmt: AudioFormat = field(default_factory=AudioFormat)
    barge_in: BargeIn = BargeIn.HALF_DUPLEX
    voice: str | None = None
    session_id: str = "default"
    # Time from "the human stopped talking" to "the agent produced its first
    # word". Covers STT plus the model's time-to-first-token. Past this the
    # turn is abandoned: a voice UI that stays silent for 30 s is broken in a
    # way the human cannot distinguish from a crash, so we would rather say so.
    first_response_timeout_s: float = 20.0
    # A whole turn's ceiling, including synthesis and playback of a long reply.
    turn_timeout_s: float = 300.0


class ConversationSession:
    """One live conversation between one microphone and one agent.

    Not thread-safe and not meant to be: drive it from a single event loop, one
    instance per caller/device. Multiplexing belongs in the transport
    (`laconv.server`), which owns one of these per websocket.
    """

    def __init__(
        self,
        stt: Stt,
        agent: Agent,
        tts: Tts,
        speaker: Speaker | None = None,
        config: SessionConfig | None = None,
        endpointer: Endpointer | None = None,
    ) -> None:
        self.config = config or SessionConfig()
        self.stt = stt
        self.agent = agent
        self.tts = tts
        self.speaker = speaker or NullSpeaker()
        self.endpointer = endpointer or Endpointer(fmt=self.config.fmt)
        self.machine = TurnMachine(barge_in=self.config.barge_in)
        self.transcript = Transcript()

        self._observers: list[Observer] = []
        # Strong references to in-flight async observer callbacks. The event
        # loop only holds a *weak* reference to a running task, so a
        # fire-and-forget `ensure_future` can be garbage-collected mid-flight
        # and the event is then silently never delivered. This bit CI on 3.12
        # and 3.13 as a websocket client that sometimes never got `speaking`.
        self._observer_tasks: set[asyncio.Task] = set()
        self._turn_task: asyncio.Task | None = None
        # Monotonic turn id. Every await in a turn is a chance for that turn to
        # have been cancelled and replaced; anything writing back into session
        # state checks its generation first. Without this fence an interrupted
        # turn's late TTS lands on top of the new one.
        self._generation = 0
        self._pending: bytearray = bytearray()  # partial frame carried between pushes
        self._muted = False
        # Set by `inject_text` to make the next committed turn skip STT. A
        # field rather than an argument because the commit is decided inside
        # the state machine, and a typed message must take the *same* path as
        # a spoken one -- a side door around the machine is how you end up
        # with a text turn that cannot be interrupted.
        self._forced_text: str | None = None

    # -- observation --------------------------------------------------------

    def subscribe(self, observer: Observer) -> None:
        self._observers.append(observer)

    def _emit(self, type_: str, text: str = "", detail: dict | None = None) -> None:
        event = SessionEvent(type_, self.machine.state, text, detail or {})
        for observer in self._observers:
            try:
                result = observer(event)
                if asyncio.iscoroutine(result):
                    # Fire and forget: an observer must never be able to stall
                    # the audio path. A UI that falls behind drops frames of
                    # *its own* rendering, not of the microphone. The trade is
                    # that an async observer's events are not ordered against
                    # the audio they describe -- see protocol.py.
                    task = asyncio.ensure_future(result)
                    self._observer_tasks.add(task)
                    task.add_done_callback(self._observer_tasks.discard)
            except Exception:  # noqa: BLE001 - a broken UI must not end a call
                log.exception("observer failed")

    # -- audio in -----------------------------------------------------------

    async def push_audio(self, pcm: bytes) -> None:
        """Feed captured microphone PCM. Any length; framing is handled here.

        Callers get whatever their device hands them (often 512 or 1024 sample
        buffers), which is rarely a whole number of VAD frames. Buffering the
        remainder here means every caller does not reimplement it, badly.
        """
        if self.machine.state is State.CLOSED:
            return
        self._pending += pcm
        n = self.config.fmt.frame_bytes
        while len(self._pending) >= n:
            frame = bytes(self._pending[:n])
            del self._pending[:n]
            await self._push_frame(frame)

    async def _push_frame(self, frame: bytes) -> None:
        if self._muted:
            # Half-duplex: the frame is dropped *before* the VAD sees it, so
            # the noise floor never learns the agent's own voice. Feeding it
            # and ignoring the answer would poison the floor for the next turn.
            return
        result = self.endpointer.push(frame)
        if result.boundary is Boundary.SPEECH_START:
            await self._feed(Event.SPEECH_STARTED)
        elif result.boundary in (Boundary.SPEECH_END, Boundary.MAX_LENGTH):
            if result.boundary is Boundary.MAX_LENGTH:
                self._emit("max_length", detail={"speech_ms": result.speech_ms})
            await self._feed(Event.SPEECH_ENDED)

    # -- machine plumbing ---------------------------------------------------

    async def _feed(self, event: Event) -> None:
        """Run one event through the machine and perform the actions it names."""
        if not self.machine.accepts(event):
            # Expected in normal operation: a turn task can finish at the same
            # moment a barge-in cancels it. Log, do not raise -- the machine's
            # own strictness is for unit tests, not for a live call.
            log.debug("dropping %s in %s", event.value, self.machine.state.value)
            return
        before = self.machine.state
        transition = self.machine.feed(event)
        for action in transition.actions:
            await self._perform(action)
        if transition.changed:
            self._emit("state", detail={"from": before.value, "to": transition.state.value})

    async def _perform(self, action: Action) -> None:
        if action is Action.START_CAPTURE:
            self._emit("listening")
        elif action is Action.STOP_TTS:
            await self.speaker.stop()
            self._emit("interrupted")
        elif action is Action.CANCEL_AGENT:
            await self._cancel_turn()
        elif action is Action.DISCARD_UTTERANCE:
            self.endpointer.discard()
        elif action is Action.MUTE_MIC:
            self._muted = True
        elif action is Action.UNMUTE_MIC:
            self._muted = False
            # The room the VAD calibrated against is the room *before* the
            # speaker was playing; re-arm rather than trust a stale floor.
            self.endpointer.reset()
        elif action is Action.COMMIT_UTTERANCE:
            pcm = self.endpointer.take()
            forced, self._forced_text = self._forced_text, None
            self._generation += 1
            self._turn_task = asyncio.ensure_future(
                self._run_turn(self._generation, pcm, forced)
            )

    async def _cancel_turn(self) -> None:
        task = self._turn_task
        self._turn_task = None
        # Bump the generation *before* awaiting the cancellation, so anything
        # the dying task does on its way out is already fenced off.
        self._generation += 1
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass  # expected: we asked for it
            except Exception:  # noqa: BLE001
                log.exception("turn task died while being cancelled")

    # -- one agent turn -----------------------------------------------------

    async def _run_turn(self, generation: int, pcm: bytes, forced: str | None = None) -> None:
        try:
            await asyncio.wait_for(
                self._turn_body(generation, pcm, forced), self.config.turn_timeout_s
            )
        except asyncio.CancelledError:
            # Re-raise: swallowing this is exactly the bug that keeps an
            # interrupted agent talking.
            raise
        except asyncio.TimeoutError:
            self._emit("error", "turn timed out")
            await self._fail(generation)
        except Exception as exc:  # noqa: BLE001 - one bad turn is not a dead call
            log.exception("turn failed")
            self._emit("error", str(exc))
            await self._fail(generation)

    async def _fail(self, generation: int) -> None:
        if generation == self._generation:
            await self._feed(Event.AGENT_FAILED)

    async def _turn_body(self, generation: int, pcm: bytes, forced: str | None = None) -> None:
        duration_ms = self.config.fmt.ms_for(pcm)
        if forced is None:
            utterance = await asyncio.wait_for(
                self.stt.transcribe(pcm, self.config.fmt.sample_rate),
                self.config.first_response_timeout_s,
            )
            if generation != self._generation:
                return
            text = (utterance.text or "").strip()
        else:
            text = forced.strip()
        self._emit("transcript", text, detail={"duration_ms": duration_ms})
        if not text:
            # STT heard nothing intelligible. Saying "sorry, I did not catch
            # that" is the caller's decision, not the framework's; we just give
            # the floor back so the human can try again immediately.
            await self._end_turn(generation)
            return
        self.transcript.add("user", text)

        chunker = SentenceChunker()
        spoken: list[str] = []
        stream = self.agent.reply(text, session_id=self.config.session_id)
        first = True
        try:
            while True:
                try:
                    delta = await asyncio.wait_for(
                        stream.__anext__(),
                        self.config.first_response_timeout_s if first else None,
                    )
                except StopAsyncIteration:
                    break
                first = False
                if generation != self._generation:
                    return
                self._emit("delta", delta)
                for chunk in chunker.push(delta):
                    await self._speak(generation, chunk, spoken)
            for chunk in chunker.flush():
                await self._speak(generation, chunk, spoken)
        except asyncio.CancelledError:
            # Record what the agent actually got to say before it was cut off.
            # Without this an interrupted answer vanishes from the transcript,
            # and any history the caller replays to the model claims the agent
            # said nothing -- or, worse, said the whole thing.
            if spoken:
                self.transcript.add("agent", " ".join(spoken), interrupted=True)
            raise
        finally:
            # The backend may hold a socket open; closing it on the way out
            # (including on cancellation) is what keeps a barge-in from leaking
            # one connection per interruption.
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()

        if generation != self._generation:
            return
        self.transcript.add("agent", " ".join(spoken))
        await self._end_turn(generation)

    async def _speak(self, generation: int, chunk: str, spoken: list[str]) -> None:
        if generation != self._generation:
            return
        wav = await self.tts.synthesize(chunk, voice=self.config.voice)
        if generation != self._generation:
            return
        spoken.append(chunk)
        if self.machine.state is State.THINKING:
            await self._feed(Event.AGENT_AUDIO_STARTED)
        # Emitted per chunk, not only on the first: a UI that highlights what
        # is being said needs every chunk, and it is also the only event that
        # tells a caller which sentence was cut off by a barge-in.
        self._emit("speaking", chunk)
        await self.speaker.play(wav)

    async def _end_turn(self, generation: int) -> None:
        if generation != self._generation:
            return
        await self._feed(Event.AGENT_AUDIO_FINISHED)

    # -- lifecycle ----------------------------------------------------------

    async def inject_text(self, text: str) -> None:
        """Take a turn from typed text instead of from the microphone.

        Runs through SPEECH_STARTED/SPEECH_ENDED so a mixed text-and-voice
        client shares one state machine: typing while the agent is speaking
        interrupts it exactly like talking over it would.
        """
        text = (text or "").strip()
        if not text or not self.machine.accepts(Event.SPEECH_STARTED):
            return
        self._forced_text = text
        await self._feed(Event.SPEECH_STARTED)
        await self._feed(Event.SPEECH_ENDED)

    async def close(self) -> None:
        await self._feed(Event.CLOSE)
        await self._cancel_turn()
        await self.speaker.stop()
        self._emit("closed")
        # Let any queued async observer callbacks finish before the caller
        # tears the transport down; otherwise the last few events -- including
        # `closed` itself -- are dropped on the floor.
        pending = list(self._observer_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def wait_for_turn(self) -> None:
        """Await the in-flight agent turn. Mostly for tests and CLI examples."""
        task = self._turn_task
        if task:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    @property
    def state(self) -> State:
        return self.machine.state

    @property
    def muted(self) -> bool:
        return self._muted


async def drive(session: ConversationSession, frames: AsyncIterator[bytes]) -> None:
    """Pump an audio source into a session until it runs dry.

    A three-line helper, but it is the three lines everybody gets wrong: they
    forget to close, and the last turn never completes.
    """
    async for pcm in frames:
        await session.push_audio(pcm)
    await session.wait_for_turn()
