"""The turn-taking state machine: who holds the floor, and what to do about it.

This module is the heart of LaConv and it is deliberately the most boring code
in the repo: no audio, no sockets, no clock, no I/O. It takes an event, returns
the new state plus a list of *actions* the caller should perform.

Why actions instead of callbacks: a state machine that calls `tts.stop()`
directly can only be tested with a TTS engine attached, which in practice means
it is tested with a microphone in the room, which in practice means it is never
tested. Returning `[Action.STOP_TTS]` lets a unit test assert on the decision
and lets `session.py` own the messy part. Every behaviour claim in the README
about barge-in is a `list ==` assertion in tests/test_turn.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    """Where the conversation is. Exactly one party holds the floor.

    IDLE      nobody is doing anything; the mic is open but we are not
              accumulating an utterance.
    LISTENING the human is mid-utterance; audio is being accumulated.
    THINKING  the human finished, the agent has the turn but has produced no
              audio yet (STT + LLM latency lives here).
    SPEAKING  the agent is emitting audio.
    CLOSED    terminal; the session is over and no event revives it.
    """

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    CLOSED = "closed"


class Event(str, Enum):
    """Things that happen to a conversation.

    These are *decided* facts, not raw signals. `SPEECH_STARTED` does not mean
    "one loud frame arrived"; it means the endpointer is confident speech
    began. Debouncing belongs in endpoint.py so that this machine has no
    tunables and no timing of its own.
    """

    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"  # endpointed: the human finished a sentence
    AGENT_AUDIO_STARTED = "agent_audio_started"
    AGENT_AUDIO_FINISHED = "agent_audio_finished"
    AGENT_FAILED = "agent_failed"
    CLOSE = "close"


class Action(str, Enum):
    """What the shell should do about a transition.

    Ordering within a returned list is significant: STOP_TTS is always emitted
    before anything that starts new work, because the whole point of barge-in
    is that the speaker goes quiet *first*.
    """

    START_CAPTURE = "start_capture"  # begin accumulating utterance audio
    COMMIT_UTTERANCE = "commit_utterance"  # hand the buffer to STT -> agent
    DISCARD_UTTERANCE = "discard_utterance"  # throw the buffer away unheard
    STOP_TTS = "stop_tts"  # kill agent audio *now* (barge-in)
    CANCEL_AGENT = "cancel_agent"  # abandon the in-flight agent response
    MUTE_MIC = "mute_mic"  # half-duplex: stop feeding VAD while we speak
    UNMUTE_MIC = "unmute_mic"


class BargeIn(str, Enum):
    """How the machine reacts to human speech while the agent is speaking.

    FULL       human speech interrupts: TTS stops, the agent response is
               cancelled, we listen. Requires that the microphone does not hear
               the speaker -- i.e. headphones, a directional mic, or acoustic
               echo cancellation. LaConv does NOT ship AEC (see README), so
               choosing FULL on an open-air speaker means the agent will hear
               its own voice and interrupt itself mid-sentence.
    HALF_DUPLEX the mic is gated while the agent speaks. No interruption is
               possible, but the failure mode is "you had to wait", not "the
               bot talks over itself". This is the default precisely because
               there is no AEC to make FULL safe by default.
    HOLD       speech during SPEAKING is detected and ignored (the mic stays
               live so an application can show a "hold on" indicator), but the
               agent finishes its turn. Useful for backchannels ("mhm").
    """

    FULL = "full"
    HALF_DUPLEX = "half_duplex"
    HOLD = "hold"


class IllegalTransition(RuntimeError):
    """An event arrived that the current state has no answer for.

    Raised rather than ignored: in a duplex audio system a silently dropped
    event is a conversation that hangs with both parties waiting, which is
    nearly impossible to debug after the fact. Callers that expect to race
    (e.g. a late AGENT_AUDIO_FINISHED after a barge-in) should use
    `TurnMachine.accepts` instead of catching this.
    """

    def __init__(self, state: State, event: Event) -> None:
        super().__init__(f"cannot handle {event.value} while {state.value}")
        self.state = state
        self.event = event


@dataclass
class Transition:
    """The result of feeding one event to the machine."""

    previous: State
    event: Event
    state: State
    actions: list[Action] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.previous is not self.state


@dataclass
class TurnMachine:
    """Legal turn transitions for one conversation.

    Deterministic and clock-free: feed it the same events in the same order and
    it produces the same actions. That is what makes `pytest` a real test of
    barge-in behaviour rather than a smoke test.
    """

    barge_in: BargeIn = BargeIn.HALF_DUPLEX
    state: State = State.IDLE
    turns: int = 0  # completed agent turns, for logging/metrics
    interruptions: int = 0

    # -- queries ------------------------------------------------------------

    def accepts(self, event: Event) -> bool:
        """True if `feed(event)` would not raise. Cheap look-before-you-leap."""
        return event in _LEGAL.get(self.state, ())

    @property
    def mic_gated(self) -> bool:
        """True when the shell should stop pushing frames into the VAD.

        Only HALF_DUPLEX gates, and only while audio is actually playing --
        THINKING keeps the mic live so a human can barge in on a slow LLM,
        which is the interruption people actually want most.
        """
        return self.barge_in is BargeIn.HALF_DUPLEX and self.state is State.SPEAKING

    # -- the machine --------------------------------------------------------

    def feed(self, event: Event) -> Transition:
        if not self.accepts(event):
            raise IllegalTransition(self.state, event)

        before = self.state
        actions: list[Action] = []

        if event is Event.CLOSE:
            # Closing mid-turn must silence the speaker and drop the buffer,
            # otherwise a hung-up call keeps talking to an empty room.
            if before is State.SPEAKING:
                actions.append(Action.STOP_TTS)
            if before in (State.SPEAKING, State.THINKING):
                actions.append(Action.CANCEL_AGENT)
            if before is State.LISTENING:
                actions.append(Action.DISCARD_UTTERANCE)
            self.state = State.CLOSED
            return Transition(before, event, self.state, actions)

        if event is Event.SPEECH_STARTED:
            if before is State.IDLE:
                actions.append(Action.START_CAPTURE)
                self.state = State.LISTENING
            elif before is State.THINKING:
                # The human started again before the agent said anything. The
                # pending reply answers a question that is being replaced, so
                # drop it rather than speaking it after the new utterance.
                self.interruptions += 1
                actions += [Action.CANCEL_AGENT, Action.START_CAPTURE]
                self.state = State.LISTENING
            elif before is State.SPEAKING:
                if self.barge_in is BargeIn.FULL:
                    self.interruptions += 1
                    # STOP_TTS first: silence beats bookkeeping.
                    actions += [
                        Action.STOP_TTS,
                        Action.CANCEL_AGENT,
                        Action.START_CAPTURE,
                    ]
                    self.state = State.LISTENING
                else:
                    # HOLD (and HALF_DUPLEX, if a frame slipped through before
                    # the gate closed): note it, keep the floor.
                    self.interruptions += 1
            return Transition(before, event, self.state, actions)

        if event is Event.SPEECH_ENDED:
            actions.append(Action.COMMIT_UTTERANCE)
            self.state = State.THINKING
            return Transition(before, event, self.state, actions)

        if event is Event.AGENT_AUDIO_STARTED:
            if self.barge_in is BargeIn.HALF_DUPLEX:
                actions.append(Action.MUTE_MIC)
            self.state = State.SPEAKING
            return Transition(before, event, self.state, actions)

        if event is Event.AGENT_AUDIO_FINISHED:
            self.turns += 1
            # Only unmute if we actually muted -- a silent turn never gated.
            if before is State.SPEAKING and self.barge_in is BargeIn.HALF_DUPLEX:
                actions.append(Action.UNMUTE_MIC)
            self.state = State.IDLE
            return Transition(before, event, self.state, actions)

        if event is Event.AGENT_FAILED:
            # STT died, the LLM 500'd, TTS refused. Either way the agent gives
            # the floor back instead of leaving the human waiting on silence.
            if before is State.SPEAKING:
                actions.append(Action.STOP_TTS)
                if self.barge_in is BargeIn.HALF_DUPLEX:
                    actions.append(Action.UNMUTE_MIC)
            self.state = State.IDLE
            return Transition(before, event, self.state, actions)

        raise AssertionError(f"unhandled event {event}")  # pragma: no cover


# Kept as data rather than as `if` chains so `accepts()` and `feed()` can never
# disagree about what is legal -- they read the same table.
_LEGAL: dict[State, tuple[Event, ...]] = {
    State.IDLE: (Event.SPEECH_STARTED, Event.CLOSE),
    # LISTENING has no AGENT_FAILED: once the human has the floor, a late
    # failure from an already-cancelled generation must not steal it back.
    # session.py drops such reports instead (see `_generation` fencing).
    State.LISTENING: (Event.SPEECH_ENDED, Event.CLOSE),
    # AGENT_AUDIO_FINISHED is legal from THINKING as well as SPEAKING: an
    # agent that replies with nothing sayable (empty string, a tool call that
    # produced no prose) still has to hand the floor back, and calling that a
    # failure would make the caller announce an error that did not happen.
    State.THINKING: (
        Event.SPEECH_STARTED,
        Event.AGENT_AUDIO_STARTED,
        Event.AGENT_AUDIO_FINISHED,
        Event.AGENT_FAILED,
        Event.CLOSE,
    ),
    State.SPEAKING: (
        Event.SPEECH_STARTED,
        Event.AGENT_AUDIO_FINISHED,
        Event.AGENT_FAILED,
        Event.CLOSE,
    ),
    State.CLOSED: (),
}
