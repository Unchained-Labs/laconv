"""The state machine, exhaustively, with no audio anywhere.

If barge-in regresses, it regresses here first.
"""

from __future__ import annotations

import pytest

from laconv.turn import Action, BargeIn, Event, IllegalTransition, State, TurnMachine


def run(machine: TurnMachine, *events: Event) -> list[list[Action]]:
    return [machine.feed(event).actions for event in events]


def test_clean_turn_cycle():
    m = TurnMachine(barge_in=BargeIn.FULL)
    assert m.state is State.IDLE

    assert m.feed(Event.SPEECH_STARTED).actions == [Action.START_CAPTURE]
    assert m.state is State.LISTENING

    assert m.feed(Event.SPEECH_ENDED).actions == [Action.COMMIT_UTTERANCE]
    assert m.state is State.THINKING

    assert m.feed(Event.AGENT_AUDIO_STARTED).actions == []
    assert m.state is State.SPEAKING

    assert m.feed(Event.AGENT_AUDIO_FINISHED).actions == []
    assert m.state is State.IDLE
    assert m.turns == 1
    assert m.interruptions == 0


def test_full_barge_in_stops_tts_before_anything_else():
    m = TurnMachine(barge_in=BargeIn.FULL)
    run(m, Event.SPEECH_STARTED, Event.SPEECH_ENDED, Event.AGENT_AUDIO_STARTED)

    actions = m.feed(Event.SPEECH_STARTED).actions
    # Order is the contract: silence first, bookkeeping second.
    assert actions == [Action.STOP_TTS, Action.CANCEL_AGENT, Action.START_CAPTURE]
    assert actions[0] is Action.STOP_TTS
    assert m.state is State.LISTENING
    assert m.interruptions == 1
    assert m.turns == 0  # an interrupted turn is not a completed turn


def test_hold_mode_notices_but_keeps_the_floor():
    m = TurnMachine(barge_in=BargeIn.HOLD)
    run(m, Event.SPEECH_STARTED, Event.SPEECH_ENDED, Event.AGENT_AUDIO_STARTED)

    assert m.feed(Event.SPEECH_STARTED).actions == []
    assert m.state is State.SPEAKING
    assert m.interruptions == 1


def test_half_duplex_gates_the_mic_only_while_speaking():
    m = TurnMachine(barge_in=BargeIn.HALF_DUPLEX)
    assert m.mic_gated is False

    m.feed(Event.SPEECH_STARTED)
    m.feed(Event.SPEECH_ENDED)
    assert m.state is State.THINKING
    # Thinking keeps the mic live: interrupting a slow model is the
    # interruption people actually want.
    assert m.mic_gated is False

    assert m.feed(Event.AGENT_AUDIO_STARTED).actions == [Action.MUTE_MIC]
    assert m.mic_gated is True
    assert m.feed(Event.AGENT_AUDIO_FINISHED).actions == [Action.UNMUTE_MIC]
    assert m.mic_gated is False


def test_speech_during_thinking_cancels_the_pending_reply():
    m = TurnMachine()
    run(m, Event.SPEECH_STARTED, Event.SPEECH_ENDED)

    actions = m.feed(Event.SPEECH_STARTED).actions
    assert actions == [Action.CANCEL_AGENT, Action.START_CAPTURE]
    assert m.state is State.LISTENING
    assert m.interruptions == 1


def test_silent_agent_turn_ends_without_claiming_failure():
    m = TurnMachine()
    run(m, Event.SPEECH_STARTED, Event.SPEECH_ENDED)
    assert m.state is State.THINKING

    # No AGENT_AUDIO_STARTED ever arrives (empty reply). The floor still
    # comes back, and no mic unmute is emitted because none was muted.
    assert m.feed(Event.AGENT_AUDIO_FINISHED).actions == []
    assert m.state is State.IDLE
    assert m.turns == 1


def test_agent_failure_while_speaking_silences_and_reopens():
    m = TurnMachine(barge_in=BargeIn.HALF_DUPLEX)
    run(m, Event.SPEECH_STARTED, Event.SPEECH_ENDED, Event.AGENT_AUDIO_STARTED)

    assert m.feed(Event.AGENT_FAILED).actions == [Action.STOP_TTS, Action.UNMUTE_MIC]
    assert m.state is State.IDLE


def test_agent_failure_while_thinking_just_reopens():
    m = TurnMachine()
    run(m, Event.SPEECH_STARTED, Event.SPEECH_ENDED)
    assert m.feed(Event.AGENT_FAILED).actions == []
    assert m.state is State.IDLE


@pytest.mark.parametrize(
    "state_events, illegal",
    [
        ((), Event.SPEECH_ENDED),  # cannot end speech that never started
        ((), Event.AGENT_AUDIO_STARTED),  # cannot speak without being asked
        ((Event.SPEECH_STARTED,), Event.SPEECH_STARTED),  # already listening
        ((Event.SPEECH_STARTED,), Event.AGENT_AUDIO_STARTED),
        # A late failure must not steal the floor back from a human who is
        # already talking -- LISTENING deliberately has no AGENT_FAILED.
        ((Event.SPEECH_STARTED,), Event.AGENT_FAILED),
    ],
)
def test_illegal_transitions_raise(state_events, illegal):
    m = TurnMachine()
    run(m, *state_events)
    assert m.accepts(illegal) is False
    with pytest.raises(IllegalTransition):
        m.feed(illegal)


def test_close_is_terminal_and_silences_everything():
    m = TurnMachine(barge_in=BargeIn.FULL)
    run(m, Event.SPEECH_STARTED, Event.SPEECH_ENDED, Event.AGENT_AUDIO_STARTED)

    assert m.feed(Event.CLOSE).actions == [Action.STOP_TTS, Action.CANCEL_AGENT]
    assert m.state is State.CLOSED
    for event in Event:
        assert m.accepts(event) is False


def test_close_while_listening_drops_the_buffer():
    m = TurnMachine()
    m.feed(Event.SPEECH_STARTED)
    assert m.feed(Event.CLOSE).actions == [Action.DISCARD_UTTERANCE]


def test_accepts_agrees_with_feed_for_every_state_and_event():
    """The table and the branches must never disagree -- that is the bug class
    where an event is 'legal' but falls through to `raise AssertionError`."""
    reach = {
        State.IDLE: (),
        State.LISTENING: (Event.SPEECH_STARTED,),
        State.THINKING: (Event.SPEECH_STARTED, Event.SPEECH_ENDED),
        State.SPEAKING: (
            Event.SPEECH_STARTED,
            Event.SPEECH_ENDED,
            Event.AGENT_AUDIO_STARTED,
        ),
        State.CLOSED: (Event.CLOSE,),
    }
    for state, path in reach.items():
        for event in Event:
            m = TurnMachine()
            run(m, *path)
            assert m.state is state
            if m.accepts(event):
                m.feed(event)  # must not raise
            else:
                with pytest.raises(IllegalTransition):
                    m.feed(event)
