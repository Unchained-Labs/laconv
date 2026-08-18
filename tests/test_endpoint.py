"""VAD and endpointing against synthetic PCM.

Every timing assertion here is exact, because elapsed time is counted in frames
consumed rather than read off a clock. A test that says "700 ms of silence ends
the turn" is testing the actual rule, not the CI runner's mood.
"""

from __future__ import annotations

import pytest

from laconv.audio import AudioFormat, AudioFormatError, dbfs, rms, silence, to_wav, tone
from laconv.endpoint import Boundary, Endpointer
from laconv.vad import EnergyVad, ScriptedVad


@pytest.fixture
def fmt() -> AudioFormat:
    return AudioFormat(sample_rate=16000, frame_ms=20)


def push_all(ep: Endpointer, pcm: bytes) -> list[Boundary]:
    return [ep.push(frame).boundary for frame in ep.fmt.frames(pcm)]


# -- audio helpers ---------------------------------------------------------


def test_rms_of_silence_is_zero_and_of_a_tone_is_not(fmt):
    assert rms(silence(100, fmt)) == 0.0
    assert dbfs(silence(100, fmt)) == float("-inf")
    level = rms(tone(100, fmt, amplitude=0.5))
    assert 0.3 < level < 0.4  # RMS of a sine is amplitude / sqrt(2)


def test_frame_maths_rejects_impossible_framing():
    with pytest.raises(AudioFormatError):
        AudioFormat(sample_rate=44100, frame_ms=1)  # 44.1 samples, not a frame


def test_frames_drops_the_trailing_partial_frame(fmt):
    pcm = silence(50, fmt)  # 2.5 frames at 20 ms
    frames = fmt.frames(pcm)
    assert len(frames) == 2
    assert all(len(f) == fmt.frame_bytes for f in frames)


def test_wav_header_is_a_riff_header_of_the_right_length(fmt):
    wav = to_wav(silence(100, fmt), fmt)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert len(wav) == 44 + 100 * 16000 // 1000 * 2


# -- VAD -------------------------------------------------------------------


def test_energy_vad_calibrates_then_detects(fmt):
    vad = EnergyVad(fmt=fmt)
    # Calibration frames always answer False, whatever they contain.
    for frame in fmt.frames(silence(200, fmt)):
        assert vad.is_speech(frame) is False
    assert vad.noise_floor == pytest.approx(0.0, abs=1e-6)

    speech = [vad.is_speech(f) for f in fmt.frames(tone(200, fmt))]
    assert all(speech)

    quiet = [vad.is_speech(f) for f in fmt.frames(silence(200, fmt))]
    assert not any(quiet)


def test_energy_vad_ignores_room_tone_below_the_absolute_floor(fmt):
    """A very quiet room drives the adaptive floor to ~0; without `min_level`
    any ratio above ~0 is still ~0 and a mouse becomes a sentence."""
    vad = EnergyVad(fmt=fmt)
    for frame in fmt.frames(silence(300, fmt)):
        vad.is_speech(frame)
    hiss = tone(200, fmt, amplitude=0.001)  # -60 dBFS
    assert not any(vad.is_speech(f) for f in fmt.frames(hiss))


def test_energy_vad_does_not_go_deaf_during_a_long_sentence(fmt):
    """The floor must not adapt upward on speech frames -- that bug makes the
    VAD lose the second half of every long utterance."""
    vad = EnergyVad(fmt=fmt)
    for frame in fmt.frames(silence(300, fmt)):
        vad.is_speech(frame)
    long_speech = fmt.frames(tone(5_000, fmt))
    assert all(vad.is_speech(f) for f in long_speech)


def test_reset_forgets_calibration(fmt):
    vad = EnergyVad(fmt=fmt)
    for frame in fmt.frames(silence(300, fmt)):
        vad.is_speech(frame)
    vad.reset()
    assert vad.noise_floor == 0.0
    assert vad.is_speech(fmt.frames(tone(20, fmt))[0]) is False  # recalibrating


def test_vad_rejects_wrong_sized_frames(fmt):
    with pytest.raises(ValueError, match="frames"):
        EnergyVad(fmt=fmt).is_speech(b"\x00" * 10)


# -- endpointing -----------------------------------------------------------


def test_endpoints_after_exactly_the_configured_silence(fmt):
    ep = Endpointer(fmt=fmt, silence_ms=700, min_speech_ms=100, preroll_ms=0)
    push_all(ep, silence(300, fmt))  # calibration
    boundaries = push_all(ep, tone(500, fmt))
    assert Boundary.SPEECH_START in boundaries

    # 34 frames = 680 ms: not yet.
    for _ in range(34):
        assert ep.push(silence(20, fmt)).boundary is Boundary.NONE
    # frame 35 crosses 700 ms.
    assert ep.push(silence(20, fmt)).boundary is Boundary.SPEECH_END


def test_a_click_is_not_a_sentence(fmt):
    """40 ms of noise then silence must produce no boundary at all -- the
    machine never leaves IDLE, which is the whole point of min_speech_ms."""
    ep = Endpointer(fmt=fmt, silence_ms=200, min_speech_ms=300, start_frames=1)
    push_all(ep, silence(300, fmt))
    boundaries = push_all(ep, tone(40, fmt) + silence(600, fmt))
    assert Boundary.SPEECH_END not in boundaries
    assert ep.active is False


def test_start_frames_debounces_a_single_loud_frame(fmt):
    """This is the barge-in guard: one loud frame must not kill the agent."""
    ep = Endpointer(fmt=fmt, start_frames=3, preroll_ms=0)
    push_all(ep, silence(300, fmt))
    assert ep.push(fmt.frames(tone(20, fmt))[0]).boundary is Boundary.NONE
    assert ep.push(fmt.frames(silence(20, fmt))[0]).boundary is Boundary.NONE
    assert ep.active is False

    boundaries = push_all(ep, tone(100, fmt))
    assert boundaries.count(Boundary.SPEECH_START) == 1


def test_max_length_forces_an_endpoint(fmt):
    ep = Endpointer(fmt=fmt, max_utterance_ms=400, preroll_ms=0, min_speech_ms=40)
    push_all(ep, silence(300, fmt))
    boundaries = push_all(ep, tone(2_000, fmt))
    # Reusable, not stuck: continuous speech past the cap keeps producing
    # endpoints rather than buffering forever or going silent.
    assert boundaries.count(Boundary.MAX_LENGTH) >= 2


def test_preroll_keeps_the_first_consonant(fmt):
    """Without pre-roll every transcript starts mid-word, because by the time
    energy crosses the threshold the attack is already gone."""
    ep = Endpointer(fmt=fmt, preroll_ms=100, silence_ms=200, min_speech_ms=40,
                    start_frames=2)
    push_all(ep, silence(300, fmt))
    push_all(ep, tone(200, fmt) + silence(300, fmt))
    captured = ep.take()
    # 100 ms of pre-roll + 200 ms speech + 200 ms of the trailing silence that
    # ended the turn, give or take the frame the VAD opened on.
    assert ep.fmt.ms_for(captured) >= 400


def test_take_and_discard_are_independent(fmt):
    ep = Endpointer(fmt=fmt, silence_ms=200, min_speech_ms=40, preroll_ms=0)
    push_all(ep, silence(300, fmt))
    push_all(ep, tone(200, fmt) + silence(300, fmt))
    assert ep.take() != b""
    assert ep.take() == b""  # take is destructive

    push_all(ep, tone(200, fmt) + silence(300, fmt))
    ep.discard()
    assert ep.take() == b""


def test_endpointer_accepts_any_vad(fmt):
    """The Vad protocol is the seam for Silero et al. -- prove it is one."""
    # 3 speech frames (start_frames=3), then 10 silent frames (200 ms).
    scripted = ScriptedVad([True] * 3 + [False] * 20)
    ep = Endpointer(fmt=fmt, vad=scripted, silence_ms=200, min_speech_ms=40,
                    preroll_ms=0, start_frames=3)
    boundaries = push_all(ep, tone(20, fmt) * 23)
    assert boundaries.count(Boundary.SPEECH_START) == 1
    assert boundaries.count(Boundary.SPEECH_END) == 1


def test_silence_ms_below_one_frame_is_rejected(fmt):
    with pytest.raises(ValueError, match="at least one frame"):
        Endpointer(fmt=fmt, silence_ms=5)
