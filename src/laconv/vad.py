"""Voice activity detection: is this 20 ms of audio speech?

What ships here is an *energy* VAD with an adaptive noise floor and hysteresis.
Be clear about what that is and is not:

  it does     separate speech from room tone in a quiet-to-moderate room, with
              no model, no wheels to build, and no dependency;
  it does not distinguish speech from a slammed door, a dog, a TV, or a second
              person across the room. It is an *activity* detector, not a
              *voice* detector.

Rejected alternatives and why:
  webrtcvad   a C extension, unmaintained upstream, and its own GMM is only
              modestly better than energy on the near-field mic case LaConv
              targets. Not worth becoming a build dependency.
  Silero VAD  genuinely better, and genuinely 100+ MB of torch. So `Vad` below
              is a two-method protocol: bring Silero when your room is loud
              (see README, "Plugging in a better VAD") and LaConv will use it
              without changing a line of the turn machine.

Everything here is a pure function of the frames pushed in -- there is no
wall clock anywhere in this module or in endpoint.py. Elapsed time is derived
from samples consumed, which is what makes the endpointing tests deterministic
instead of flaky-under-CI-load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from laconv.audio import AudioFormat, rms


@runtime_checkable
class Vad(Protocol):
    """The whole contract. Anything with these two methods can replace EnergyVad."""

    def is_speech(self, frame: bytes) -> bool:
        """Classify exactly one frame of `AudioFormat.frame_bytes` bytes."""
        ...

    def reset(self) -> None:
        """Forget adaptive state, e.g. after the device changed or a long gap."""
        ...


@dataclass
class EnergyVad:
    """RMS threshold over an adaptive noise floor, with hysteresis.

    Two knobs matter and both are ratios above the measured floor rather than
    absolute levels, because absolute levels are a property of somebody's USB
    microphone gain and never survive being copied between machines:

      `start_ratio` how far above the floor a frame must be to *open* the gate;
      `stop_ratio`  how far above to *keep* it open. stop < start is the
                    hysteresis that stops a gate chattering on every syllable
                    boundary -- without it you get SPEECH_STARTED/ENDED pairs
                    at 50 Hz and the endpointer above turns them into a dozen
                    half-utterances per sentence.
    """

    fmt: AudioFormat = field(default_factory=AudioFormat)
    start_ratio: float = 3.0
    stop_ratio: float = 1.8
    # Absolute floor under which nothing counts as speech no matter how quiet
    # the room is. In a silent studio the adaptive floor tends to ~0, and any
    # ratio times ~0 is ~0, which would make a mouse fart a sentence.
    min_level: float = 0.005
    # How fast the noise floor tracks the room. Deliberately asymmetric: rise
    # slowly (an air-con kicking in should be learned) but fall fast (once the
    # room quietens we should not stay deaf for ten seconds).
    floor_rise: float = 0.002
    floor_fall: float = 0.05
    calibration_frames: int = 10

    noise_floor: float = 0.0
    _calibrated: int = 0
    _open: bool = False
    last_level: float = 0.0

    def reset(self) -> None:
        self.noise_floor = 0.0
        self._calibrated = 0
        self._open = False
        self.last_level = 0.0

    def is_speech(self, frame: bytes) -> bool:
        if len(frame) != self.fmt.frame_bytes:
            raise ValueError(
                f"expected {self.fmt.frame_bytes}-byte frames, got {len(frame)}"
            )
        level = rms(frame)
        self.last_level = level

        # Calibrate on the opening frames. If the human starts talking in the
        # first 200 ms we learn a too-high floor -- that is why `reset()` is
        # exposed and why the session calls it when a device changes.
        if self._calibrated < self.calibration_frames:
            self._calibrated += 1
            n = self._calibrated
            self.noise_floor += (level - self.noise_floor) / n
            return False

        floor = max(self.noise_floor, self.min_level / self.start_ratio)
        threshold = floor * (self.stop_ratio if self._open else self.start_ratio)
        speech = level >= max(threshold, self.min_level)

        if not speech:
            # Only adapt on non-speech frames: adapting during speech teaches
            # the floor that speech is background, and the VAD goes deaf
            # halfway through a long sentence.
            rate = self.floor_rise if level > self.noise_floor else self.floor_fall
            self.noise_floor += (level - self.noise_floor) * rate

        self._open = speech
        return speech


@dataclass
class ScriptedVad:
    """A VAD that replays a fixed answer sequence. Tests only.

    Provided in the library, not in tests/, so that downstream projects can
    test *their* barge-in handling against LaConv without a microphone either.
    """

    answers: list[bool]
    index: int = 0

    def is_speech(self, frame: bytes) -> bool:  # noqa: ARG002 - frame is ignored by design
        if self.index >= len(self.answers):
            return False
        value = self.answers[self.index]
        self.index += 1
        return value

    def reset(self) -> None:
        self.index = 0
