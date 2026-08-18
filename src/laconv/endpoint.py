"""Endpointing: turning frame-by-frame VAD into "the human finished a sentence".

VAD answers "is there sound now". Endpointing answers "may I reply yet", and
they are not the same question -- the gap between them is every awkward pause
you have had with a voice assistant. This module holds all of the timing so the
turn machine holds none of it.

Three timers, and the trade-off each one buys:

  `min_speech_ms`   sound shorter than this is a cough, a click, a chair. Too
                    low and the agent answers doors; too high and "yes" is
                    never heard.
  `silence_ms`      how long a pause must last before we call end-of-turn. This
                    is *the* latency/interruption dial: 500 ms feels snappy and
                    cuts people off mid-thought; 1200 ms never interrupts and
                    feels like a bad phone line. Default 700 ms.
  `max_utterance_ms` a hard cap. Someone reading a paragraph aloud, or a VAD
                    stuck open on a noisy fan, must not buffer forever and
                    must not be silently dropped -- we endpoint and let the
                    agent answer what it has.

Time is counted in *frames consumed*, never from a clock. A test can push
90 frames of silence and know exactly 1800 ms passed, on any CI runner, at any
load. That is the only reason these tests are worth running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from laconv.audio import AudioFormat
from laconv.vad import EnergyVad, Vad


class Boundary(str, Enum):
    """What the endpointer decided about the frame it was just handed."""

    NONE = "none"
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"  # a real endpoint: reply now
    MAX_LENGTH = "max_length"  # forced endpoint: the cap was hit


@dataclass
class EndpointResult:
    boundary: Boundary
    is_speech: bool
    speech_ms: float = 0.0
    silence_ms: float = 0.0

    def __bool__(self) -> bool:
        return self.boundary is not Boundary.NONE


@dataclass
class Endpointer:
    """Frames in, utterance boundaries out.

    Also owns the *pre-roll*: a short ring of frames from before speech was
    detected. Without it every transcript starts mid-word, because by the time
    energy crosses the threshold the first consonant is already gone. 300 ms of
    pre-roll costs 9.6 kB of RAM and is the single cheapest transcription-
    quality win in the whole pipeline.
    """

    fmt: AudioFormat = field(default_factory=AudioFormat)
    vad: Vad = None  # type: ignore[assignment]  # defaulted in __post_init__
    min_speech_ms: int = 200
    silence_ms: int = 700
    max_utterance_ms: int = 30_000
    preroll_ms: int = 300
    # Consecutive speech frames needed to *declare* SPEECH_START. Distinct from
    # min_speech_ms: this one gates the barge-in signal, so it is the knob that
    # decides whether a cough kills the agent's sentence.
    start_frames: int = 3

    _speech_frames: int = 0
    _silence_frames: int = 0
    _run: int = 0
    _active: bool = False
    _buffer: list[bytes] = field(default_factory=list)
    _preroll: list[bytes] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.vad is None:
            self.vad = EnergyVad(fmt=self.fmt)
        if self.silence_ms < self.fmt.frame_ms:
            raise ValueError("silence_ms must be at least one frame")

    # -- derived timings ----------------------------------------------------

    @property
    def _frames_of_silence_needed(self) -> int:
        return max(1, round(self.silence_ms / self.fmt.frame_ms))

    @property
    def _min_speech_frames(self) -> int:
        return max(1, round(self.min_speech_ms / self.fmt.frame_ms))

    @property
    def _max_frames(self) -> int:
        return max(1, round(self.max_utterance_ms / self.fmt.frame_ms))

    @property
    def _preroll_frames(self) -> int:
        return max(0, round(self.preroll_ms / self.fmt.frame_ms))

    @property
    def active(self) -> bool:
        return self._active

    @property
    def buffered_ms(self) -> float:
        return len(self._buffer) * self.fmt.frame_ms

    # -- the loop -----------------------------------------------------------

    def push(self, frame: bytes) -> EndpointResult:
        """Feed exactly one frame. Returns at most one boundary."""
        speech = self.vad.is_speech(frame)

        if self._active:
            self._buffer.append(frame)
        else:
            self._preroll.append(frame)
            if len(self._preroll) > self._preroll_frames:
                self._preroll.pop(0)

        if speech:
            self._run += 1
            self._silence_frames = 0
            self._speech_frames += 1
        else:
            self._run = 0
            if self._active:
                self._silence_frames += 1

        if not self._active:
            if self._run >= self.start_frames:
                self._active = True
                # Move the pre-roll in front of the frames that triggered us.
                self._buffer = [*self._preroll]
                self._preroll = []
                self._speech_frames = self._run
                self._silence_frames = 0
                return EndpointResult(Boundary.SPEECH_START, True, self._speech_ms())
            return EndpointResult(Boundary.NONE, speech)

        if len(self._buffer) >= self._max_frames:
            return self._close(Boundary.MAX_LENGTH)

        if self._silence_frames >= self._frames_of_silence_needed:
            if self._speech_frames < self._min_speech_frames:
                # A door, not a sentence. Abandon quietly and go back to
                # waiting -- the caller learns nothing happened, which is
                # correct: the turn machine never left IDLE.
                self._reset_utterance()
                return EndpointResult(Boundary.NONE, False)
            return self._close(Boundary.SPEECH_END)

        return EndpointResult(Boundary.NONE, speech, self._speech_ms(), self._silence_ms())

    def take(self) -> bytes:
        """Remove and return the buffered utterance, trailing silence included.

        The trailing silence is left in on purpose: STT models use it as an
        end-of-speech cue and trimming it costs you the last word more often
        than it saves bandwidth.
        """
        pcm = b"".join(self._buffer)
        self._buffer = []
        return pcm

    def discard(self) -> None:
        self._buffer = []

    def reset(self) -> None:
        """Full reset, including VAD calibration. For device changes."""
        self._reset_utterance()
        self._preroll = []
        self.vad.reset()

    # -- internals ----------------------------------------------------------

    def _close(self, boundary: Boundary) -> EndpointResult:
        result = EndpointResult(boundary, False, self._speech_ms(), self._silence_ms())
        self._active = False
        self._run = 0
        self._speech_frames = 0
        self._silence_frames = 0
        return result

    def _reset_utterance(self) -> None:
        self._active = False
        self._run = 0
        self._speech_frames = 0
        self._silence_frames = 0
        self._buffer = []

    def _speech_ms(self) -> float:
        return self._speech_frames * self.fmt.frame_ms

    def _silence_ms(self) -> float:
        return self._silence_frames * self.fmt.frame_ms
