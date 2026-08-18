"""PCM helpers, in pure Python because the obvious stdlib answer is gone.

`audioop` (which had a C `rms()`) is deprecated since 3.11 and *removed* in
3.13, so a zero-dependency package cannot use it. `array` + a Python loop is
roughly 20x slower, which sounds fatal and is not: at 16 kHz a 20 ms frame is
320 samples, so a frame costs tens of microseconds against a 20 ms budget. If
you are running many hundreds of concurrent sessions on one process, plug in
numpy yourself -- `Frame.rms` is the only hot spot and it is one function.
"""

from __future__ import annotations

import array
import math
from dataclasses import dataclass

# 16 kHz mono 16-bit is what LaVoix's STT path wants and what every VAD in the
# world is tuned for. LaConv does not resample: it asserts, loudly, rather than
# silently degrading transcription quality with a naive decimation.
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_FRAME_MS = 20
INT16_MAX = 32768.0


class AudioFormatError(ValueError):
    pass


@dataclass(frozen=True)
class AudioFormat:
    """Mono signed-16-bit PCM at a fixed rate, in fixed-size frames."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    frame_ms: int = DEFAULT_FRAME_MS

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise AudioFormatError("sample_rate must be positive")
        if self.frame_ms <= 0:
            raise AudioFormatError("frame_ms must be positive")
        if (self.sample_rate * self.frame_ms) % 1000:
            raise AudioFormatError(
                f"{self.frame_ms} ms is not a whole number of samples at "
                f"{self.sample_rate} Hz"
            )

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_ms // 1000

    @property
    def frame_bytes(self) -> int:
        return self.frame_samples * 2

    def ms_for(self, pcm: bytes) -> float:
        return len(pcm) / 2 / self.sample_rate * 1000.0

    def frames(self, pcm: bytes) -> list[bytes]:
        """Split PCM into whole frames, dropping any trailing partial frame.

        Dropping is right for a live stream: the remainder belongs to the next
        read. Callers that own a whole file should pad first if they care about
        the last few milliseconds.
        """
        if len(pcm) % 2:
            raise AudioFormatError("PCM length is odd; not 16-bit samples")
        n = self.frame_bytes
        return [pcm[i : i + n] for i in range(0, len(pcm) - n + 1, n)]


def rms(pcm: bytes) -> float:
    """Root-mean-square amplitude, normalised to 0.0-1.0.

    Normalised rather than raw int16 so thresholds are portable across
    capture devices and so a config file full of `0.02` reads as "2% of full
    scale" instead of "655, and don't ask".
    """
    if not pcm:
        return 0.0
    if len(pcm) % 2:
        raise AudioFormatError("PCM length is odd; not 16-bit samples")
    samples = array.array("h")
    samples.frombytes(pcm)
    total = 0
    for s in samples:
        total += s * s
    return math.sqrt(total / len(samples)) / INT16_MAX


def dbfs(pcm: bytes) -> float:
    """RMS as dB below full scale. Silence is -inf, not a large negative fudge."""
    level = rms(pcm)
    return -math.inf if level <= 0 else 20 * math.log10(level)


def tone(
    ms: int,
    fmt: AudioFormat | None = None,
    freq: float = 220.0,
    amplitude: float = 0.3,
) -> bytes:
    """Synthetic voiced-ish audio, for tests and the offline example.

    Lives in the library rather than in tests/ on purpose: the runnable example
    has to work on a machine with no microphone, and the whole design claim of
    this project is that you can exercise the conversation loop without one.
    """
    fmt = fmt or AudioFormat()
    n = int(fmt.sample_rate * ms / 1000)
    out = array.array("h")
    for i in range(n):
        out.append(int(amplitude * INT16_MAX * math.sin(2 * math.pi * freq * i / fmt.sample_rate)))
    return out.tobytes()


def silence(ms: int, fmt: AudioFormat | None = None) -> bytes:
    fmt = fmt or AudioFormat()
    return b"\x00\x00" * int(fmt.sample_rate * ms / 1000)


def wav_header(pcm_len: int, fmt: AudioFormat | None = None) -> bytes:
    """A 44-byte RIFF header, so we can hand LaVoix a real .wav.

    `wave` from the stdlib would do this, but only through a file object; this
    is the entire format for the one case we produce (mono, 16-bit, PCM).
    """
    fmt = fmt or AudioFormat()
    import struct

    byte_rate = fmt.sample_rate * 2
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + pcm_len,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        1,  # mono
        fmt.sample_rate,
        byte_rate,
        2,  # block align
        16,  # bits per sample
        b"data",
        pcm_len,
    )


def to_wav(pcm: bytes, fmt: AudioFormat | None = None) -> bytes:
    return wav_header(len(pcm), fmt) + pcm
