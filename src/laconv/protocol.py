"""The LaConv wire protocol: what a remote microphone and the server say.

Transport-agnostic on purpose. Everything here is `bytes`/`dict` in, `bytes`/
`dict` out, so the whole protocol is unit-tested without opening a socket and
`server.py` stays a thin adapter over `websockets`.

Framing
-------
Binary frames are raw audio, never JSON. Text frames are JSON control messages,
never audio. There is no length prefix, no envelope, no base64 -- a websocket
already frames things, and base64'ing PCM to fit it inside JSON would cost 33%
bandwidth for nothing.

    client -> server   binary: mono s16le PCM at the rate agreed in `hello`
    client -> server   text:   {"type": "hello" | "bye" | "text"}
    server -> client   binary: WAV (RIFF header included) to play
    server -> client   text:   {"type": "ready" | "state" | "transcript" |
                                "delta" | "speaking" | "stop_audio" |
                                "error" | "closed"}

Ordering
--------
Binary audio frames are sent inline with the turn, so `stop_audio` is always
ordered *before* any audio that follows it -- that ordering is what makes
barge-in correct. Descriptive text frames (`transcript`, `delta`, `speaking`,
`state`) go through the session's observer path, which is deliberately
fire-and-forget so a slow client cannot stall the audio pipeline. A client must
therefore not assume a `speaking` frame arrives before the WAV it describes.

The one message that matters most
--------------------------------
`stop_audio`. A remote device buffers the WAVs it has been sent; when the human
interrupts, the *server* knows first and the device is still playing. Barge-in
is not implemented until the device drops its buffer, so `stop_audio` is sent
before anything else on an interruption and a client that ignores it does not
have barge-in no matter what the server does.

Why WebSocket and not WebRTC (a v1 non-goal, stated plainly)
------------------------------------------------------------
WebRTC is the right answer for lossy networks: Opus, jitter buffering, packet
loss concealment, and AEC that browsers already implement. LaConv v1 does not
use it because doing it properly means `aiortc` (a large native dependency),
ICE, and a TURN server to survive NAT -- an operational burden that buys
nothing on the network this actually runs on, which is a LAN or a Tailscale
mesh where TCP loss is rare and RTT is single-digit milliseconds. On such a
link, PCM over WebSocket adds a few tens of milliseconds versus Opus over
SRTP, and costs zero infrastructure.

The honest cost of that choice: over the open internet, or on flaky Wi-Fi,
TCP head-of-line blocking will make audio stutter and this protocol has no
concealment. If you need that, WebRTC is the answer and this is not it. The
`Speaker`/`push_audio` seams are where a WebRTC transport would attach.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from laconv.audio import DEFAULT_FRAME_MS, DEFAULT_SAMPLE_RATE, AudioFormat
from laconv.turn import BargeIn

PROTOCOL_VERSION = 1


class ProtocolError(ValueError):
    pass


@dataclass
class Hello:
    """The client's opening message; the only place format is negotiated."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    frame_ms: int = DEFAULT_FRAME_MS
    session_id: str = "default"
    barge_in: BargeIn = BargeIn.HALF_DUPLEX
    version: int = PROTOCOL_VERSION

    @property
    def fmt(self) -> AudioFormat:
        return AudioFormat(sample_rate=self.sample_rate, frame_ms=self.frame_ms)


def parse_hello(message: str | bytes | dict) -> Hello:
    """Validate a client hello. Rejects rather than coerces.

    Silently resampling a client that opened at 44.1 kHz would produce a
    session that mostly works and transcribes badly, which is the worst
    possible failure: it looks like the STT model is bad.
    """
    payload = decode_control(message)
    if payload.get("type") != "hello":
        raise ProtocolError(f"expected hello, got {payload.get('type')!r}")

    version = int(payload.get("version", PROTOCOL_VERSION))
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version {version}")

    barge_in_raw = payload.get("barge_in", BargeIn.HALF_DUPLEX.value)
    try:
        barge_in = BargeIn(barge_in_raw)
    except ValueError as exc:
        raise ProtocolError(f"unknown barge_in mode {barge_in_raw!r}") from exc

    hello = Hello(
        sample_rate=int(payload.get("sample_rate", DEFAULT_SAMPLE_RATE)),
        frame_ms=int(payload.get("frame_ms", DEFAULT_FRAME_MS)),
        session_id=str(payload.get("session_id", "default")),
        barge_in=barge_in,
        version=version,
    )
    # Force the framing maths now, so an impossible rate/frame combination is
    # a refused connection rather than a session that fails on its first frame.
    _ = hello.fmt
    return hello


def decode_control(message: str | bytes | dict) -> dict:
    if isinstance(message, dict):
        payload = message
    else:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"control frame is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("control frame must be a JSON object")
    if "type" not in payload:
        raise ProtocolError("control frame has no 'type'")
    return payload


def encode(type_: str, **fields: object) -> str:
    return json.dumps({"type": type_, **fields}, separators=(",", ":"))


def ready(fmt: AudioFormat, session_id: str, barge_in: BargeIn) -> str:
    return encode(
        "ready",
        version=PROTOCOL_VERSION,
        sample_rate=fmt.sample_rate,
        frame_ms=fmt.frame_ms,
        session_id=session_id,
        barge_in=barge_in.value,
        # Advertised so a client never has to guess what this build can do.
        # Absent entries are absent capabilities, not defaults -- see README.
        capabilities=["vad", "endpointing", "barge_in", "streaming_tts_chunks"],
    )


SendBytes = Callable[[bytes], Awaitable[None]]
SendText = Callable[[str], Awaitable[None]]


class WireSpeaker:
    """A `Speaker` that plays audio by sending it down the wire.

    Holds the barge-in contract for remote devices: `stop()` sends `stop_audio`
    and marks a generation boundary, so any WAV that was already being awaited
    when the interruption landed is dropped instead of being sent late.
    """

    def __init__(self, send_bytes: SendBytes, send_text: SendText) -> None:
        self._send_bytes = send_bytes
        self._send_text = send_text
        self._generation = 0
        self.stops = 0

    async def play(self, wav: bytes) -> None:
        generation = self._generation
        await self._send_bytes(wav)
        if generation != self._generation:
            # We were interrupted mid-send. Tell the client again: the first
            # stop_audio may have raced ahead of the audio it was meant to
            # cancel, and a device that has already queued this WAV would
            # otherwise play the interrupted sentence anyway.
            await self._send_text(encode("stop_audio"))

    async def stop(self) -> None:
        self._generation += 1
        self.stops += 1
        await self._send_text(encode("stop_audio"))


class LocalSpeaker:
    """Plays through the default output device via `sounddevice`.

    Optional extra (`pip install "laconv[device]"`). Kept deliberately dumb:
    one chunk at a time, stop() cancels the in-flight playback. Anything
    smarter (crossfades, a mixing queue) belongs in an application.
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        try:
            import sounddevice  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on host
            raise RuntimeError(
                "LocalSpeaker needs the 'device' extra: pip install 'laconv[device]'"
            ) from exc
        self.sample_rate = sample_rate
        self._task: asyncio.Task | None = None

    async def play(self, wav: bytes) -> None:  # pragma: no cover - needs hardware
        import sounddevice as sd

        pcm = wav[44:] if wav[:4] == b"RIFF" else wav
        self._task = asyncio.ensure_future(asyncio.to_thread(self._blocking_play, sd, pcm))
        await self._task

    def _blocking_play(self, sd, pcm: bytes) -> None:  # pragma: no cover
        import array

        samples = array.array("h")
        samples.frombytes(pcm)
        sd.play(samples, self.sample_rate, blocking=True)

    async def stop(self) -> None:  # pragma: no cover - needs hardware
        import sounddevice as sd

        sd.stop()
        if self._task and not self._task.done():
            self._task.cancel()
