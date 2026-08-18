"""The wire protocol, without opening a socket."""

from __future__ import annotations

import json

import pytest

from laconv.audio import AudioFormat
from laconv.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    WireSpeaker,
    decode_control,
    encode,
    parse_hello,
    ready,
)
from laconv.turn import BargeIn


def test_hello_defaults_to_16k_20ms_half_duplex():
    hello = parse_hello(json.dumps({"type": "hello"}))
    assert hello.sample_rate == 16000
    assert hello.frame_ms == 20
    assert hello.barge_in is BargeIn.HALF_DUPLEX
    assert hello.fmt.frame_bytes == 640


def test_hello_rejects_a_rate_it_cannot_frame():
    """Better a refused connection than a session that transcribes badly and
    looks like a bad STT model."""
    with pytest.raises(Exception, match="whole number of samples"):
        parse_hello({"type": "hello", "sample_rate": 44100, "frame_ms": 1})


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"type": "bye"}, "expected hello"),
        ({"type": "hello", "version": 99}, "unsupported protocol version"),
        ({"type": "hello", "barge_in": "telepathy"}, "unknown barge_in"),
    ],
)
def test_hello_rejections(payload, match):
    with pytest.raises(ProtocolError, match=match):
        parse_hello(payload)


def test_control_frames_must_be_json_objects_with_a_type():
    with pytest.raises(ProtocolError, match="not JSON"):
        decode_control("{not json")
    with pytest.raises(ProtocolError, match="must be a JSON object"):
        decode_control("[1, 2]")
    with pytest.raises(ProtocolError, match="no 'type'"):
        decode_control('{"text": "hi"}')


def test_ready_advertises_only_what_exists():
    payload = json.loads(ready(AudioFormat(), "kitchen", BargeIn.FULL))
    assert payload["type"] == "ready"
    assert payload["version"] == PROTOCOL_VERSION
    assert payload["session_id"] == "kitchen"
    assert payload["barge_in"] == "full"
    # Absent means absent. If these ever appear here they must also be built.
    for absent in ("aec", "wake_word", "webrtc", "streaming_stt"):
        assert absent not in payload["capabilities"]


async def test_wire_speaker_stop_sends_stop_audio():
    sent_text: list[str] = []
    sent_bytes: list[bytes] = []

    async def send_bytes(data: bytes) -> None:
        sent_bytes.append(data)

    async def send_text(data: str) -> None:
        sent_text.append(data)

    speaker = WireSpeaker(send_bytes, send_text)
    await speaker.play(b"RIFFaudio")
    assert sent_bytes == [b"RIFFaudio"]
    assert sent_text == []

    await speaker.stop()
    assert json.loads(sent_text[-1]) == {"type": "stop_audio"}


async def test_wire_speaker_repeats_stop_if_it_raced_a_send():
    """A device that already queued the interrupted WAV would otherwise play
    it after the stop_audio that was meant to cancel it."""
    sent_text: list[str] = []
    speaker = WireSpeaker(None, None)  # type: ignore[arg-type]

    async def send_bytes(data: bytes) -> None:  # noqa: ARG001
        await speaker.stop()  # the interruption lands mid-send

    async def send_text(data: str) -> None:
        sent_text.append(data)

    speaker._send_bytes = send_bytes
    speaker._send_text = send_text

    await speaker.play(b"RIFFaudio")
    assert [json.loads(t)["type"] for t in sent_text] == ["stop_audio", "stop_audio"]


def test_encode_round_trips():
    assert json.loads(encode("delta", text="hi")) == {"type": "delta", "text": "hi"}
