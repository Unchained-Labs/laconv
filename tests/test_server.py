"""The websocket transport, over a real loopback socket.

Skipped when the `server` extra is absent, so the core test suite still runs
with zero dependencies installed. CI installs it.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from laconv.audio import AudioFormat, silence, tone
from laconv.engines import Utterance
from laconv.server import VoiceServer

websockets = pytest.importorskip("websockets")

FMT = AudioFormat()


class FakeStt:
    async def transcribe(self, pcm: bytes, sample_rate: int) -> Utterance:  # noqa: ARG002
        return Utterance(text="hello from the wire")


class FakeAgent:
    async def reply(self, text: str, *, session_id: str):  # noqa: ARG002
        yield "Right away. "
        yield "Here is the answer."


class FakeTts:
    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:  # noqa: ARG002
        return b"RIFF" + text.encode()


@pytest.fixture
async def serving():
    server = VoiceServer(lambda: (FakeStt(), FakeAgent(), FakeTts()), port=0)
    async with websockets.serve(server.handle, "127.0.0.1", 0) as ws_server:
        port = ws_server.sockets[0].getsockname()[1]
        yield server, f"ws://127.0.0.1:{port}"


async def read_until(ws, wanted: str, budget: float = 5.0, audio: list | None = None) -> dict:
    """Collect frames until a control frame of type `wanted` shows up.

    Binary frames go into `audio` rather than being dropped: descriptive text
    frames travel the fire-and-forget observer path, so a WAV can legitimately
    arrive *before* the `transcript` that describes the turn it belongs to. A
    helper that quietly discarded them made this suite pass on 3.10 and time
    out on 3.12 for reasons that had nothing to do with the server.
    """
    deadline = asyncio.get_running_loop().time() + budget
    while asyncio.get_running_loop().time() < deadline:
        message = await asyncio.wait_for(ws.recv(), timeout=budget)
        if isinstance(message, bytes):
            if audio is not None:
                audio.append(message)
            continue
        payload = json.loads(message)
        if payload.get("type") == wanted:
            return payload
    raise AssertionError(f"never saw {wanted}")


async def drain_turn(ws, budget: float = 5.0) -> tuple[list[bytes], list[dict]]:
    """Read every frame of one turn, up to the transition back to idle."""
    audio: list[bytes] = []
    control: list[dict] = []
    deadline = asyncio.get_running_loop().time() + budget
    while asyncio.get_running_loop().time() < deadline:
        message = await asyncio.wait_for(ws.recv(), timeout=budget)
        if isinstance(message, bytes):
            audio.append(message)
            continue
        payload = json.loads(message)
        control.append(payload)
        if payload.get("type") == "state" and payload.get("to") == "idle":
            return audio, control
    raise AssertionError("the turn never came back to idle")


async def test_a_full_turn_over_the_wire(serving):
    _, url = serving
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"type": "hello", "session_id": "kitchen"}))
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready"
        assert ready["sample_rate"] == 16000
        assert "aec" not in ready["capabilities"]

        await ws.send(silence(400, FMT))
        await ws.send(tone(600, FMT))
        await ws.send(silence(900, FMT))

        audio, control = await drain_turn(ws)

        by_type: dict[str, list[dict]] = {}
        for frame in control:
            by_type.setdefault(frame["type"], []).append(frame)

        assert [f["text"] for f in by_type["transcript"]] == ["hello from the wire"]
        assert [f["text"] for f in by_type["speaking"]] == [
            "Right away.",
            "Here is the answer.",
        ]
        # One WAV per sentence: the reply was not synthesised in one blocking
        # call at the end.
        assert len(audio) == 2
        assert all(a.startswith(b"RIFF") for a in audio)


async def test_a_bad_hello_is_refused_with_a_reason(serving):
    _, url = serving
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"type": "hello", "version": 99}))
        error = json.loads(await ws.recv())
        assert error["type"] == "error"
        assert "unsupported protocol version" in error["message"]


async def test_a_typed_message_takes_a_turn(serving):
    _, url = serving
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"type": "hello"}))
        await read_until(ws, "ready")
        await ws.send(json.dumps({"type": "text", "text": "what time is it"}))
        transcript = await read_until(ws, "transcript")
        assert transcript["text"] == "what time is it"


async def test_the_server_refuses_connections_past_its_cap():
    server = VoiceServer(lambda: (FakeStt(), FakeAgent(), FakeTts()), max_sessions=1)
    async with websockets.serve(server.handle, "127.0.0.1", 0) as ws_server:
        port = ws_server.sockets[0].getsockname()[1]
        url = f"ws://127.0.0.1:{port}"
        async with websockets.connect(url) as first:
            await first.send(json.dumps({"type": "hello", "session_id": "a"}))
            await read_until(first, "ready")

            async with websockets.connect(url) as second:
                error = json.loads(await second.recv())
                assert error["type"] == "error"
                assert "full" in error["message"]


async def test_barge_in_sends_stop_audio_down_the_wire(serving):
    """Remote barge-in is only real if the device is told to drop its buffer."""

    class SlowAgent:
        async def reply(self, text: str, *, session_id: str):  # noqa: ARG002
            for i in range(30):
                await asyncio.sleep(0.02)
                yield f"Sentence {i}. "

    server, url = serving
    server.engines = lambda: (FakeStt(), SlowAgent(), FakeTts())

    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"type": "hello", "barge_in": "full"}))
        await read_until(ws, "ready")

        await ws.send(silence(400, FMT))
        await ws.send(tone(600, FMT))
        await ws.send(silence(900, FMT))
        heard: list[bytes] = []
        await read_until(ws, "speaking", audio=heard)

        await ws.send(tone(400, FMT))  # talk over it
        stop = await read_until(ws, "stop_audio", audio=heard)
        assert stop["type"] == "stop_audio"
