"""The LaVoix and LeClanker adapters, against throwaway local servers.

Real sockets, no mocks: the parsers in these two modules are hand-rolled
(because the core has no HTTP dependency), so mocking them out would test
nothing. Both servers here are ~20 lines and bind to an ephemeral port.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from laconv.backends.lavoix import LavoixError, LavoixStt, LavoixTts
from laconv.backends.leclanker import EchoAgent, LeClankerAgent, LeClankerError

# -- LeClanker SSE ---------------------------------------------------------


async def sse_server(events: list[dict], status: int = 200, raw: str | None = None):
    """A one-shot HTTP server that replies with an SSE stream."""
    received: dict = {}

    async def handle(reader, writer):
        line = await reader.readline()
        received["request"] = line.decode()
        length = 0
        while True:
            header = await reader.readline()
            if header in (b"\r\n", b"\n", b""):
                break
            name, _, value = header.decode().partition(":")
            if name.lower() == "content-length":
                length = int(value.strip())
        received["body"] = json.loads((await reader.readexactly(length)).decode())

        body = raw if raw is not None else "".join(
            f"data: {json.dumps(e)}\n\n" for e in events
        )
        writer.write(
            f"HTTP/1.1 {status} OK\r\nContent-Type: text/event-stream\r\n\r\n".encode()
            + body.encode()
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, port, received


async def collect(agent: LeClankerAgent, text: str = "hi") -> list[str]:
    return [delta async for delta in agent.reply(text, session_id="test")]


async def test_tokens_stream_through_and_done_ends_it():
    server, port, received = await sse_server(
        [
            {"type": "token", "text": "It is "},
            {"type": "tool", "name": "weather"},
            {"type": "token", "text": "sunny."},
            {"type": "done", "reply": "It is sunny.", "streamed": True},
        ]
    )
    async with server:
        agent = LeClankerAgent(f"http://127.0.0.1:{port}")
        tools: list[dict] = []
        agent.on_tool.append(tools.append)
        assert await collect(agent) == ["It is ", "sunny."]

    assert received["body"] == {"message": "hi", "thread_id": "laconv:test"}
    # Tool events are surfaced to a hook but never spoken.
    assert [t["name"] for t in tools] == ["weather"]


async def test_a_non_streaming_reply_is_recovered_from_done():
    """LeClanker sets streamed=false for models that do not stream. Ignoring
    `done` here would make those replies completely silent."""
    server, port, _ = await sse_server(
        [{"type": "done", "reply": "The whole answer.", "streamed": False}]
    )
    async with server:
        agent = LeClankerAgent(f"http://127.0.0.1:{port}")
        assert await collect(agent) == ["The whole answer."]


async def test_an_error_before_any_token_is_spoken_not_swallowed():
    server, port, _ = await sse_server([{"type": "error", "error": "no model"}])
    async with server:
        agent = LeClankerAgent(f"http://127.0.0.1:{port}")
        spoken = await collect(agent)
    assert spoken and "no model" in spoken[0]


async def test_an_error_after_tokens_raises_instead_of_babbling():
    server, port, _ = await sse_server(
        [{"type": "token", "text": "It is "}, {"type": "error", "error": "boom"}]
    )
    async with server:
        agent = LeClankerAgent(f"http://127.0.0.1:{port}")
        with pytest.raises(LeClankerError, match="boom"):
            await collect(agent)


async def test_malformed_and_keepalive_lines_are_skipped():
    raw = (
        ": keep-alive\n\n"
        "event: message\n"
        'data: {"type": "token", "text": "ok"}\n\n'
        "data: {truncated\n\n"
        'data: {"type": "done", "reply": "ok", "streamed": true}\n\n'
    )
    server, port, _ = await sse_server([], raw=raw)
    async with server:
        agent = LeClankerAgent(f"http://127.0.0.1:{port}")
        assert await collect(agent) == ["ok"]


async def test_a_non_200_is_an_error_not_a_silent_empty_reply():
    server, port, _ = await sse_server([], status=500, raw="nope")
    async with server:
        agent = LeClankerAgent(f"http://127.0.0.1:{port}")
        with pytest.raises(LeClankerError, match="500"):
            await collect(agent)


async def test_cancelling_mid_stream_propagates():
    """Barge-in cancels this generator; swallowing CancelledError here is
    exactly how an interrupted agent keeps talking."""
    server, port, _ = await sse_server(
        [{"type": "token", "text": f"{i} "} for i in range(200)]
    )
    async with server:
        agent = LeClankerAgent(f"http://127.0.0.1:{port}")
        stream = agent.reply("hi", session_id="test")
        assert await stream.__anext__()
        await stream.aclose()  # must not hang or raise


async def test_echo_agent_needs_no_server():
    assert [d async for d in EchoAgent().reply("hello", session_id="x")] == [
        "You said: hello"
    ]


# -- LaVoix ----------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # noqa: ANN001 - silence the test output
        pass

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's naming
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self.server.last_body = body  # type: ignore[attr-defined]
        self.server.last_type = self.headers.get("Content-Type", "")  # type: ignore[attr-defined]

        mode = self.server.mode  # type: ignore[attr-defined]
        if self.path.endswith("/transcribe"):
            payload = json.dumps({"text": "hello world", "language": "en"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        elif mode == "json-error":
            payload = json.dumps({"detail": "key expired"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        elif mode == "http-error":
            payload = b"boom"
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
        else:
            payload = b"RIFF....WAVEfake"
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def lavoix_server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    server.mode = "audio"  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


async def test_stt_posts_a_real_wav_in_a_multipart_body(lavoix_server):
    server, url = lavoix_server
    pcm = b"\x01\x00" * 1600  # 100 ms at 16 kHz
    utterance = await LavoixStt(url, provider="mistral", language="en").transcribe(pcm, 16000)

    assert utterance.text == "hello world"
    assert utterance.duration_ms == pytest.approx(100.0)
    body = server.last_body  # type: ignore[attr-defined]
    assert b"multipart/form-data" in server.last_type.encode()  # type: ignore[attr-defined]
    assert b'name="provider"' in body and b"mistral" in body
    assert b'name="language"' in body
    assert b'filename="utterance.wav"' in body
    assert b"RIFF" in body and b"WAVE" in body


async def test_tts_returns_audio(lavoix_server):
    server, url = lavoix_server
    audio = await LavoixTts(url, voice="alto").synthesize("hello", voice="tenor")
    assert audio.startswith(b"RIFF")
    assert json.loads(server.last_body)["voice"] == "tenor"  # type: ignore[attr-defined]


async def test_tts_refuses_to_play_a_json_error_as_audio(lavoix_server):
    """Handing a JSON body to a sound card is a memorably bad way to learn
    your API key expired."""
    server, url = lavoix_server
    server.mode = "json-error"  # type: ignore[attr-defined]
    with pytest.raises(LavoixError, match="expected audio"):
        await LavoixTts(url).synthesize("hello")


async def test_http_errors_surface_with_the_body(lavoix_server):
    server, url = lavoix_server
    server.mode = "http-error"  # type: ignore[attr-defined]
    with pytest.raises(LavoixError, match="500"):
        await LavoixTts(url).synthesize("hello")


async def test_an_unreachable_lavoix_is_a_clear_error():
    with pytest.raises(LavoixError, match="unreachable"):
        # Port 1 is reserved and nothing listens on it.
        await LavoixTts("http://127.0.0.1:1", timeout_s=2).synthesize("hi")
