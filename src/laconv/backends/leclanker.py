"""The agent side: LeClanker's `POST /api/chat/stream` (SSE).

LeClanker decides *what* to say. This module is the thinnest possible adapter
between its SSE event stream and `laconv.engines.Agent`.

Event shapes LeClanker emits (from its runtime.chat_stream):

    {"type": "token",       "text": "..."}       incremental reply text
    {"type": "tool",        "name": ..., ...}    a tool started
    {"type": "tool_result", "name": ..., ...}    a tool finished
    {"type": "error",       "error": "..."}
    {"type": "done",        "reply": "...", "streamed": bool}

Only `token` is speakable. `tool` / `tool_result` are dropped rather than
narrated -- reading "calling web_search" aloud is a design choice a caller can
make on top (subscribe to `on_tool`), not one a voice framework should impose.

`done` carries the complete reply, and LeClanker sets `streamed: false` when no
token ever arrived (e.g. a non-streaming model). That is the case this adapter
must not get wrong: if we ignore `done`, those replies are silent.

Why raw sockets instead of httpx/aiohttp: same reason as the LaVoix backend --
zero runtime dependencies. SSE over `asyncio.open_connection` is ~50 lines
because the framing is "lines, blank line ends an event", and we only need to
parse `data:`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import ssl
from collections.abc import AsyncIterator
from urllib.parse import urlsplit


class LeClankerError(RuntimeError):
    pass


class LeClankerAgent:
    """Streams LeClanker replies as text deltas.

    Cancellation: the async generator below is cancelled by
    `ConversationSession` on barge-in. It must not swallow `CancelledError` --
    note that the cleanup is in `finally`, not in `except Exception`, precisely
    so an interruption closes the socket instead of being caught and logged
    while the model keeps generating.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8484",
        thread_prefix: str = "laconv",
        connect_timeout_s: float = 10.0,
        speak_errors: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.thread_prefix = thread_prefix
        self.connect_timeout_s = connect_timeout_s
        # An error the human can hear beats an error only the log has: on a
        # voice device there is no console to check.
        self.speak_errors = speak_errors
        self.on_tool: list = []

    async def reply(self, text: str, *, session_id: str) -> AsyncIterator[str]:
        thread_id = f"{self.thread_prefix}:{session_id}"
        body = json.dumps({"message": text, "thread_id": thread_id}).encode()
        streamed = False
        async for event in self._sse("/api/chat/stream", body):
            kind = event.get("type")
            if kind == "token":
                chunk = event.get("text") or ""
                if chunk:
                    streamed = True
                    yield chunk
            elif kind == "error":
                message = event.get("error") or "the agent failed"
                if self.speak_errors and not streamed:
                    streamed = True
                    yield f"Sorry, something went wrong: {message}"
                else:
                    raise LeClankerError(message)
            elif kind == "done":
                if not streamed:
                    # Non-streaming model path. Without this the caller hears
                    # nothing at all and concludes the mic is broken.
                    reply = (event.get("reply") or "").strip()
                    if reply:
                        yield reply
                return
            elif kind in ("tool", "tool_result"):
                for hook in self.on_tool:
                    hook(event)

    # -- minimal SSE client -------------------------------------------------

    async def _sse(self, path: str, body: bytes) -> AsyncIterator[dict]:
        parts = urlsplit(self.base_url)
        host = parts.hostname or "127.0.0.1"
        secure = parts.scheme == "https"
        port = parts.port or (443 if secure else 80)
        context = ssl.create_default_context() if secure else None

        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port, ssl=context), self.connect_timeout_s
        )
        try:
            request = (
                f"POST {parts.path}{path} HTTP/1.1\r\n"
                f"Host: {parts.netloc}\r\n"
                "Content-Type: application/json\r\n"
                "Accept: text/event-stream\r\n"
                f"Content-Length: {len(body)}\r\n"
                "Connection: close\r\n\r\n"
            ).encode()
            writer.write(request + body)
            await writer.drain()

            status_line = await reader.readline()
            if not status_line:
                raise LeClankerError("LeClanker closed the connection immediately")
            code = int(status_line.split()[1])
            while True:
                header = await reader.readline()
                if header in (b"\r\n", b"\n", b""):
                    break
            if code != 200:
                rest = await reader.read(400)
                raise LeClankerError(f"{path} -> {code}: {rest.decode('utf-8', 'replace')}")

            # `Connection: close` above means no chunked framing to unpick --
            # the body is a plain byte stream until EOF. Worth the header: a
            # chunked-transfer parser is another 40 lines for zero benefit on
            # a stream we read to the end anyway.
            async for line in reader:
                text = line.decode("utf-8", "replace").rstrip("\r\n")
                if not text.startswith("data:"):
                    continue  # blank separator lines and any `event:` fields
                payload = text[5:].strip()
                if not payload:
                    continue
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue  # a keep-alive comment or a truncated frame
        finally:
            writer.close()
            # Already tearing down; a peer that hung up first is not an error.
            with contextlib.suppress(Exception):
                await writer.wait_closed()


class EchoAgent:
    """Repeats what it heard. For wiring up a device before LeClanker exists."""

    async def reply(self, text: str, *, session_id: str) -> AsyncIterator[str]:  # noqa: ARG002
        yield f"You said: {text}"
