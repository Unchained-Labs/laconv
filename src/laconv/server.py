"""WebSocket transport: one connection, one `ConversationSession`.

The only module that needs a third-party package (`websockets`), which is why
it is an optional extra and why every decision it could have made lives in
`protocol.py` instead. Import it and you need the extra; import anything else
in LaConv and you need nothing.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from laconv import protocol
from laconv.engines import Agent, Stt, Tts
from laconv.protocol import WireSpeaker
from laconv.session import ConversationSession, SessionConfig, SessionEvent

log = logging.getLogger("laconv.server")

# Which session events are worth a text frame, and under what name. Events not
# listed are internal -- a device does not need to know the mic was re-armed.
_FORWARD = {
    "state": "state",
    "transcript": "transcript",
    "delta": "delta",
    "speaking": "speaking",
    "interrupted": "interrupted",
    "error": "error",
    "listening": "listening",
    "closed": "closed",
}

EngineFactory = Callable[[], "tuple[Stt, Agent, Tts]"]


class VoiceServer:
    """Serves `ws://host:port/` to microphone clients.

    `engines` is a factory rather than three instances so each connection can
    get its own agent state (LeClanker threads are per session_id) without the
    server knowing anything about what an agent is.
    """

    def __init__(
        self,
        engines: EngineFactory,
        host: str = "127.0.0.1",
        port: int = 8092,
        max_sessions: int = 8,
    ) -> None:
        self.engines = engines
        self.host = host
        self.port = port
        # A cap, because each session holds an utterance buffer and an agent
        # connection; an unbounded server on a Raspberry Pi is a memory bug
        # waiting for a badly written client to reconnect in a loop.
        self.max_sessions = max_sessions
        self.sessions: dict[int, ConversationSession] = {}

    async def serve_forever(self) -> None:
        try:
            import websockets
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError(
                "the websocket server needs the 'server' extra: "
                "pip install 'laconv[server]'"
            ) from exc

        async with websockets.serve(self.handle, self.host, self.port, max_size=2**22):
            log.info("laconv listening on ws://%s:%d", self.host, self.port)
            await asyncio.Future()  # run until cancelled

    async def handle(self, websocket) -> None:  # noqa: ANN001 - websockets' own type
        if len(self.sessions) >= self.max_sessions:
            await websocket.send(protocol.encode("error", message="server is full"))
            await websocket.close()
            return

        session: ConversationSession | None = None
        key = id(websocket)
        try:
            first = await websocket.recv()
            hello = protocol.parse_hello(first)

            stt, agent, tts = self.engines()
            speaker = WireSpeaker(websocket.send, websocket.send)
            session = ConversationSession(
                stt,
                agent,
                tts,
                speaker=speaker,
                config=SessionConfig(
                    fmt=hello.fmt,
                    barge_in=hello.barge_in,
                    session_id=hello.session_id,
                ),
            )
            session.subscribe(_forwarder(websocket))
            self.sessions[key] = session
            await websocket.send(
                protocol.ready(hello.fmt, hello.session_id, hello.barge_in)
            )

            async for message in websocket:
                if isinstance(message, bytes):
                    await session.push_audio(message)
                    continue
                payload = protocol.decode_control(message)
                kind = payload.get("type")
                if kind == "bye":
                    break
                if kind == "text":
                    # A typed message from a client with a keyboard. Goes
                    # through the same turn machinery so a mixed text/voice
                    # client cannot desynchronise the state machine.
                    await session.inject_text(payload.get("text", ""))
                else:
                    log.debug("ignoring control frame %r", kind)
        except protocol.ProtocolError as exc:
            await websocket.send(protocol.encode("error", message=str(exc)))
        except Exception:  # noqa: BLE001 - one bad client is not a dead server
            log.exception("session failed")
        finally:
            self.sessions.pop(key, None)
            if session is not None:
                try:
                    await session.close()
                except Exception:  # noqa: BLE001
                    # Closing sends `stop_audio`, which fails when the client
                    # hung up first -- the common case, not an error. The
                    # session's own cleanup has already run by then.
                    log.debug("close raced the client hanging up")


def _forwarder(websocket) -> Callable[[SessionEvent], object]:  # noqa: ANN001
    async def send(event: SessionEvent) -> None:
        name = _FORWARD.get(event.type)
        if name is None:
            return
        try:
            await websocket.send(
                protocol.encode(name, state=event.state.value, text=event.text, **event.detail)
            )
        except Exception:  # noqa: BLE001 - the connection is going away anyway
            log.debug("dropped %s for a closed socket", name)

    return send
