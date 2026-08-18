"""`laconv` command line: serve, check, and a conversation you can run offline.

`laconv simulate` is the one that matters for review -- it plays a scripted
conversation through the real session, VAD, endpointer and turn machine with
fake engines, so you can see the state transitions and the barge-in without
owning a microphone.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys

from laconv import __version__
from laconv.audio import AudioFormat, silence, tone
from laconv.session import ConversationSession, SessionConfig, SessionEvent
from laconv.turn import BargeIn


def _print_event(event: SessionEvent) -> None:
    detail = " ".join(f"{k}={v}" for k, v in event.detail.items())
    text = f" {event.text!r}" if event.text else ""
    print(f"  [{event.state.value:9}] {event.type}{text}{(' ' + detail) if detail else ''}")


async def _simulate(barge_in: BargeIn, slow_agent: bool) -> int:
    from laconv.engines import Utterance

    fmt = AudioFormat()

    class FakeStt:
        async def transcribe(self, pcm: bytes, sample_rate: int) -> Utterance:  # noqa: ARG002
            return Utterance(text="what is the weather like today")

    class FakeAgent:
        async def reply(self, text: str, *, session_id: str):  # noqa: ARG002
            for part in ["It is sunny. ", "Fifteen degrees, ", "with a light wind. ",
                         "Rain is expected after six."]:
                if slow_agent:
                    await asyncio.sleep(0.25)
                yield part

    class FakeTts:
        async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:  # noqa: ARG002
            return tone(len(text) * 10, fmt)

    session = ConversationSession(
        FakeStt(), FakeAgent(), FakeTts(),
        config=SessionConfig(fmt=fmt, barge_in=barge_in),
    )
    session.subscribe(_print_event)

    print(f"barge_in={barge_in.value}\n")
    print("-- the human speaks for 800 ms, then goes quiet --")
    await session.push_audio(silence(400, fmt))  # let the VAD learn the room
    await session.push_audio(tone(800, fmt))
    await session.push_audio(silence(900, fmt))

    # Deliberately do NOT wait for the turn: the point of this demo is to
    # start talking over the agent while it is mid-answer.
    await asyncio.sleep(0.4)
    print(f"\n-- the human starts talking again (state={session.state.value}) --")
    await session.push_audio(tone(400, fmt))
    await session.push_audio(silence(900, fmt))
    await session.wait_for_turn()
    await session.close()

    print(f"\nturns={session.machine.turns} interruptions={session.machine.interruptions}")
    for turn in session.transcript.turns:
        print(f"  {turn.role}: {turn.text[:70]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="laconv", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"laconv {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="run a scripted conversation, no hardware")
    sim.add_argument("--barge-in", choices=[m.value for m in BargeIn],
                     default=BargeIn.FULL.value)
    sim.add_argument("--slow-agent", action="store_true",
                     help="add think-time so the interruption lands mid-reply")

    srv = sub.add_parser("serve", help="websocket server in front of LaVoix + LeClanker")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8092)
    srv.add_argument("--lavoix", default="http://127.0.0.1:8090")
    srv.add_argument("--leclanker", default="http://127.0.0.1:8484")

    chk = sub.add_parser("check", help="are LaVoix and LeClanker reachable?")
    chk.add_argument("--lavoix", default="http://127.0.0.1:8090")
    chk.add_argument("--leclanker", default="http://127.0.0.1:8484")

    args = parser.parse_args(argv)

    if args.command == "simulate":
        return asyncio.run(_simulate(BargeIn(args.barge_in), args.slow_agent))

    if args.command == "serve":
        from laconv.backends.lavoix import LavoixStt, LavoixTts
        from laconv.backends.leclanker import LeClankerAgent
        from laconv.server import VoiceServer

        def engines():
            return (
                LavoixStt(args.lavoix),
                LeClankerAgent(args.leclanker),
                LavoixTts(args.lavoix),
            )

        server = VoiceServer(engines, host=args.host, port=args.port)
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(server.serve_forever())
        return 0

    if args.command == "check":
        import urllib.error
        import urllib.request

        ok = True
        for name, url in (("lavoix", f"{args.lavoix}/healthz"),
                          ("leclanker", f"{args.leclanker}/api/health")):
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    print(f"  {name:10} {response.status} {url}")
            except (urllib.error.URLError, OSError) as exc:
                ok = False
                print(f"  {name:10} UNREACHABLE {url} ({exc})")
        return 0 if ok else 1

    return 2  # pragma: no cover - argparse rejects unknown commands first


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
