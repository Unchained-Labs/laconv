#!/usr/bin/env python3
"""Talk to LeClanker through the machine's own microphone.

    pip install "laconv[device]"
    python examples/mic_to_leclanker.py

Needs real hardware, so nothing in `tests/` covers it -- that is the point of
`offline_conversation.py`, which covers the same code path without a mic.

Requires LaVoix on :8090 and LeClanker on :8484 (`laconv check`).
"""

from __future__ import annotations

import asyncio
import queue
import sys

from laconv import AudioFormat, ConversationSession, SessionConfig, SessionEvent
from laconv.backends.lavoix import LavoixStt, LavoixTts
from laconv.backends.leclanker import LeClankerAgent
from laconv.protocol import LocalSpeaker
from laconv.turn import BargeIn

FMT = AudioFormat()
LAVOIX = "http://127.0.0.1:8090"
LECLANKER = "http://127.0.0.1:8484"

# half_duplex is the honest default on a laptop: the built-in mic hears the
# built-in speaker, and without AEC `full` makes the agent interrupt itself
# roughly one sentence in. Put on headphones and this becomes BargeIn.FULL.
BARGE_IN = BargeIn.HALF_DUPLEX


def show(event: SessionEvent) -> None:
    if event.type == "transcript":
        print(f"\nyou: {event.text}")
    elif event.type == "speaking":
        print(f"agent: {event.text}")
    elif event.type == "interrupted":
        print("  (interrupted)")
    elif event.type == "error":
        print(f"  ! {event.text}", file=sys.stderr)


async def main() -> None:
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("pip install 'laconv[device]'") from None

    session = ConversationSession(
        LavoixStt(LAVOIX),
        LeClankerAgent(LECLANKER),
        LavoixTts(LAVOIX),
        speaker=LocalSpeaker(FMT.sample_rate),
        config=SessionConfig(fmt=FMT, barge_in=BARGE_IN, session_id="laptop"),
    )
    session.subscribe(show)

    # sounddevice calls back on its own thread; the session is single-loop and
    # not thread-safe, so the queue is the handoff. Doing the session work
    # inside the callback would block the audio thread and drop frames --
    # which, in a VAD pipeline, means missing exactly the frames that carry
    # the start of speech.
    frames: queue.Queue[bytes] = queue.Queue()

    def on_audio(indata, frames_count, time_info, status) -> None:  # noqa: ANN001, ARG001
        frames.put(bytes(indata))

    print(f"listening (barge_in={BARGE_IN.value}). ctrl-c to stop.")
    with sd.RawInputStream(
        samplerate=FMT.sample_rate,
        blocksize=FMT.frame_samples,
        dtype="int16",
        channels=1,
        callback=on_audio,
    ):
        try:
            while True:
                try:
                    chunk = frames.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.005)
                    continue
                await session.push_audio(chunk)
        except KeyboardInterrupt:
            pass
        finally:
            await session.close()


if __name__ == "__main__":
    asyncio.run(main())
