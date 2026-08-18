# laconv 🗣️

> *conv* — for **conversation**. The part of a voice agent that is neither the
> model nor the codec.

**Turn-taking, endpointing and barge-in for voice agents.** LaConv is the layer
that decides *when to listen* and *when to stop speaking*.

- Pure Python, **zero runtime dependencies** (like its sibling [`soif`](https://github.com/Unchained-Labs/soif)).
- The turn-taking state machine is **unit-testable with no audio hardware** —
  that is the design constraint the whole architecture is bent around.
- Every gap is named below, in the same place you would look for the feature.

## The split

Three repos, three jobs, and the boundaries are not arbitrary:

| | decides | shape |
|---|---|---|
| **[LeClanker](https://github.com/guilyx/LeClanker)** | **what** to say | stateful agent, SSE token stream |
| **[LaVoix](https://github.com/Unchained-Labs/lavoix)** | **how** speech and text convert | stateless request/response STT + TTS |
| **LaConv** | **when** to listen, and when to stop speaking | stateful per-conversation |

LaVoix is stateless by design: give it a file, get text; give it text, get a
WAV. That is correct and complete for what it is, and it is also why it cannot
hold a conversation — a conversation is entirely a question of *timing*, and a
request/response service has no notion of time passing.

LaConv is deliberately **not** inside LeClanker. LeClanker's job is reasoning
and tools; it should not know what a 20 ms PCM frame is, and it should not grow
a second personality that owns microphones. The seam is narrow on purpose:
LaConv needs exactly three things from the outside world — `Stt`, `Agent`,
`Tts` (see `engines.py`) — and LeClanker satisfies `Agent` through its existing
`POST /api/chat/stream`, with no changes to LeClanker at all.

```
  mic ──► VAD ──► endpointer ──► [ TurnMachine ] ──► LaVoix STT ──► LeClanker
                                       │                                │
  speaker ◄── chunked TTS ◄── LaVoix ◄─┴──────── sentence chunker ◄─────┘
                    ▲
                    └── barge-in: STOP_TTS, immediately
```

## What LaConv actually does

- **VAD** — energy-based with an adaptive noise floor and hysteresis (`vad.py`).
- **Endpointing** — deciding a person finished a *sentence*, not just that the
  room went quiet: minimum speech length, silence hangover, a hard cap, and a
  pre-roll buffer so transcripts do not start mid-word (`endpoint.py`).
- **Barge-in** — three policies (`full`, `half_duplex`, `hold`), with `STOP_TTS`
  emitted before anything else on an interruption (`turn.py`).
- **Turn-taking** — `idle → listening → thinking → speaking`, with illegal
  transitions raising rather than being silently dropped.
- **Chunked TTS** — the reply is cut at sentence boundaries and synthesised
  clause by clause, so audio starts after the first clause rather than after
  the model finishes thinking (`chunking.py`).
- **WebSocket transport** — binary PCM up, binary WAV down, JSON control frames
  both ways, including the `stop_audio` frame that makes barge-in work on a
  *remote* device (`protocol.py`, `server.py`).

## Install

```bash
pip install laconv                    # core: zero dependencies
pip install "laconv[server]"          # + websockets, for the transport
pip install "laconv[device]"          # + sounddevice, for a local mic/speaker
```

## Run it without a microphone

```bash
laconv simulate --barge-in full --slow-agent
```

That plays a scripted conversation through the *real* VAD, endpointer, session
and turn machine with fake engines, and prints every state transition —
including the interruption. See also `examples/offline_conversation.py`.

## Quick start

```python
import asyncio
from laconv import ConversationSession, SessionConfig, BargeIn
from laconv.backends.lavoix import LavoixStt, LavoixTts
from laconv.backends.leclanker import LeClankerAgent

session = ConversationSession(
    stt=LavoixStt("http://127.0.0.1:8090"),
    agent=LeClankerAgent("http://127.0.0.1:8484"),
    tts=LavoixTts("http://127.0.0.1:8090"),
    config=SessionConfig(barge_in=BargeIn.HALF_DUPLEX, session_id="kitchen"),
)
session.subscribe(print)

async def main():
    while chunk := read_from_your_microphone():   # any length, s16le mono 16 kHz
        await session.push_audio(chunk)

asyncio.run(main())
```

Or serve it to remote devices:

```bash
laconv check                   # is LaVoix up? is LeClanker up?
laconv serve --port 8092       # ws://127.0.0.1:8092
```

## The turn machine, on its own

It is a pure function of events. No clock, no audio, no I/O — feed it events,
get back the new state and a list of actions for the caller to perform:

```python
from laconv import TurnMachine, Event, Action, BargeIn

m = TurnMachine(barge_in=BargeIn.FULL)
m.feed(Event.SPEECH_STARTED).actions        # [START_CAPTURE]
m.feed(Event.SPEECH_ENDED).actions          # [COMMIT_UTTERANCE]
m.feed(Event.AGENT_AUDIO_STARTED)
m.feed(Event.SPEECH_STARTED).actions        # [STOP_TTS, CANCEL_AGENT, START_CAPTURE]
```

Returning actions instead of calling `tts.stop()` directly is what makes
`tests/test_turn.py` a real test of barge-in rather than a smoke test. A voice
framework whose interruption logic can only be exercised with a mic in the room
is a framework whose interruption logic is never exercised.

## Barge-in modes, and why the default is the timid one

| mode | what happens when you talk over the agent | needs |
|---|---|---|
| `full` | TTS stops, the agent generation is cancelled, we listen | headphones, a directional mic, **or AEC** |
| `half_duplex` *(default)* | the mic is gated while audio plays; you cannot interrupt | nothing |
| `hold` | the interruption is detected and counted, the agent finishes | nothing |

The default is `half_duplex` because **LaConv does not ship acoustic echo
cancellation** (below). On an open-air speaker, `full` means the microphone
hears the agent's own voice, the VAD calls it speech, and the agent interrupts
itself mid-sentence. `half_duplex` fails as "you had to wait"; `full` without
AEC fails as "it cannot finish a sentence". Pick `full` when you know the
speaker is not audible to the mic.

Note that **all three modes let you interrupt while the agent is still
thinking** — the mic is only gated once audio is actually playing, and
interrupting a slow model is the interruption people want most.

## What is NOT here

These are named because a framework that silently lacks barge-in is worse than
one that documents the gap. None of the following exists in this repo; there is
no flag that enables them.

- **Acoustic echo cancellation (AEC).** Not implemented. Doing it properly is
  an adaptive filter (NLMS/AES) that needs a reference signal, a known
  loopback delay, and per-device tuning — a signal-processing project, not a
  weekend of Python, and impossible to do well without the hardware in hand.
  The consequence is the `half_duplex` default above. If your device or OS
  provides AEC (PulseAudio's `module-echo-cancel`, a browser's
  `echoCancellation: true`, most USB conference speakerphones), turn it on
  there and set `barge_in="full"`.
- **Wake word.** Not implemented. Every credible option (openWakeWord, Porcupine,
  Snowboy) is a model file plus a runtime, which is the opposite of a
  zero-dependency core. The `Vad` protocol is the hook: a wake-word detector
  that only answers `is_speech=True` after the phrase fires drops straight in.
- **Streaming STT.** Not available, because LaVoix does not expose it —
  `POST /v1/stt/transcribe` takes a whole file. LaConv is therefore
  *utterance*-streaming, not *word*-streaming: transcription starts when you
  stop talking. `Stt` is one async method, so a streaming backend can replace
  it without touching the turn machine, but partial-hypothesis endpointing
  (using the ASR's own confidence to decide you finished) is genuinely not
  possible today.
- **WebRTC.** A deliberate v1 non-goal, argued in `protocol.py`: doing it
  properly means `aiortc`, ICE, and a TURN server, which buys nothing on the
  LAN/Tailscale link this targets and costs real operational burden. The
  honest downside: over the open internet or flaky Wi-Fi, TCP head-of-line
  blocking will make audio stutter and this protocol has no concealment.
- **Speaker diarisation / multi-party.** One human, one agent. A second voice
  in the room is just more speech to this VAD.
- **A neural VAD.** Energy VAD separates speech from room tone; it does not
  separate speech from a slammed door, a dog, or a television. See below.
- **Authentication on the websocket.** There is none. Bind to `127.0.0.1` or
  put it behind Tailscale; do not expose port 8092 to a network you do not own.

## Plugging in a better VAD

`Vad` is a two-method protocol (`is_speech`, `reset`). Silero is the usual
upgrade, and costs about ten lines:

```python
import torch
from laconv import Endpointer, AudioFormat

class SileroVad:
    def __init__(self, threshold=0.5):
        self.model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
        self.threshold = threshold
    def is_speech(self, frame: bytes) -> bool:
        audio = torch.frombuffer(frame, dtype=torch.int16).float() / 32768.0
        return self.model(audio, 16000).item() >= self.threshold
    def reset(self) -> None:
        self.model.reset_states()

endpointer = Endpointer(fmt=AudioFormat(frame_ms=32), vad=SileroVad())
```

Nothing else changes: the turn machine never sees a frame.

## Wire protocol

Binary frames are audio, text frames are JSON. Never both.

```
client -> server   binary   mono s16le PCM at the rate agreed in `hello`
client -> server   text     {"type":"hello"|"bye"|"text"}
server -> client   binary   WAV to play
server -> client   text     {"type":"ready"|"state"|"transcript"|"delta"
                            |"speaking"|"interrupted"|"stop_audio"|"error"}
```

`stop_audio` is the one that matters: a remote device buffers the WAVs it has
been sent, so **barge-in is not implemented until the client drops its buffer
on `stop_audio`.** A client that ignores that frame does not have barge-in, no
matter what the server does.

The `ready` frame advertises capabilities. Absent means absent — `aec`,
`wake_word`, `webrtc` and `streaming_stt` are not in the list, and a test
asserts they never quietly appear.

## Layout

```
src/laconv/
  turn.py        the state machine — pure, no clock, no I/O
  vad.py         energy VAD + the Vad protocol
  endpoint.py    VAD -> "they finished a sentence", timed in frames not seconds
  chunking.py    token stream -> speakable sentences
  engines.py     the three protocols LaConv needs: Stt, Agent, Tts
  session.py     the async shell that wires it together and cancels things
  protocol.py    wire format (transport-agnostic) + speakers
  server.py      websocket transport (needs the `server` extra)
  backends/
    lavoix.py    STT + TTS over HTTP, stdlib only
    leclanker.py SSE agent client, stdlib only
```

Everything above `session.py` in that list is synchronous and pure. That is not
tidiness for its own sake: it is why `pytest -q` covers barge-in, endpointing
timing, and the generation fence that stops an interrupted turn resurfacing.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

No test touches an audio device. Timing assertions are exact because elapsed
time in `endpoint.py` is counted in *frames consumed*, not read off a clock —
"700 ms of silence ends the turn" is a test of the rule, not of the CI runner's
mood.

## Ecosystem

- LaVoix on `:8090` — STT/TTS
- LeClanker on `:8484` — the agent
- LaConv on `:8092` — this, when serving remote devices

A small `voice` connector inside LeClanker (so it can originate calls rather
than only answer them) is a plausible next step and is deliberately not built
here.

## License

MIT.
