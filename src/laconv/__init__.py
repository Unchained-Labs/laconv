"""LaConv -- the conversation layer for voice agents.

LeClanker decides WHAT to say. LaVoix turns speech into text and text into
speech. LaConv decides WHEN to listen and when to stop speaking.

Nothing imported here needs a microphone, a sound card, or a network. The
backends (`laconv.backends.*`) and the websocket server (`laconv.server`) are
imported explicitly so that `import laconv` stays dependency-free.
"""

from laconv.audio import AudioFormat, silence, to_wav, tone
from laconv.chunking import SentenceChunker, Transcript, chunk_text
from laconv.endpoint import Boundary, Endpointer
from laconv.engines import Agent, NullSpeaker, Speaker, Stt, Tts, Utterance
from laconv.session import ConversationSession, SessionConfig, SessionEvent
from laconv.turn import (
    Action,
    BargeIn,
    Event,
    IllegalTransition,
    State,
    Transition,
    TurnMachine,
)
from laconv.vad import EnergyVad, ScriptedVad, Vad

__version__ = "0.1.0"

__all__ = [
    "Action",
    "Agent",
    "AudioFormat",
    "BargeIn",
    "Boundary",
    "ConversationSession",
    "EnergyVad",
    "Endpointer",
    "Event",
    "IllegalTransition",
    "NullSpeaker",
    "ScriptedVad",
    "SentenceChunker",
    "SessionConfig",
    "SessionEvent",
    "Speaker",
    "State",
    "Stt",
    "Transcript",
    "Transition",
    "Tts",
    "TurnMachine",
    "Utterance",
    "Vad",
    "__version__",
    "chunk_text",
    "silence",
    "to_wav",
    "tone",
]
