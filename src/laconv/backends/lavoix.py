"""STT and TTS through LaVoix (https://github.com/Unchained-Labs/lavoix).

LaVoix is stateless request/response: `POST /v1/stt/transcribe` takes a whole
file, `POST /v1/tts/synthesize` returns a whole WAV. That is the correct shape
for LaVoix and the reason LaConv exists -- conversation is the part it does not
do.

Implemented on `urllib` rather than `httpx`, which LaVoix's own client uses.
Why: LaConv's core has zero runtime dependencies and this is the only HTTP the
default path needs, so adding httpx would double the install for one POST. The
cost is that we build a multipart body by hand (35 lines, below) and that the
calls are blocking, hence `asyncio.to_thread`.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
import uuid

from laconv.audio import AudioFormat, to_wav
from laconv.engines import Utterance


class LavoixError(RuntimeError):
    pass


def _multipart(fields: dict[str, str], filename: str, payload: bytes) -> tuple[bytes, str]:
    boundary = f"----laconv{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n"
            f"{value}\r\n".encode()
        )
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{filename}\"\r\nContent-Type: audio/wav\r\n\r\n".encode()
    )
    parts.append(payload)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _post(url: str, data: bytes, content_type: str, timeout: float) -> tuple[bytes, str]:
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": content_type}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(), response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        raise LavoixError(f"{url} -> {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise LavoixError(f"{url} unreachable: {exc.reason}") from exc


class LavoixStt:
    """`POST /v1/stt/transcribe`, one utterance at a time."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8090",
        provider: str | None = None,
        language: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.provider = provider
        self.language = language
        self.timeout_s = timeout_s

    async def transcribe(self, pcm: bytes, sample_rate: int) -> Utterance:
        fmt = AudioFormat(sample_rate=sample_rate)
        wav = to_wav(pcm, fmt)
        fields = {k: v for k, v in
                  (("provider", self.provider), ("language", self.language)) if v}
        body, content_type = _multipart(fields, "utterance.wav", wav)
        raw, _ = await asyncio.to_thread(
            _post, f"{self.base_url}/v1/stt/transcribe", body, content_type, self.timeout_s
        )
        payload = json.loads(raw.decode("utf-8"))
        return Utterance(
            text=payload.get("text", ""),
            language=payload.get("language"),
            duration_ms=fmt.ms_for(pcm),
            raw=payload,
        )


class LavoixTts:
    """`POST /v1/tts/synthesize`, one sentence chunk at a time.

    Chunk-at-a-time is what `chunking.py` buys us: LaVoix has no streaming
    synthesis endpoint, so the only way to start speaking before the model has
    finished thinking is to synthesise the first clause on its own.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8090",
        provider: str | None = None,
        voice: str = "default",
        speed: float = 1.0,
        timeout_s: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.provider = provider
        self.voice = voice
        self.speed = speed
        self.timeout_s = timeout_s

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        payload = json.dumps(
            {
                "text": text,
                "voice": voice or self.voice,
                "speed": self.speed,
                "provider": self.provider,
            }
        ).encode()
        audio, content_type = await asyncio.to_thread(
            _post,
            f"{self.base_url}/v1/tts/synthesize",
            payload,
            "application/json",
            self.timeout_s,
        )
        if content_type.startswith("application/json"):
            # LaVoix answers errors as JSON with a 200 in some provider paths;
            # handing that to a sound card produces a burst of noise, which is
            # a memorably bad way to learn your API key expired.
            raise LavoixError(f"expected audio, got JSON: {audio[:200]!r}")
        return audio


def healthz(base_url: str = "http://127.0.0.1:8090", timeout_s: float = 5.0) -> dict:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/healthz", timeout=timeout_s) as r:
        return json.loads(r.read().decode())
