"""Free, local, and multilingual text-to-speech voices: `Edge` (Microsoft Edge free neural
TTS with word-boundary timing), `QwenTTS` (Alibaba's Apache-2.0 open-source `Qwen3-TTS`,
local or HTTP/OpenAI-compatible endpoint), and `SystemVoice` (offline OS-native TTS such as
macOS `say`, Linux `espeak-ng`, and Windows `System.Speech`)."""

import asyncio
import base64
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Unpack

from typing_extensions import TypedDict

from manimgx.audio.fal import _ordered, _pieces
from manimgx.audio.sound import Speech, timed

__all__ = [
    "Edge",
    "QwenTTS",
    "SystemVoice",
]

_HANGUL_RE = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]")
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_HANZI_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_CJK_CHAR_RE = re.compile(
    r"[\u1100-\u11ff\u3040-\u30ff\u3130-\u318f\u3400-\u9fff\uac00-\ud7af\uf900-\ufaff]"
)


def _detect_script(text: str) -> Literal["ko", "ja", "zh", "en"]:
    """Detect the primary script in `text` for automatic multilingual voice selection."""
    if _HANGUL_RE.search(text):
        return "ko"
    if _KANA_RE.search(text):
        return "ja"
    if _HANZI_RE.search(text):
        return "zh"
    return "en"


def _expand_cjk_pieces(
    pieces: Sequence[tuple[str, float, float]],
) -> list[tuple[str, float, float]]:
    """Subdivide multi-character CJK/Korean word tokens into syllable/character intervals.

    Services such as Microsoft Edge TTS emit `WordBoundary` timestamps per space-separated
    Korean Eojeol (for example `"삼각형의"` or `"원을"`), whereas `Speech` and `Scene.say`
    segment CJK and Hangul per syllable so bracketed spans like `"여기에 [원]을 그립니다."`
    can target the noun without its particle. Splitting multi-character CJK tokens
    proportionally across their alphanumeric characters preserves exact Eojeol boundaries
    while giving each syllable its own monotonic interval.
    """
    expanded: list[tuple[str, float, float]] = []
    for token, start, end in pieces:
        chars = [c for c in token if c.isalnum()]
        if len(chars) > 1 and _CJK_CHAR_RE.search(token):
            m = len(chars)
            span = max(0.0, end - start)
            for j, ch in enumerate(chars):
                expanded.append((ch, start + span * j / m, start + span * (j + 1) / m))
        else:
            expanded.append((token, start, end))
    return expanded


# ── Edge (Microsoft Edge free neural TTS) ───────────────────────────────────────────────


class EdgeSettings(TypedDict, total=False, closed=True):
    """Settings for Microsoft Edge's free neural text-to-speech ([`Edge`][manimgx.Edge])."""

    rate: str
    """Speaking rate adjustment, such as `"+0%"`, `"+15%"`, or `"-10%"` (default `"+0%"`)."""
    volume: str
    """Volume adjustment, such as `"+0%"` or `"+10%"` (default `"+0%"`)."""
    pitch: str
    """Pitch adjustment, such as `"+0Hz"` or `"-5Hz"` (default `"+0Hz"`)."""


_EDGE_DEFAULT_VOICES: dict[str, str] = {
    "ko": "ko-KR-SunHiNeural",
    "ja": "ja-JP-NanamiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "en": "en-US-AriaNeural",
}


@dataclass(frozen=True, slots=True, init=False, repr=False)
class Edge:
    """Microsoft Edge's free neural text-to-speech service, as a voice:
    `Edge(voice, **settings)`.

    Requires no API key, supports Korean (`"ko-KR-SunHiNeural"`, `"ko-KR-InJoonNeural"`),
    English (`"en-US-AriaNeural"`, `"en-US-GuyNeural"`), Japanese (`"ja-JP-NanamiNeural"`),
    Chinese (`"zh-CN-XiaoxiaoNeural"`), and 40+ languages, and extracts neural word-boundary
    timestamps for [`Scene.say`][manimgx.Scene.say]. When `voice` is `"auto"` (the default),
    the voice is chosen from the script of the text (`"ko-KR-SunHiNeural"` for Korean).

    Args:
        voice: The neural voice's name, or `"auto"` to pick by language.
        **settings: Rate, volume, and pitch settings ([`EdgeSettings`][manimgx.audio.tts.EdgeSettings]).

    Examples:
        Speak Korean or English with automatic word-level animation sync and no API key:

        ```py
        voice = m.Edge("ko-KR-SunHiNeural", rate="+5%")
        ```
    """

    voice: str
    """The neural voice's name, or `"auto"` to pick from the text's script."""
    settings: dict[str, object]
    """The voice's rate, volume, and pitch settings, as given."""

    def __init__(
        self,
        voice: str = "auto",
        **settings: Unpack[EdgeSettings],
    ) -> None:
        object.__setattr__(self, "voice", voice)
        object.__setattr__(
            self, "settings", {k: _ordered(v) for k, v in sorted(settings.items())}
        )

    def __repr__(self) -> str:
        """The call that makes it: what names it in the cache."""
        given = "".join(f", {k}={v!r}" for k, v in self.settings.items())
        return f"Edge({self.voice!r}{given})"

    def resolve_voice(self, text: str) -> str:
        """Return the concrete neural voice name for `text`."""
        if self.voice != "auto":
            return self.voice
        return _EDGE_DEFAULT_VOICES[_detect_script(text)]

    def __call__(self, text: str) -> Speech:
        chosen = self.resolve_voice(text)
        rate = str(self.settings.get("rate", "+0%"))
        volume = str(self.settings.get("volume", "+0%"))
        pitch = str(self.settings.get("pitch", "+0Hz"))
        audio, raw_pieces = _synthesize_edge(text, chosen, rate, volume, pitch)
        pieces = _expand_cjk_pieces(_pieces(raw_pieces))
        return Speech(audio, text=text, words=timed(text, pieces))


def _synthesize_edge(
    text: str, voice: str, rate: str, volume: str, pitch: str
) -> tuple[bytes, list[dict[str, object]]]:
    """Synthesize `text` with `edge_tts` (Python module or CLI) and return `(audio, stamps)`."""
    try:
        edge_tts = importlib.import_module("edge_tts")
    except ModuleNotFoundError:
        edge_tts = None

    if edge_tts is not None:

        async def _run() -> tuple[bytes, list[dict[str, object]]]:
            try:
                communicate = edge_tts.Communicate(
                    text,
                    voice,
                    rate=rate,
                    volume=volume,
                    pitch=pitch,
                    boundary="WordBoundary",
                )
            except TypeError:
                communicate = edge_tts.Communicate(
                    text, voice, rate=rate, volume=volume, pitch=pitch
                )
            chunks: list[bytes] = []
            stamps: list[dict[str, object]] = []
            async for item in communicate.stream():
                kind = item.get("type")
                if kind == "audio" and isinstance(item.get("data"), bytes):
                    chunks.append(item["data"])
                elif kind in ("WordBoundary", "SentenceBoundary"):
                    offset_s = float(item.get("offset", 0)) / 1e7
                    duration_s = float(item.get("duration", 0)) / 1e7
                    stamps.append(
                        {
                            "text": str(item.get("text", "")),
                            "start": offset_s,
                            "end": offset_s + duration_s,
                        }
                    )
            return b"".join(chunks), stamps

        return asyncio.run(_run())

    cli = shutil.which("edge-tts")
    if cli is not None:
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "speech.mp3"
            subs = Path(tmp) / "speech.vtt"
            try:
                subprocess.run(
                    [
                        cli,
                        f"--voice={voice}",
                        f"--rate={rate}",
                        f"--volume={volume}",
                        f"--pitch={pitch}",
                        f"--text={text}",
                        f"--write-media={media}",
                        f"--write-subtitles={subs}",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (OSError, subprocess.CalledProcessError):
                pass
            else:
                stamps = (
                    _parse_vtt(subs.read_text(encoding="utf-8"))
                    if subs.exists()
                    else []
                )
                return media.read_bytes(), stamps

    raise RuntimeError(
        "edge-tts is not installed: run `uv pip install edge-tts` (or `pip install"
        " edge-tts`) to use `m.Edge`, or use `m.SystemVoice()` for offline OS speech."
    )


_VTT_CUE_RE = re.compile(
    r"(\d+):(\d+):(\d+)[.,](\d+)\s*-->\s*(\d+):(\d+):(\d+)[.,](\d+)\s*\n([^\n]+)"
)


def _parse_vtt(vtt: str) -> list[dict[str, object]]:
    """Parse WebVTT cues emitted by the `edge-tts` CLI into timestamp records."""
    stamps: list[dict[str, object]] = []
    for m in _VTT_CUE_RE.finditer(vtt):
        h0, m0, s0, ms0, h1, m1, s1, ms1, cue = m.groups()
        start = (
            int(h0) * 3600 + int(m0) * 60 + int(s0) + int(ms0.ljust(3, "0")[:3]) / 1000
        )
        end = (
            int(h1) * 3600 + int(m1) * 60 + int(s1) + int(ms1.ljust(3, "0")[:3]) / 1000
        )
        for word in cue.strip().split():
            stamps.append({"text": word, "start": start, "end": end})
    return stamps


# ── QwenTTS (Apache-2.0 Open-Source Qwen3-TTS, local or HTTP/OpenAI endpoint) ───────────


class QwenTTSSettings(TypedDict, total=False, closed=True):
    """Settings for Alibaba's Apache-2.0 open-source `Qwen3-TTS` ([`QwenTTS`][manimgx.QwenTTS])."""

    model: str
    """HuggingFace model ID, local MLX directory, or endpoint model name (default
    `"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"`)."""
    language: Literal[
        "Auto",
        "Korean",
        "English",
        "Chinese",
        "Japanese",
        "Spanish",
        "French",
        "German",
        "Italian",
        "Portuguese",
        "Russian",
    ]
    """Target language (default `"Auto"`)."""
    prompt: str | None
    """Optional style or emotion instruction, in words (default: none)."""
    base_url: str | None
    """URL of a local or remote OpenAI-compatible / Qwen3-TTS HTTP server (default:
    `QWEN_TTS_URL` if set, otherwise local MLX / `qwen_tts` package or
    `http://localhost:8000/v1/audio/speech`)."""
    temperature: float
    """Sampling temperature above 0 and up to 1 (default 0.9)."""
    seed: int
    """Deterministic RNG seed for reproducible voice timbre and prosody (default
    `20261017`, the signature `gptssam` / 지피티쌤 Aiden voice seed)."""
    compact: bool
    """Whether to trim leading/trailing dead air and compact internal breath pauses
    while preserving a 25 ms consonant safety margin (default `True`)."""
    speed: float
    """Playback tempo multiplier via `ffmpeg atempo` without pitch shift (default `1.0`)."""


_QWEN_DEFAULT_VOICES: dict[str, str] = {
    "ko": "Aiden",
    "ja": "Ono_Anna",
    "zh": "Vivian",
    "en": "Ryan",
}


_DEFAULT_MLX_MODEL_PATH = Path(
    "/Users/codingo/Documents/ttsMaker/models/qwen3-customvoice-8bit"
)
_DEFAULT_MLX_PYTHON = Path("/Users/codingo/Documents/ttsMaker/.venv/bin/python")


@dataclass(frozen=True, slots=True, init=False, repr=False)
class QwenTTS:
    """Alibaba's Apache-2.0 open-source `Qwen3-TTS` model, as a local or self-hosted voice:
    `QwenTTS(voice, **settings)`.

    Supports Korean (`"Aiden"` — the signature 지피티쌤 clear mid-range male voice — and
    `"Sohee"`), Japanese (`"Ono_Anna"`), Chinese (`"Vivian"`, `"Serena"`, `"Uncle_Fu"`,
    `"Dylan"`, `"Eric"`), and English (`"Ryan"`, `"Aiden"`). When `voice` is `"auto"` (the
    default), selects `"Aiden"` for Korean text. Speaks locally via Apple Silicon MLX
    (`mlx_audio`), the `qwen_tts` Python library, or a local/remote HTTP endpoint
    (`base_url` or `QWEN_TTS_URL`).

    Args:
        voice: Speaker name (such as `"Aiden"` or `"Sohee"` for Korean, or `"Ryan"` for
            English), or `"auto"`.
        key: Optional Bearer API key (`QWEN_TTS_KEY` or `DASHSCOPE_API_KEY` if needed).
        **settings: Model, language, prompt, seed, pause-compaction, speed, and endpoint
            settings ([`QwenTTSSettings`][manimgx.audio.tts.QwenTTSSettings]).

    Examples:
        Speak Korean with the signature 지피티쌤 voice (`"Aiden"`, `seed=20261017`):

        ```py
        voice = m.QwenTTS("Aiden", language="Korean", seed=20261017, speed=1.18)
        ```
    """

    voice: str
    """The speaker's name, or `"auto"` to pick from the text's script."""
    settings: dict[str, object]
    """The model's settings, as given."""
    key: str | None = field(default=None, compare=False)
    """Optional API key, if the endpoint requires one."""

    def __init__(
        self,
        voice: str = "auto",
        *,
        key: str | None = None,
        **settings: Unpack[QwenTTSSettings],
    ) -> None:
        object.__setattr__(self, "voice", voice)
        object.__setattr__(
            self, "settings", {k: _ordered(v) for k, v in sorted(settings.items())}
        )
        object.__setattr__(self, "key", key)

    def __repr__(self) -> str:
        """The call that makes it, without its key: what names it in the cache."""
        given = "".join(f", {k}={v!r}" for k, v in self.settings.items())
        return f"QwenTTS({self.voice!r}{given})"

    def resolve_voice(self, text: str) -> str:
        """Return the concrete speaker name for `text`."""
        if self.voice != "auto":
            return self.voice
        return _QWEN_DEFAULT_VOICES[_detect_script(text)]

    def __call__(self, text: str) -> Speech:
        chosen = self.resolve_voice(text)
        model = str(self.settings.get("model", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"))
        raw_lang = str(self.settings.get("language", "Auto"))
        if raw_lang == "Auto":
            script = _detect_script(text)
            language = {
                "ko": "Korean",
                "ja": "Japanese",
                "zh": "Chinese",
                "en": "English",
            }[script]
        else:
            language = raw_lang
        prompt = self.settings.get("prompt")
        base_url = self.settings.get("base_url") or os.environ.get("QWEN_TTS_URL")
        key = (
            self.key
            or os.environ.get("QWEN_TTS_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
        )
        seed = int(self.settings.get("seed", 20261017))  # ty: ignore[invalid-argument-type]
        compact = bool(self.settings.get("compact", True))
        speed = float(self.settings.get("speed", 1.0))  # ty: ignore[invalid-argument-type]
        audio, raw_pieces = _synthesize_qwen(
            text=text,
            voice=chosen,
            model=model,
            language=language,
            prompt=str(prompt) if prompt is not None else None,
            base_url=str(base_url) if base_url is not None else None,
            key=key,
            temperature=float(self.settings.get("temperature", 0.9)),  # ty: ignore[invalid-argument-type]
            seed=seed,
            compact=compact,
            speed=speed,
        )
        pieces = _expand_cjk_pieces(_pieces(raw_pieces))
        return Speech(audio, text=text, words=timed(text, pieces))


def _synthesize_qwen_mlx(
    *,
    text: str,
    voice: str,
    model: str,
    language: str,
    prompt: str | None,
    temperature: float,
    seed: int,
    compact: bool,
    speed: float,
) -> bytes | None:
    """Synthesize `text` locally via Apple Silicon MLX `Qwen3-TTS` if installed."""
    mlx_model_dir = Path(
        os.environ.get("QWEN_TTS_MLX_MODEL", str(_DEFAULT_MLX_MODEL_PATH))
    )
    if Path(model).exists():
        mlx_model_dir = Path(model)
    if not mlx_model_dir.exists():
        return None

    mlx_python = Path(os.environ.get("QWEN_TTS_PYTHON", str(_DEFAULT_MLX_PYTHON)))
    if not mlx_python.exists():
        mlx_python = Path(sys.executable)

    helper_code = """
import json, sys, numpy as np, soundfile as sf
import mlx.core as mx
from mlx_audio.tts.utils import load_model

cfg = json.loads(sys.argv[1])
model = load_model(cfg["model_dir"])
mx.random.seed(int(cfg["seed"]))
chunks = []
for res in model.generate_custom_voice(
    text=cfg["text"],
    speaker=cfg["voice"],
    language=cfg["language"],
    instruct=cfg["prompt"],
    temperature=float(cfg["temperature"]),
    top_k=50,
    top_p=1.0,
    repetition_penalty=1.05,
    max_tokens=1600,
):
    chunks.append(np.array(res.audio, dtype=np.float32))
audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
sr = int(getattr(model, "sample_rate", 24000))

if cfg["compact"]:
    frame_len = max(1, int(sr * 10.0 / 1000.0))
    n_frames = len(audio) // frame_len
    if n_frames > 0:
        frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
        rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
        db = 20.0 * np.log10(rms)
        is_speech = db > -35.0
        idx = np.where(is_speech)[0]
        if len(idx) > 0:
            margin_f = max(1, int(25.0 / 10.0))
            max_pause_f = max(1, int(55.0 / 10.0))
            f_start = max(0, int(idx[0]) - margin_f)
            f_end = min(n_frames, int(idx[-1]) + 1 + margin_f)
            trimmed_f = is_speech[f_start:f_end]
            out_pieces = []
            i = 0
            nf = len(trimmed_f)
            while i < nf:
                if trimmed_f[i]:
                    j = i
                    while j < nf and trimmed_f[j]:
                        j += 1
                    s0 = (f_start + i) * frame_len
                    s1 = min(len(audio), (f_start + j) * frame_len)
                    out_pieces.append(audio[s0:s1])
                    i = j
                else:
                    j = i
                    while j < nf and not trimmed_f[j]:
                        j += 1
                    run_f = j - i
                    keep_f = min(run_f, max_pause_f + 2 * margin_f)
                    if keep_f > 0:
                        half = keep_f // 2
                        rem = keep_f - half
                        s_left0 = (f_start + i) * frame_len
                        s_left1 = min(len(audio), (f_start + i + half) * frame_len)
                        s_right0 = max(s_left1, (f_start + j - rem) * frame_len)
                        s_right1 = min(len(audio), (f_start + j) * frame_len)
                        out_pieces.append(audio[s_left0:s_left1])
                        if s_right1 > s_right0:
                            out_pieces.append(audio[s_right0:s_right1])
                    i = j
            if out_pieces:
                audio = np.concatenate(out_pieces)
    fade = min(int(sr * 0.012), len(audio) // 4)
    if fade > 0:
        audio[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    cur_rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
    if cur_rms > 1e-5:
        audio = audio * (0.18 / cur_rms)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0.95:
        audio = audio * (0.95 / peak)

sf.write(cfg["out_wav"], audio, sr)
"""
    with tempfile.TemporaryDirectory() as tmp:
        out_wav = Path(tmp) / "qwen_out.wav"
        out_mp3 = Path(tmp) / "qwen_out.mp3"
        cfg = {
            "model_dir": str(mlx_model_dir),
            "text": text,
            "voice": voice,
            "language": language,
            "prompt": prompt,
            "temperature": temperature,
            "seed": seed,
            "compact": compact,
            "out_wav": str(out_wav),
        }
        res = subprocess.run(
            [str(mlx_python), "-c", helper_code, json.dumps(cfg)],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0 or not out_wav.exists():
            return None
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is not None:
            cmd = [ffmpeg, "-y", "-i", str(out_wav)]
            if abs(speed - 1.0) > 1e-3:
                cmd += ["-af", f"atempo={speed:.4f}"]
            cmd += ["-c:a", "libmp3lame", "-q:a", "2", str(out_mp3)]
            mp3_res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if mp3_res.returncode == 0 and out_mp3.exists():
                return out_mp3.read_bytes()
        return out_wav.read_bytes()


def _synthesize_qwen(
    *,
    text: str,
    voice: str,
    model: str,
    language: str,
    prompt: str | None,
    base_url: str | None,
    key: str | None,
    temperature: float,
    seed: int = 20261017,
    compact: bool = True,
    speed: float = 1.0,
) -> tuple[bytes, list[dict[str, object]]]:
    """Synthesize `text` using local MLX `Qwen3-TTS`, `qwen_tts`, or an HTTP endpoint."""
    if base_url is None:
        mlx_bytes = _synthesize_qwen_mlx(
            text=text,
            voice=voice,
            model=model,
            language=language,
            prompt=prompt,
            temperature=temperature,
            seed=seed,
            compact=compact,
            speed=speed,
        )
        if mlx_bytes is not None:
            return mlx_bytes, []

        try:
            import soundfile as sf  # ty: ignore[unresolved-import]
            from qwen_tts import Qwen3TTSModel  # ty: ignore[unresolved-import]
        except ModuleNotFoundError:
            Qwen3TTSModel = None
            sf = None
        if Qwen3TTSModel is not None and sf is not None:
            tts_model = Qwen3TTSModel.from_pretrained(model)
            wavs, sr = tts_model.generate_custom_voice(
                text=text,
                language=language,
                speaker=voice,
                instruct=prompt or "",
                temperature=temperature,
            )
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                sf.write(str(tmp_path), wavs[0], sr)
                return tmp_path.read_bytes(), []
            finally:
                tmp_path.unlink(missing_ok=True)

    url = base_url or "http://localhost:8000/v1/audio/speech"
    payload: dict[str, object] = {
        "model": model,
        "input": text,
        "text": text,
        "voice": voice,
        "language": language,
        "temperature": temperature,
    }
    if prompt is not None:
        payload["prompt"] = prompt
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    raw = _post_tts(url, payload, headers)
    if (
        raw[:4] in (b"RIFF", b"fLaC", b"OggS")
        or raw[:3] == b"ID3"
        or raw[:2] == b"\xff\xfb"
    ):
        return raw, []
    reply = json.loads(raw.decode("utf-8"))
    stamps = reply.get("timestamps")
    stamps_list = stamps if isinstance(stamps, list) else []
    if isinstance(reply.get("audio"), str):
        return base64.b64decode(reply["audio"]), stamps_list
    if isinstance(reply.get("audio"), dict) and "url" in reply["audio"]:
        audio_url = str(reply["audio"]["url"])
        return _post_tts(audio_url, None, None), stamps_list
    raise RuntimeError(f"Unexpected QwenTTS response from {url}")


def _post_tts(
    url: str,
    body: dict[str, object] | None,
    headers: dict[str, str] | None,
) -> bytes:
    """GET `url`, or POST `body` as JSON, and return the response bytes."""
    request = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers=headers or {},
        method="GET" if body is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"{url}: HTTP {error.code}: {detail}") from None
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not connect to QwenTTS endpoint at {url} ({error.reason}). Start"
            " a local Qwen3-TTS server, set QWEN_TTS_URL, or install `qwen-tts`."
        ) from None


# ── SystemVoice (100% offline OS-native TTS) ────────────────────────────────────────────


class SystemVoiceSettings(TypedDict, total=False, closed=True):
    """Settings for the operating system's built-in offline TTS ([`SystemVoice`][manimgx.SystemVoice])."""

    rate: int
    """Speaking rate in words per minute (default 180)."""


_MACOS_DEFAULT_VOICES: dict[str, str] = {
    "ko": "Yuna",
    "ja": "Kyoko",
    "zh": "Tingting",
    "en": "Samantha",
}


@dataclass(frozen=True, slots=True, init=False, repr=False)
class SystemVoice:
    """The operating system's built-in offline text-to-speech engine, as a voice:
    `SystemVoice(voice, **settings)`.

    Works 100% offline with zero API keys or network calls: uses `/usr/bin/say` on macOS
    (defaulting to `"Yuna"` for Korean and `"Samantha"` for English), `espeak-ng` / `espeak`
    on Linux, and `System.Speech.Synthesis` on Windows. Word timings are automatically
    estimated from the synthesized waveform for [`Scene.say`][manimgx.Scene.say].

    Args:
        voice: OS voice name (such as `"Yuna"` or `"Samantha"`), or `"auto"`.
        **settings: Speaking rate settings
            ([`SystemVoiceSettings`][manimgx.audio.tts.SystemVoiceSettings]).

    Examples:
        Offline Korean/English voice with zero dependencies:

        ```py
        voice = m.SystemVoice("Yuna", rate=185)
        ```
    """

    voice: str
    """The OS voice's name, or `"auto"` to pick from the text's script."""
    settings: dict[str, object]
    """The voice's settings, as given."""

    def __init__(
        self,
        voice: str = "auto",
        **settings: Unpack[SystemVoiceSettings],
    ) -> None:
        object.__setattr__(self, "voice", voice)
        object.__setattr__(
            self, "settings", {k: _ordered(v) for k, v in sorted(settings.items())}
        )

    def __repr__(self) -> str:
        """The call that makes it: what names it in the cache."""
        given = "".join(f", {k}={v!r}" for k, v in self.settings.items())
        return f"SystemVoice({self.voice!r}{given})"

    def resolve_voice(self, text: str) -> str:
        """Return the concrete OS voice name for `text`."""
        if self.voice != "auto":
            return self.voice
        script = _detect_script(text)
        if sys.platform == "darwin":
            return _MACOS_DEFAULT_VOICES[script]
        return script

    def __call__(self, text: str) -> Speech:
        chosen = self.resolve_voice(text)
        rate = int(self.settings.get("rate", 180))  # ty: ignore[invalid-argument-type]
        audio = _synthesize_system(text, chosen, rate)
        return Speech(audio, text=text)


def _synthesize_system(text: str, voice: str, rate: int) -> bytes:
    """Synthesize `text` using the OS native offline speech synthesizer and return WAV bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "speech.wav"
        if sys.platform == "darwin" and Path("/usr/bin/say").exists():
            cmd = [
                "/usr/bin/say",
                "-v",
                voice,
                "-r",
                str(rate),
                "-o",
                str(out),
                "--data-format=LEI16@24000",
                "--",
                text,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode != 0:
                # Fallback without -v if the specific voice isn't installed on this Mac
                subprocess.run(
                    [
                        "/usr/bin/say",
                        "-r",
                        str(rate),
                        "-o",
                        str(out),
                        "--data-format=LEI16@24000",
                        "--",
                        text,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            return out.read_bytes()

        for binary in ("espeak-ng", "espeak"):
            exe = shutil.which(binary)
            if exe is not None:
                subprocess.run(
                    [exe, "-v", voice, "-s", str(rate), "-w", str(out), "--", text],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return out.read_bytes()

        if sys.platform == "win32":
            ps = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.SetOutputToWaveFile({json.dumps(str(out))}); "
                f"$s.Speak({json.dumps(text)}); "
                "$s.Dispose()"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                check=True,
                capture_output=True,
                text=True,
            )
            return out.read_bytes()

    raise RuntimeError(
        "No offline system TTS binary (/usr/bin/say or espeak-ng) found."
    )
