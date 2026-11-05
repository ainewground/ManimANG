"""Free, local, and multilingual voices (`Edge`, `QwenTTS`, `SystemVoice`) and Korean
`Scene.say` word-boundary synchronization."""

import base64
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import manimgx as m
from manimgx.audio import tts
from manimgx.audio.sound import RATE, _identity, _wav, spans, times

SECOND_TENTH = RATE // 10


def _fake_wav(seconds: float = 1.0) -> bytes:
    samples = np.full((int(RATE * seconds), 1), 0.1, dtype=np.float32)
    return _wav(samples)


class TestAutoVoiceSelection:
    def test_edge_auto_selects_voice_by_script(self) -> None:
        voice = m.Edge()
        assert voice.resolve_voice("안녕하세요, 원을 그립니다.") == "ko-KR-SunHiNeural"
        assert voice.resolve_voice("こんにちは") == "ja-JP-NanamiNeural"
        assert voice.resolve_voice("你好世界") == "zh-CN-XiaoxiaoNeural"
        assert voice.resolve_voice("Hello world") == "en-US-AriaNeural"
        explicit = m.Edge("ko-KR-InJoonNeural", rate="+10%")
        assert explicit.resolve_voice("Hello world") == "ko-KR-InJoonNeural"
        assert repr(explicit) == "Edge('ko-KR-InJoonNeural', rate='+10%')"

    def test_qwen_tts_auto_selects_sohee_for_korean(self) -> None:
        voice = m.QwenTTS()
        assert voice.resolve_voice("이차함수의 그래프입니다.") == "Sohee"
        assert voice.resolve_voice("こんにちは") == "Ono_Anna"
        assert voice.resolve_voice("你好") == "Vivian"
        assert voice.resolve_voice("Hello") == "Ryan"
        with_key = m.QwenTTS("Sohee", key="secret", language="Korean")
        without_key = m.QwenTTS("Sohee", language="Korean")
        assert _identity(with_key) == _identity(without_key)
        assert repr(with_key) == "QwenTTS('Sohee', language='Korean')"

    def test_system_voice_auto_selects_yuna_on_macos(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tts.sys, "platform", "darwin")
        voice = m.SystemVoice(rate=190)
        assert voice.resolve_voice("안녕하세요") == "Yuna"
        assert voice.resolve_voice("Hello") == "Samantha"
        assert repr(voice) == "SystemVoice('auto', rate=190)"


class TestKoreanSaySync:
    def test_edge_splits_korean_eojeol_boundaries_for_particle_spans(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("manimgx.scene._voice_folder", lambda scene: tmp_path)

        def fake_edge(
            text: str, voice: str, rate: str, volume: str, pitch: str
        ) -> tuple[bytes, list[dict[str, object]]]:
            assert voice == "ko-KR-SunHiNeural"
            stamps: list[dict[str, object]] = [
                {"text": "여기에", "start": 0.0, "end": 0.3},
                {"text": "원을", "start": 0.4, "end": 0.6},
                {"text": "그립니다.", "start": 0.7, "end": 1.1},
            ]
            return _fake_wav(1.2), stamps

        monkeypatch.setattr(tts, "_synthesize_edge", fake_edge)
        script = "여기에 [원]을 그립니다."
        clean_text, (span,) = spans(script)
        speech = m.Edge()(clean_text)
        # "원을" is (0.4, 0.6) across 2 syllables -> "원" is (0.4, 0.5), "을" is (0.5, 0.6)
        assert times(speech, span) == pytest.approx((0.4, 0.5))

        class KoreanScene(m.Scene):
            voice = m.Edge()

            def construct(self) -> None:
                circle = m.Circle()
                self.say("여기에 [원]을 그립니다.", m.Create(circle))

        film = KoreanScene().render()
        assert len(film.clips) == 1
        assert film.clips[0].sound.duration == pytest.approx(1.2)

    def test_vtt_parser_extracts_word_boundaries(self) -> None:
        vtt = (
            "WEBVTT\n\n"
            "00:00:00.100 --> 00:00:00.500\n"
            "여기에 원을\n\n"
            "00:00:00.600 --> 00:00:01.000\n"
            "그립니다\n"
        )
        stamps = tts._parse_vtt(vtt)
        assert stamps == [
            {"text": "여기에", "start": 0.1, "end": 0.5},
            {"text": "원을", "start": 0.1, "end": 0.5},
            {"text": "그립니다", "start": 0.6, "end": 1.0},
        ]


class TestQwenAndSystemProviders:
    def test_qwen_tts_http_json_and_raw_audio(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wav = _fake_wav(0.5)
        encoded = base64.b64encode(wav).decode("ascii")
        reply = {
            "audio": encoded,
            "timestamps": [
                {"text": "안녕", "start": 0.0, "end": 0.2},
                {"text": "하세요", "start": 0.2, "end": 0.5},
            ],
        }
        monkeypatch.setattr(
            tts,
            "_post_tts",
            lambda url, body, headers: json.dumps(reply).encode("utf-8"),
        )
        speech = m.QwenTTS(base_url="http://localhost:8000/tts")("안녕하세요")
        assert speech.duration == pytest.approx(0.5)
        assert [w.text for w in speech.words] == ["안", "녕", "하", "세", "요"]
        assert speech.words[0].start == pytest.approx(0.0)
        assert speech.words[1].end == pytest.approx(0.2)

    def test_system_voice_and_manimang_voice_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("manimgx.scene._voice_folder", lambda scene: tmp_path)
        monkeypatch.setenv("MANIMANG_VOICE", "system")
        if sys.platform != "darwin" or not Path("/usr/bin/say").exists():
            monkeypatch.setattr(
                tts, "_synthesize_system", lambda text, voice, rate: _fake_wav(0.4)
            )

        class OfflineScene(m.Scene):
            def construct(self) -> None:
                dot = m.Dot()
                self.say("안녕 [점]!", m.FadeIn(dot))

        film = OfflineScene().render()
        assert len(film.clips) == 1
        assert film.clips[0].sound.duration > 0

    def test_korean_captions_keep_eojeols_whole(self) -> None:
        from manimgx.audio.sound import lines

        text = (
            "트랜스포머의 토큰 임베딩들이 3차원 구면 다양체 위에 놓여 있고, "
            "자기 어텐션 수식이 토큰 사이의 연결 강도를 결정합니다."
        )
        speech = m.Speech(_fake_wav(4.0), text=text)
        captions = [caption for _, _, caption in lines(speech)]
        assert captions == [
            "트랜스포머의 토큰 임베딩들이 3차원 구면",
            "다양체 위에 놓여 있고, 자기 어텐션 수식이",
            "토큰 사이의 연결 강도를 결정합니다.",
        ]
