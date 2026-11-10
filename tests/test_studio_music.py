import base64

import pytest

import studio_music
import studio_pricing


def test_music_prompt_is_instrumental_and_uses_project_context():
    prompt = studio_music.music_prompt({
        "genre": "mystery",
        "logline": "A detective discovers a hidden family secret.",
        "visual_style": "moody, cinematic",
    })
    assert "instrumental-only" in prompt
    assert "no vocals" in prompt.lower()
    assert "mystery" in prompt
    assert "hidden family secret" in prompt
    assert "Do not imitate" in prompt


def test_audio_parser_reads_documented_output_audio_shape():
    raw = b"ID3" + (b"x" * 2048)
    payload = {"output_audio": {"data": base64.b64encode(raw).decode("ascii")}}
    assert studio_music._audio_from_response(payload) == raw


def test_audio_parser_reads_rest_steps_shape():
    raw = b"ID3" + (b"y" * 2048)
    payload = {"steps": [{"type": "model_output", "content": [
        {"type": "audio", "data": base64.b64encode(raw).decode("ascii")}
    ]}]}
    assert studio_music._audio_from_response(payload) == raw


def test_audio_parser_rejects_missing_audio():
    with pytest.raises(studio_music.MusicGenerationError):
        studio_music._audio_from_response({"steps": []})


def test_render_estimate_includes_music_only_when_no_creator_track():
    without_music = studio_pricing.render_cost(90, music_generated=False)
    with_music = studio_pricing.render_cost(90, music_generated=True)
    assert with_music.total_usd - without_music.total_usd == pytest.approx(0.08)
    assert with_music.usage()["units"]["generated_music"] is True
