"""Who Viyo puts in a scene's picture and video."""
from studio_worker import characters_in_scene

CAST = [{"name": "Maya Lin"}, {"name": "Julian Vance"}, {"name": "Ama"}]


def names(scene):
    return [c["name"] for c in characters_in_scene(scene, CAST)]


def test_first_names_in_the_scene_text_count():
    s = {"heading": "INT. CAR - NIGHT", "visual": "Maya grabs Julian's arm as Ama watches", "dialogue": []}
    assert names(s) == ["Maya Lin", "Julian Vance", "Ama"]


def test_a_last_name_in_a_place_name_does_not_add_the_character():
    s = {"heading": "EXT. VANCE MANOR - GATES", "visual": "Maya alone in the rain", "dialogue": []}
    assert names(s) == ["Maya Lin"]


def test_speakers_count_and_full_names_still_work():
    s = {"heading": "", "visual": "Rain on the gates",
         "dialogue": [{"character": "JULIAN (V.O.)", "text": "Run."}, {"character": "Maya Lin", "text": "Julian!"}]}
    assert names(s) == ["Maya Lin", "Julian Vance"]


def test_a_shared_first_name_needs_the_full_name():
    cast = [{"name": "Sam Cole"}, {"name": "Sam Reed"}]
    s = {"heading": "", "visual": "Sam waits. Sam Reed arrives.", "dialogue": []}
    assert [c["name"] for c in characters_in_scene(s, cast)] == ["Sam Reed"]


def test_the_scene_writer_is_told_to_show_people_together():
    import studio_ai
    p = studio_ai.breakdown_prompt({"kind": "episode", "aspect_ratio": "9:16"}, "INT. ROOM")
    assert "TOGETHER in one shot" in p and "heard, not in frame" in p
