"""Google's refusals are shown in plain words, not raw error text."""
import studio_ai

FILTERED = ("Error code: 400 - {'error': {'message': \"Unable to show the generated video. The video was filtered out "
            "because it violated Google's [Generative AI Prohibited Use policy](https://policies.google.com/...)\"}}")
LIKENESS = ("Error code: 400 - {'error': {'message': \"Input blocked: Sorry, we can't create videos with real people's "
            "names or likenesses. Please remove the reference\", 'code': 'content_blocked'}}")


def test_filtered_videos_explain_what_to_change():
    e = studio_ai._classify(Exception(FILTERED))
    assert isinstance(e, studio_ai.PolicyBlocked)
    assert "less graphically" in str(e) and "Error code" not in str(e) and "http" not in str(e)


def test_real_person_blocks_suggest_a_new_name():
    e = studio_ai._classify(Exception(LIKENESS))
    assert isinstance(e, studio_ai.PolicyBlocked) and "different name" in str(e)


def test_busy_is_still_busy():
    assert isinstance(studio_ai._classify(Exception("429 RESOURCE_EXHAUSTED")), studio_ai.ProviderBusy)
