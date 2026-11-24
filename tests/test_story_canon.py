"""The story canon (studio_canon.py + creator_studio_canon.sql) on a real Postgres, with a fake AI extractor.
No AI provider is called and nothing is paid."""
import uuid

import pytest

import studio_canon as canon_mod
from conftest import ROOT, connect
from studio_canon import CanonConflict, CanonError, CanonRepo
from studio_store import PgStore


@pytest.fixture(scope="module")
def schema(migrated):
    conn = connect()
    conn.execute((ROOT / "creator_studio_schema.sql").read_text())
    conn.execute((ROOT / "creator_studio_canon.sql").read_text())
    conn.execute((ROOT / "creator_studio_canon.sql").read_text())        # safe to run twice
    conn.close()


@pytest.fixture
def repo(schema, db):
    db.execute("truncate public.studio_stories cascade")
    owner = uuid.uuid4()
    db.execute("insert into auth.users (id) values (%s)", (owner,))
    store = PgStore(db)
    story = canon_mod.create_story(store, str(owner), "The Letter")
    return CanonRepo(store, story["id"], str(owner))


class FakeExtractor:
    """Stands in for the AI that reads a chapter and reports what it establishes. Returns what it was given."""
    def __init__(self, by_chapter: dict):
        self.by_chapter = by_chapter
        self.calls = 0

    def extract(self, chapter: int, passage: str) -> dict:
        self.calls += 1
        return self.by_chapter.get(chapter, {})


CH5 = ("The rain had not stopped for three days. Kalisa stepped onto the bridge at dawn and did not see the truck. "
       "By noon the whole village knew that Kalisa was dead, and Mama Uwase had locked the letter in her drawer.")
CH7 = ("Mama Uwase sat with Amani by the fire and finally told her the truth: Kalisa was your father, child. "
       "Amani said nothing for a long time.")


def _people(repo):
    amani = repo.add_entity("character", "Amani", aliases=["the girl"], by="creator")
    kalisa = repo.add_entity("character", "Kalisa", by="creator")
    uwase = repo.add_entity("character", "Mama Uwase", aliases=["Uwase"], by="creator")
    home = repo.add_entity("place", "the house on the hill", aliases=["home"], by="creator")
    market = repo.add_entity("place", "the market", by="creator")
    church = repo.add_entity("place", "the church", by="creator")
    letter = repo.add_entity("object", "the letter", by="creator")
    secret = repo.add_entity("secret", "Kalisa is Amani's father", profile={"truth": "Kalisa is Amani's real father"},
                             by="creator")
    return {k: v["id"] for k, v in dict(amani=amani, kalisa=kalisa, uwase=uwase, home=home, market=market,
                                         church=church, letter=letter, secret=secret).items()}


# ------------------------------------------------------------------------------------------- the seven required cases

def test_a_character_who_dies_in_chapter_5_cannot_appear_alive_in_chapter_10(repo):
    p = _people(repo)
    repo.creator_fact({"kind": "status", "subject_id": p["kalisa"], "value": "alive", "valid_from_chapter": 1})
    ai = FakeExtractor({5: {"facts": [{"kind": "status", "subject": "Kalisa", "value": "dead", "seq": 2,
                                       "quote": "the whole village knew that Kalisa was dead"}]}})
    got = repo.ingest(ai.extract(5, CH5), CH5, chapter=5, source_ref="chapter-5")
    assert len(got["facts"]) == 1 and not got["rejected"]
    death = got["facts"][0]
    assert death["status"] == "proposed"
    # Not canon until the creator approves it.
    assert repo.load().state_at(10).status_of(p["kalisa"]) == "alive"
    repo.approve(death["id"])
    c = repo.load()
    assert c.state_at(4).status_of(p["kalisa"]) == "alive"
    assert c.state_at(10).status_of(p["kalisa"]) == "dead"
    rep = c.check([{"type": "speaks", "subject": "Kalisa", "chapter": 10, "quote": "Kalisa laughed."}])
    assert [i["rule"] for i in rep["issues"]] == ["dead_character_active"]
    issue = rep["issues"][0]
    assert "dead from chapter 5" in issue["reason"]
    assert issue["evidence"]["source_quote"] == "the whole village knew that Kalisa was dead"
    assert issue["evidence"]["source_chapter"] == 5
    # An AI suggestion that brings him back later is flagged, not accepted silently.
    row, conflicts = repo.propose_fact({"kind": "status", "subject_id": p["kalisa"], "value": "alive",
                                        "valid_from_chapter": 10, "source_quote": "Kalisa walked into the market"},
                                       passage="At noon Kalisa walked into the market as if nothing had happened.")
    assert row["status"] == "proposed"
    assert [x["reason"] for x in conflicts] == ["returns from the dead"]


def test_a_character_cannot_know_a_secret_before_learning_it(repo):
    p = _people(repo)
    ai = FakeExtractor({7: {"facts": [{"kind": "knows", "subject": "Amani", "object": "Kalisa is Amani's father",
                                       "quote": "finally told her the truth: Kalisa was your father"}]}})
    got = repo.ingest(ai.extract(7, CH7), CH7, chapter=7)
    repo.approve(got["facts"][0]["id"])
    c = repo.load()
    early = c.check([{"type": "knows", "subject": "Amani", "object": "Kalisa is Amani's father", "chapter": 4,
                      "quote": "Amani glared at her father's grave."}])
    assert [i["rule"] for i in early["issues"]] == ["knowledge_leak"]
    assert "learns it in chapter 7" in early["issues"][0]["reason"]
    late = c.check([{"type": "knows", "subject": "Amani", "object": "Kalisa is Amani's father", "chapter": 8}])
    assert late["issues"] == []
    # Someone who never learns it in the canon.
    never = c.check([{"type": "knows", "subject": "the girl", "object": "Kalisa is Amani's father", "chapter": 3}])
    assert "never learns it" not in never["issues"][0]["reason"]          # Amani (alias) does learn it, in chapter 7
    nobody = c.check([{"type": "knows", "subject": "Kalisa", "object": "Kalisa is Amani's father", "chapter": 3}])
    assert "never learns it in the canon" in nobody["issues"][0]["reason"]
    assert c.state_at(8).knows(p["amani"], p["secret"]) and not c.state_at(6).knows(p["amani"], p["secret"])


def test_an_object_cannot_be_in_two_places_at_the_same_time(repo):
    p = _people(repo)
    repo.creator_fact({"kind": "location", "subject_id": p["letter"], "object_id": p["home"], "valid_from_chapter": 2})
    c = repo.load()
    # It turns up somewhere else with no move in between.
    rep = c.check([{"type": "located", "subject": "the letter", "object": "the market", "chapter": 6}])
    assert [i["rule"] for i in rep["issues"]] == ["object_location_conflict"]
    # The same passage moves it first (a pending fact at seq 3), then shows it there: fine.
    move = {"kind": "location", "subject_id": p["letter"], "object_id": p["market"], "valid_from_chapter": 6,
            "valid_from_seq": 3}
    ok = c.check([{"type": "located", "subject": "the letter", "object": "the market", "chapter": 6, "seq": 4}],
                 pending=[move])
    assert ok["issues"] == []
    # Two places at the same moment in one passage.
    two = c.check([{"type": "located", "subject": "the letter", "object": "home", "chapter": 3, "seq": 1},
                   {"type": "located", "subject": "the letter", "object": "the church", "chapter": 3, "seq": 1}])
    assert "object_two_places" in [i["rule"] for i in two["issues"]]
    # Two approved facts that put it in two places from the same moment are caught by the audit.
    repo.creator_fact({"kind": "location", "subject_id": p["letter"], "object_id": p["church"], "valid_from_chapter": 2})
    assert "object_two_places" in [a["rule"] for a in repo.load().audit()]


def test_location_and_relationships_change_correctly_over_time(repo):
    p = _people(repo)
    for ch, place in ((1, "home"), (4, "market"), (9, "church")):
        repo.creator_fact({"kind": "location", "subject_id": p["amani"], "object_id": p[place], "valid_from_chapter": ch})
    for ch, value in ((1, "strangers"), (6, "engaged"), (12, "estranged")):
        repo.creator_fact({"kind": "relationship", "subject_id": p["amani"], "object_id": p["kalisa"],
                           "attribute": "romance", "value": value, "valid_from_chapter": ch})
    repo.creator_fact({"kind": "relationship", "subject_id": p["amani"], "object_id": p["uwase"], "attribute": "work",
                       "value": "business partners", "valid_from_chapter": 2, "valid_until_chapter": 8})
    c = repo.load()
    where = {ch: c.state_at(ch).location_of(p["amani"]) for ch in (1, 3, 4, 8, 9, 20)}
    assert where == {1: p["home"], 3: p["home"], 4: p["market"], 8: p["market"], 9: p["church"], 20: p["church"]}

    def romance(ch):
        return [f["value"] for f in c.state_at(ch).relationships_of(p["amani"]) if f["attribute"] == "romance"]
    assert romance(3) == ["strangers"] and romance(7) == ["engaged"] and romance(15) == ["estranged"]

    def work(ch):
        return [f["value"] for f in c.state_at(ch).relationships_of(p["amani"]) if f["attribute"] == "work"]
    assert work(1) == [] and work(5) == ["business partners"] and work(8) == []     # ended at chapter 8


def test_a_fact_from_chapter_3_is_still_retrievable_in_episode_30(repo):
    p = _people(repo)
    repo.creator_fact({"kind": "trait", "subject_id": p["amani"], "attribute": "scar",
                       "value": "a burn scar on her left hand", "valid_from_chapter": 3,
                       "statement": "Amani has a burn scar on her left hand."})
    # A long story: 60 chapters, with many facts about other people in between.
    for ch in range(4, 61):
        repo.creator_fact({"kind": "location", "subject_id": p["uwase"],
                           "object_id": p["market"] if ch % 2 else p["home"], "valid_from_chapter": ch})
        repo.creator_fact({"kind": "trait", "subject_id": p["kalisa"], "attribute": f"habit {ch}", "value": "x",
                           "valid_from_chapter": ch})
    for n in range(1, 31):
        repo.set_episode(n, 2 * n - 1, 2 * n)
    c = repo.load()
    task = {"at": {"episode": 30}, "text": "Amani washes her hands at the river before the wedding.", "limit": 20}
    got = c.retrieve(task)
    assert got["point"]["chapter"] == 59
    assert "Amani" in got["entities"]
    scar = [f for f in got["facts"] if "burn scar" in f["statement"]]
    assert scar and scar[0]["source_chapter"] is None and scar[0]["valid_from"]["chapter"] == 3
    assert "burn scar on her left hand" in c.render(task)
    # Only what's relevant: a scene without Amani doesn't get her scar, and Kalisa's 57 habits aren't dumped in.
    other = c.retrieve({"at": {"episode": 30}, "text": "Mama Uwase counts coins at the market.", "limit": 20})
    assert not any("burn scar" in f["statement"] for f in other["facts"])
    assert not any("habit" in f["statement"] for f in got["facts"])


def test_creator_locked_facts_override_conflicting_ai_suggestions(repo):
    p = _people(repo)
    locked = repo.creator_fact({"kind": "relationship", "subject_id": p["uwase"], "object_id": p["kalisa"],
                                "attribute": "family", "value": "sister", "valid_from_chapter": 0}, lock=True)
    passage = "Everyone in the village said Mama Uwase was only Kalisa's cousin, nothing more."
    ai = FakeExtractor({4: {"facts": [{"kind": "relationship", "subject": "Mama Uwase", "object": "Kalisa",
                                       "object_kind": "character", "attribute": "family", "value": "cousin",
                                       "quote": "Mama Uwase was only Kalisa's cousin"}]}})
    got = repo.ingest(ai.extract(4, passage), passage, chapter=4)
    suggestion = got["facts"][0]
    assert suggestion["status"] == "proposed"
    assert got["conflicts"] and got["conflicts"][0]["with"][0]["blocking"] is True
    with pytest.raises(CanonConflict) as e:
        repo.approve(suggestion["id"])
    assert e.value.conflicts[0]["fact"]["id"] == locked["id"]

    def family(c, ch, **kw):
        return [f["value"] for f in c.state_at(ch, **kw).relationships_of(p["uwase"]) if f["attribute"] == "family"]
    c = repo.load()
    assert family(c, 10) == ["sister"]
    # Even shown alongside unreviewed suggestions (review screens), the locked fact still wins.
    assert family(c, 10, include_proposed=True) == ["sister"]
    # The AI can't approve or lock anything itself.
    with pytest.raises(CanonError):
        repo.approve(suggestion["id"], by="ai")
    with pytest.raises(CanonError):
        repo.lock(suggestion["id"], by="ai")
    # A locked fact can't be retired by accident.
    with pytest.raises(CanonError):
        repo.retire(locked["id"], "changed my mind")
    assert repo.retire(locked["id"], "changed my mind", force_locked=True)["status"] == "retired"


def test_flashbacks_and_intentional_exceptions_are_not_false_contradictions(repo):
    p = _people(repo)
    repo.creator_fact({"kind": "status", "subject_id": p["kalisa"], "value": "alive", "valid_from_chapter": 0})
    repo.creator_fact({"kind": "status", "subject_id": p["kalisa"], "value": "dead", "valid_from_chapter": 5,
                       "source_quote": "the whole village knew that Kalisa was dead"})
    c = repo.load()
    # A flashback read in chapter 10 shows chapter 3, when he was alive.
    back = c.check([{"type": "speaks", "subject": "Kalisa", "chapter": 10, "mode": "flashback", "as_of": 3}])
    assert back["issues"] == []
    # A "flashback" that shows a time after his death is still wrong.
    wrong = c.check([{"type": "speaks", "subject": "Kalisa", "chapter": 10, "mode": "flashback", "as_of": 6}])
    assert [i["rule"] for i in wrong["issues"]] == ["dead_character_active"]
    # Dreams, visions, lies and rumours aren't checked as physical fact; they're listed as skipped, with why.
    dream = c.check([{"type": "appears", "subject": "Kalisa", "chapter": 10, "mode": "dream"},
                     {"type": "speaks", "subject": "Kalisa", "chapter": 10, "mode": "vision"}])
    assert dream["issues"] == [] and len(dream["skipped"]) == 2
    # From chapter 11 his ghost appears on purpose: the creator records an exception for that stretch.
    repo.add_exception("dead_character_active", "Kalisa's ghost visits Amani", subject_id=p["kalisa"],
                       from_chapter=11, until_chapter=14)
    c = repo.load()
    ghost = c.check([{"type": "appears", "subject": "Kalisa", "chapter": 12}])
    assert ghost["issues"] == [] and ghost["excepted"][0]["exception"] == "Kalisa's ghost visits Amani"
    outside = c.check([{"type": "appears", "subject": "Kalisa", "chapter": 10},
                       {"type": "appears", "subject": "Kalisa", "chapter": 15}])
    assert len(outside["issues"]) == 2                              # the exception covers chapters 11-14 only
    other = c.check([{"type": "appears", "subject": "Kalisa", "chapter": 12, "mode": "present"}])
    assert other["excepted"]                                        # same subject and rule, inside the range
    with pytest.raises(CanonError):
        repo.add_exception("dead_character_active", "x", subject_id=p["kalisa"], by="ai")


# ------------------------------------------------------------------------------------------- evidence and trust

def test_every_ai_fact_keeps_its_evidence_and_starts_as_a_proposal(repo):
    p = _people(repo)
    passage = "Amani hid the letter under the floorboards of the house on the hill."
    got = repo.ingest({"facts": [
        {"kind": "location", "subject": "the letter", "subject_kind": "object", "object": "the house on the hill",
         "quote": "hid the letter under the floorboards"},
        {"kind": "trait", "subject": "Amani", "attribute": "fear", "value": "dogs", "quote": "Amani was afraid of dogs"},
        {"kind": "trait", "subject": "Amani", "attribute": "fear", "value": "the dark", "quote": ""},
        {"kind": "status", "subject": "Gakwaya", "value": "alive", "quote": "Gakwaya waved from the road"},
    ]}, passage, chapter=3, source_ref="chapter-3/scene-2")
    assert len(got["facts"]) == 1
    kept = got["facts"][0]
    assert kept["status"] == "proposed" and kept["origin"] == "ai"
    assert kept["source_quote"] == "hid the letter under the floorboards" and kept["source_chapter"] == 3
    assert kept["source_ref"] == "chapter-3/scene-2"
    reasons = [r["reason"] for r in got["rejected"]]
    assert "the quote isn't in the passage" in reasons                      # invented quote
    assert any("every AI fact needs a quote" in r for r in reasons)          # no quote at all
    assert any(r["what"] == "new character Gakwaya" for r in got["rejected"])   # an unknown name without evidence
    assert repo.load().entity("Gakwaya") is None
    # Proposed facts are invisible to writing.
    c = repo.load()
    assert c.state_at(5).location_of(p["letter"]) is None
    assert not any("floorboards" in f["source_quote"] for f in c.retrieve({"at": 5, "entities": ["the letter"]})["facts"])


def test_the_database_refuses_canon_without_a_decision_or_evidence(repo):
    p = _people(repo)
    db = repo.store.conn
    base = {"story_id": repo.story_id, "owner_id": repo.owner_id, "kind": "trait", "subject_id": p["amani"],
            "attribute": "age", "value": "19"}
    with pytest.raises(Exception, match="check"):
        repo.store.insert("studio_story_facts", [{**base, "status": "approved", "source_quote": "she was nineteen"}])
    with pytest.raises(Exception, match="check"):
        repo.store.insert("studio_story_facts", [{**base, "origin": "ai", "source_quote": ""}])
    with pytest.raises(Exception, match="check"):
        repo.store.insert("studio_story_facts", [{**base, "source_quote": "q q q", "valid_from_chapter": 5,
                                                  "valid_until_chapter": 5}])
    rls = db.execute("select bool_and(relrowsecurity) from pg_class where relkind = 'r' and relname like 'studio_stor%'").fetchone()[0]
    assert rls is True


def test_new_entities_from_ai_are_proposals_until_a_fact_about_them_is_approved(repo):
    passage = "Gakwaya, the old fisherman, waved from the road and shouted that the boats were back."
    got = repo.ingest({"facts": [{"kind": "status", "subject": "Gakwaya", "value": "alive",
                                  "quote": "Gakwaya, the old fisherman, waved from the road"}]}, passage, chapter=2)
    e = repo.load().entity("Gakwaya")
    assert e["status"] == "proposed" and e["origin"] == "ai" and e["source_quote"]
    repo.approve(got["facts"][0]["id"])
    assert repo.load().entity("Gakwaya")["status"] == "approved"


def test_audit_finds_a_cause_after_its_effect_and_overdue_threads(repo):
    p = _people(repo)
    funeral = repo.add_event(6, "Kalisa's funeral", story_order=6, by="creator")
    repo.add_event(8, "The truck driver confesses he was paid", story_order=9, causes=[], by="creator")
    crash = repo.add_event(5, "Kalisa is killed on the bridge", story_order=7, by="creator")
    repo.store.update("studio_story_events", {"id": funeral["id"]}, {"causes": [crash["id"]]})
    t = repo.add_thread("mystery", "Who paid the truck driver?", 5, entities=[p["kalisa"]], due_chapter=20, by="creator")
    found = repo.load().audit(current_chapter=25)
    rules = [a["rule"] for a in found]
    assert "cause_after_effect" in rules and "overdue_thread" in rules
    repo.resolve_thread(t["id"], 22, "the driver confessed")
    assert "overdue_thread" not in [a["rule"] for a in repo.load().audit(current_chapter=25)]
    # A resolved thread is open before it pays off and closed after.
    c = repo.load()
    assert [x["description"] for x in c.state_at(10).threads] == ["Who paid the truck driver?"]
    assert c.state_at(23).threads == []


def test_retrieval_puts_locked_facts_first_and_includes_secrets_in_play(repo):
    p = _people(repo)
    repo.creator_fact({"kind": "trait", "subject_id": p["amani"], "attribute": "age", "value": "19",
                       "valid_from_chapter": 1})
    repo.creator_fact({"kind": "trait", "subject_id": p["amani"], "attribute": "name meaning", "value": "peace",
                       "valid_from_chapter": 1}, lock=True)
    repo.creator_fact({"kind": "knows", "subject_id": p["uwase"], "object_id": p["secret"], "valid_from_chapter": 1})
    repo.creator_fact({"kind": "rule", "value": "Nobody in the village owns a phone (1994)."})
    got = repo.load().retrieve({"at": 4, "entities": ["Amani", "Mama Uwase"]})
    assert got["facts"][0]["status"] == "creator_locked"
    assert any("phone" in f["statement"] for f in got["facts"])           # a story-wide rule is always included
    s = got["secrets"][0]
    assert s["known_by"] == ["Mama Uwase"] and s["kept_from"] == ["Amani"]
    assert s["truth"] == "Kalisa is Amani's real father"


def test_only_the_creator_can_decide():
    for who in ("ai", "assistant", "", "system"):
        with pytest.raises(CanonError):
            CanonRepo._decision(who)
    assert CanonRepo._decision("creator")["decided_by"] == "creator"
