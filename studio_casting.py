"""Automatic voice casting for Viyo Studio.

Works out who a character is (age group, gender presentation, personality)
from the breakdown and the character description, then picks a voice from
the verified voice catalog. Pure and deterministic: the same character in
the same situation always gets the same voice, and nothing here calls an
AI provider or costs money.

Rules that matter:
- A child or teenager is only ever given a voice that has been verified for
  that age group. If none exists, the character is marked `needs_voice` and
  its lines aren't voiced: an adult voice is never used quietly instead.
- Age is only taken as a number when the story states it. Otherwise it is a
  group with a confidence; a name alone never decides it.
- Casting happens once. Later episodes, retries and regenerations reuse the
  saved voice; only a deliberate change replaces it (and is logged).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

AGE_CATEGORIES = ["infant", "young_child", "older_child", "young_teen", "older_teen",
                  "young_adult", "middle_aged", "elderly"]
# Groups that must never be given a voice verified only for another group.
STRICT = {"infant", "young_child", "older_child", "young_teen", "older_teen"}
ADULT = ["young_adult", "middle_aged"]
AGE_LABEL = {"infant": "baby", "young_child": "young child", "older_child": "child", "young_teen": "young teenager",
             "older_teen": "teenager", "young_adult": "young adult", "middle_aged": "adult", "elderly": "older adult",
             "unknown": "adult (age not stated)"}
_CATEGORY_GROUP = {"infant": "child", "young_child": "child", "older_child": "child", "young_teen": "teen",
                   "older_teen": "teen", "young_adult": "adult", "middle_aged": "adult", "elderly": "elderly",
                   "unknown": "adult"}

NARRATOR_VOICE = "Charon"


# ---------------------------------------------------------------- languages

# Gemini 3.8 Flash TTS languages, from Google's speech-generation docs (6 Nov 2026).
GEMINI_TTS_LANGUAGES = [
    "Acehnese", "Afrikaans", "Akan", "Amharic", "Armenian", "Assamese", "Awadhi", "Balinese", "Bangla", "Banjar",
    "Bashkir", "Basque", "Belarusian", "Bemba", "Bhojpuri", "Bosnian", "Buginese", "Bulgarian", "Burmese",
    "Cantonese", "Catalan", "Cebuano", "Central Kurdish", "Chhattisgarhi", "Chinese", "Crimean Tatar", "Croatian",
    "Czech", "Danish", "Dutch", "Dyula", "Dzongkha", "Egyptian Arabic", "English", "Estonian", "Filipino", "Finnish",
    "French", "Galician", "Ganda", "Georgian", "German", "Greek", "Guarani", "Gujarati", "Haitian Creole",
    "Halh Mongolian", "Hausa", "Hebrew", "Hindi", "Hungarian", "Icelandic", "Igbo", "Iloko", "Indonesian",
    "Iranian Persian", "Italian", "Japanese", "Javanese", "Kabyle", "Kamba", "Kannada", "Kashmiri", "Kazakh", "Khmer",
    "Kikuyu", "Kinyarwanda", "Kongo", "Korean", "Kyrgyz", "Lao", "Latgalian", "Lingala", "Lithuanian",
    "Luxembourgish", "Macedonian", "Magahi", "Maithili", "Malayalam", "Maltese", "Manipuri", "Marathi",
    "Minangkabau", "Mizo", "Nepali", "Nigerian Fulfulde", "North Azerbaijani", "Northern Sotho", "Northern Uzbek",
    "Norwegian Bokmal", "Norwegian Nynorsk", "Nyanja", "Occitan", "Odia", "Pangasinan", "Persian (Afghanistan)",
    "Polish", "Portuguese", "Punjabi", "Romanian", "Russian", "Santali", "Serbian", "Sindhi", "Sinhala", "Slovak",
    "Slovenian", "Somali", "South Azerbaijani", "Southern Pashto", "Southern Sotho", "Spanish", "Standard Arabic",
    "Standard Latvian", "Standard Malay", "Swahili", "Swati", "Swedish", "Tajik", "Tamil", "Telugu", "Thai",
    "Tigrinya", "Tosk Albanian", "Turkish", "Uyghur", "Vietnamese",
]
_ALIASES = {
    "arabic": "standard arabic", "mandarin": "chinese", "chinese simplified": "chinese",
    "chinese traditional": "chinese", "farsi": "iranian persian", "persian": "iranian persian",
    "dari": "persian (afghanistan)", "tagalog": "filipino", "malay": "standard malay", "bahasa": "indonesian",
    "bengali": "bangla", "norwegian": "norwegian bokmal", "latvian": "standard latvian", "albanian": "tosk albanian",
    "pashto": "southern pashto", "kurdish": "central kurdish", "sorani": "central kurdish",
    "azerbaijani": "north azerbaijani", "azeri": "north azerbaijani", "uzbek": "northern uzbek",
    "mongolian": "halh mongolian", "luganda": "ganda", "chichewa": "nyanja", "fula": "nigerian fulfulde",
    "fulani": "nigerian fulfulde", "kiswahili": "swahili", "ikinyarwanda": "kinyarwanda", "oriya": "odia",
    "haitian": "haitian creole", "creole": "haitian creole", "sesotho": "southern sotho", "pedi": "northern sotho",
    "siswati": "swati", "twi": "akan", "francais": "french", "espanol": "spanish", "portugues": "portuguese",
    "deutsch": "german",
}


def _norm_lang(name: str) -> str:
    s = (name or "").lower().replace("å", "a").replace("ç", "c").replace("ñ", "n").replace("ê", "e")
    s = re.sub(r"\(.*?\)", "", s) if "afghanistan" not in s else s
    s = re.sub(r"[^a-z() ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_SUPPORTED = {_norm_lang(x) for x in GEMINI_TTS_LANGUAGES}


def language_supported(provider: str, language: str) -> tuple[bool, str]:
    """(supported, the provider's name for it). Empty means English."""
    want = _norm_lang(language or "English") or "english"
    if provider != "google":
        return False, language
    key = want if want in _SUPPORTED else _ALIASES.get(want, "")
    if key in _SUPPORTED:
        canonical = next(x for x in GEMINI_TTS_LANGUAGES if _norm_lang(x) == key)
        return True, canonical
    return False, language


# ---------------------------------------------------------------- age and gender

_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
             "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20}


def category_for_years(years: int) -> str:
    if years < 2:
        return "infant"
    if years <= 7:
        return "young_child"
    if years <= 11:
        return "older_child"
    if years <= 14:
        return "young_teen"
    if years <= 17:
        return "older_teen"
    if years <= 34:
        return "young_adult"
    if years <= 59:
        return "middle_aged"
    return "elderly"


_CUES = [  # (pattern, category) checked in order; first hit wins
    (r"\b(baby|infant|newborn|toddler)\b", "infant"),
    (r"\b(little (girl|boy|kid|one)|small (girl|boy|child)|preschool(er)?|kindergarten|young child)\b", "young_child"),
    (r"\b(schoolgirl|schoolboy|grade[- ]schooler|child|kid)\b", "older_child"),
    (r"\b(pre-?teen|tween|early teens?)\b", "young_teen"),
    (r"\b(teen(age|ager)?s?|high[- ]school(er)?|adolescent)\b", "older_teen"),
    (r"\b(elderly|old (man|woman|lady)|grand(mother|father|ma|pa)|granny|retired|aged|senior citizen|in (his|her|their) (60|70|80|90)s)\b", "elderly"),
    (r"\b(middle[- ]aged|in (his|her|their) (30|40|50)s|(30|40|50)s)\b", "middle_aged"),
    (r"\b(young (woman|man|adult|lady)|in (his|her|their) 20s|20s|college student|university student)\b", "young_adult"),
]


# Words that say the character is an adult even when a child is mentioned
# ("mother of a 7-year-old", "her son").
_ADULT_ROLE = re.compile(r"\b(mother|father|mom|mum|dad|parent|teacher|nurse|nanny|wife|husband|boss|officer|doctor|"
                         r"widow|widower|aunt|uncle|grand(mother|father|ma|pa)|ceo|manager|detective|soldier|"
                         r"professor|pastor|priest)\b")


def infer_age(text: str) -> tuple[Optional[str], Optional[int], str]:
    """(category, stated years, confidence) from free text. None when the
    text doesn't say; never guesses from a name."""
    t = (text or "").lower()
    adult_role = bool(_ADULT_ROLE.search(t))
    found = [int(m.group(1)) for m in re.finditer(r"\b(\d{1,3})[- ]?(?:years?|yrs?)[- ]old\b|\b(?:aged?) (\d{1,3})\b", t)
             if m.group(1)] + [int(m.group(1)) for m in re.finditer(r"\b(?:aged?) (\d{1,3})\b|\b(\d{1,3}) ?yo\b", t)
                               if m.group(1)]
    found += [_WORD_NUM[m.group(1)] for m in re.finditer(r"\b(" + "|".join(_WORD_NUM) + r")[- ]years?[- ]old\b", t)]
    found += [int(m.group(1)) for m in re.finditer(r"\b(?:man|woman|boy|girl|guy|lady|male|female)[,:]? (\d{1,3})\b", t)]
    for years in found:
        if 0 <= years <= 110 and not (adult_role and years < 18):
            return category_for_years(years), years, "high"
    for pattern, category in _CUES:
        if category in STRICT and adult_role:
            continue        # a child is mentioned, but this character is the adult
        if re.search(pattern, t):
            return category, None, "medium"
    return None, None, "low"


def infer_gender(text: str) -> str:
    t = f" {(text or '').lower()} "
    female = len(re.findall(r"\b(she|her|woman|women|girl|lady|mother|mom|mum|daughter|sister|wife|aunt|grandmother|queen|princess|female)\b", t))
    male = len(re.findall(r"\b(he|him|his|man|men|boy|gentleman|father|dad|son|brother|husband|uncle|grandfather|king|prince|male)\b", t))
    if female > male:
        return "female"
    if male > female:
        return "male"
    return "unspecified"


@dataclass
class Profile:
    name: str
    age_category: str = "unknown"
    age_years: Optional[int] = None
    age_confidence: str = "low"
    gender: str = "unspecified"
    personality: str = ""
    language: str = "English"
    role: str = "supporting"            # lead / supporting / minor


def build_profile(name: str, ai: Optional[dict], description: str, language: str) -> Profile:
    """The AI's answer first, checked against what the text actually says.
    A stated age in the text always wins over a guessed group."""
    ai = ai or {}
    text = " ".join(x for x in (description, ai.get("description") or "", ai.get("personality") or "") if x)
    t_cat, t_years, t_conf = infer_age(text)
    a_years = ai.get("age_years")
    try:
        a_years = int(a_years) if a_years not in (None, "") else None
    except (TypeError, ValueError):
        a_years = None
    a_cat = ai.get("age_category") if ai.get("age_category") in AGE_CATEGORIES else None
    if t_years is not None:
        cat, years, conf = t_cat, t_years, "high"
    elif a_years is not None and 0 <= a_years <= 110:
        cat, years, conf = category_for_years(a_years), a_years, (ai.get("age_confidence") or "medium")
    elif a_cat:
        cat, years, conf = a_cat, None, (ai.get("age_confidence") or "medium")
        if t_cat and t_cat != a_cat and (t_cat in STRICT) != (a_cat in STRICT):
            # The AI and the text disagree about whether this is a child: be
            # careful and trust the text, with low confidence.
            cat, conf = t_cat, "low"
    elif t_cat:
        cat, years, conf = t_cat, None, t_conf
    else:
        cat, years, conf = "unknown", None, "low"
    if conf not in ("high", "medium", "low"):
        conf = "medium"
    g = (ai.get("gender") or "").lower()
    gender = g if g in ("female", "male") else infer_gender(text)
    role = ai.get("role") if ai.get("role") in ("lead", "supporting", "minor") else "supporting"
    return Profile(name=name, age_category=cat, age_years=years, age_confidence=conf, gender=gender,
                   personality=(ai.get("personality") or "")[:300], language=language or "English", role=role)


# ---------------------------------------------------------------- casting

@dataclass
class CatalogVoice:
    provider: str
    voice_id: str
    gender: str
    descriptor: str
    age_categories: list = field(default_factory=list)
    languages: list = field(default_factory=lambda: ["all"])
    status: str = "available"


def catalog_from_rows(rows: Iterable[dict]) -> list[CatalogVoice]:
    return [CatalogVoice(r["provider"], r["voice_id"], r.get("gender") or "unspecified", r.get("descriptor") or "",
                         list(r.get("age_categories") or []), list(r.get("languages") or ["all"]),
                         r.get("status") or "available") for r in rows]


def builtin_catalog() -> list[CatalogVoice]:
    """Used only before creator_studio_casting.sql has been run: Google's 30
    voices, adult groups only (what Google documents)."""
    import studio_ai
    return [CatalogVoice("google", n, g, s, list(ADULT)) for n, g, s in studio_ai.VOICES]


_TRAITS = [  # personality words -> provider descriptors that suit them
    (r"warm|kind|loving|caring|motherly|fatherly|sweet", ["Warm", "Gentle", "Friendly", "Soft"]),
    (r"serious|stern|strict|confident|protective|authoritative|commanding|determined|strong|leader", ["Firm", "Clear", "Even", "Informative"]),
    (r"energetic|cheerful|bubbly|playful|excit|lively|happy|optimistic|curious|funny", ["Upbeat", "Lively", "Bright", "Excitable", "Youthful"]),
    (r"calm|quiet|shy|soft-spoken|gentle|timid|thoughtful|sad|melanchol", ["Soft", "Gentle", "Breezy", "Easy-going", "Smooth"]),
    (r"mysterious|seductive|charming|smooth|elegant|cunning|manipulative", ["Smooth", "Breathy"]),
    (r"rough|tough|gruff|grumpy|harsh|menacing|villain|cruel", ["Gravelly", "Firm"]),
    (r"wise|mature|experienced|elder|old", ["Mature", "Knowledgeable", "Even", "Gravelly"]),
    (r"casual|laid-back|relaxed|easygoing|easy-going|chill", ["Casual", "Easy-going"]),
    (r"young|youthful|naive|innocent", ["Youthful", "Bright"]),
]
_DELIVERY = [
    (r"energetic|cheerful|bubbly|playful|excit|lively|curious", "lively and expressive"),
    (r"serious|stern|confident|protective|authoritative|commanding", "steady and confident"),
    (r"calm|quiet|shy|gentle|thoughtful|timid", "soft and measured"),
    (r"warm|kind|loving|caring", "warm and natural"),
    (r"menacing|villain|cruel|cunning", "cool and controlled"),
]


def _wanted(profile: Profile, description: str) -> list[str]:
    text = f"{profile.personality} {description}".lower()
    out: list[str] = []
    for pattern, descs in _TRAITS:
        if re.search(pattern, text):
            out += [d for d in descs if d not in out]
    if profile.age_category == "elderly":
        out += [d for d in ("Mature", "Gravelly", "Even", "Knowledgeable") if d not in out]
    return out


def _delivery(profile: Profile, description: str) -> str:
    text = f"{profile.personality} {description}".lower()
    for pattern, d in _DELIVERY:
        if re.search(pattern, text):
            return d
    return "natural and conversational"


def _tiebreak(seed: str, voice_id: str) -> str:
    return hashlib.sha256(f"{seed}|{voice_id}".encode()).hexdigest()


@dataclass
class Decision:
    status: str                         # cast / approximate / needs_voice / unsupported_language
    provider: str = "google"
    voice: str = ""
    voice_category: str = ""
    voice_description: str = ""
    voice_style: str = ""
    language: str = "English"
    reason: str = ""


def voice_category(profile: Profile) -> str:
    g = profile.gender if profile.gender in ("female", "male") else "neutral"
    return f"{_CATEGORY_GROUP.get(profile.age_category, 'adult')}_{g}"


def describe_voice(profile: Profile, voice: Optional[CatalogVoice], language: str, description: str,
                   approximate: bool = False) -> str:
    who = {"child": "childlike", "teen": "teenage", "adult": "adult", "elderly": "older adult"}[
        _CATEGORY_GROUP.get(profile.age_category, "adult")]
    if approximate:
        who = "mature adult (closest available for an older character)"
    g = {"female": "female", "male": "male"}.get(profile.gender, "")
    base = f"A natural {who} {g} speaking voice".replace("  ", " ")
    if voice is not None and voice.descriptor:
        base += f", {voice.descriptor.lower()} in character"
    return f"{base}, clear {language} pronunciation, {_delivery(profile, description)} delivery."[:400]


def cast(profile: Profile, catalog: list[CatalogVoice], taken: set, seed: str, description: str = "",
         provider: str = "google") -> Decision:
    """Picks the voice for one character. `taken` holds voices already used
    by other characters in the production or series; they are avoided when
    an equally suitable free voice exists, never at the cost of age or
    language fit."""
    ok, lang = language_supported(provider, profile.language)
    category = voice_category(profile)
    if not ok:
        return Decision("unsupported_language", provider, "", category, "", "", profile.language,
                        f"The voice model doesn't speak {profile.language} yet.")
    usable = [v for v in catalog if v.provider == provider and v.status == "available"
              and ("all" in v.languages or lang in v.languages)]
    groups = ADULT if profile.age_category == "unknown" else [profile.age_category]
    pool = [v for v in usable if any(g in v.age_categories for g in groups)]
    approximate = False
    if not pool and profile.age_category == "elderly":
        pool = [v for v in usable if any(g in v.age_categories for g in ("middle_aged",))]
        approximate = bool(pool)
    if not pool:
        label = AGE_LABEL.get(profile.age_category, "this character")
        return Decision("needs_voice", provider, "", category, "", "", lang,
                        f"Viyo doesn't have a verified {label} voice yet.")
    if profile.gender in ("female", "male"):
        matching = [v for v in pool if v.gender == profile.gender]
        neutral = [v for v in pool if v.gender == "unspecified"]
        pool = matching or neutral
        if not pool:
            return Decision("needs_voice", provider, "", category, "", "", lang,
                            f"Viyo doesn't have a verified {profile.gender} {AGE_LABEL.get(profile.age_category, '')} voice yet.")
    wanted = _wanted(profile, description)
    taken = set(taken) | {NARRATOR_VOICE}   # keep the narrator's voice for the narrator

    def score(v: CatalogVoice):
        s = 0
        if v.descriptor in wanted:
            s += 3 + (len(wanted) - wanted.index(v.descriptor)) * 0.1
        if v.voice_id in taken:
            s -= 5
        return (-s, _tiebreak(seed, v.voice_id))

    best = sorted(pool, key=score)[0]
    style_bits = [d for d in _delivery(profile, description).split(" and ") if d not in ("natural", "conversational")]
    return Decision("approximate" if approximate else "cast", provider, best.voice_id, category,
                    describe_voice(profile, best, lang, description, approximate),
                    ", ".join(style_bits)[:80], lang,
                    "Closest available voice for an older character." if approximate else "")


def identity_seed(owner_id: str, series_or_project_id: str, name: str) -> str:
    return f"{owner_id}:{series_or_project_id}:{name.strip().lower()}"
