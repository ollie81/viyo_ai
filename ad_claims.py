"""Verified product facts, and the guard that keeps an ad to them.

The advertiser supplies the facts (features, benefits, prices, selling points, ingredients or specs,
contact details, social accounts). Everything else is an assumption, and an ad may not state an
assumption as fact. This module:

  - FactSheet.from_campaign(): the advertiser's facts, each with an id (F1, F2, ...) the script prompt
    shows, so the model can say which fact a line relies on;
  - claim_problems(text): the risky statements in a line that aren't backed by a supplied fact:
      prices and money amounts, percentages and other numeric claims ("2x", "24/7", "500 mg"),
      guarantee/superlative/health words ("guaranteed", "best", "#1", "clinically proven", "organic",
      "sugar-free", "cures", ...), ingredients/materials ("made with X", "contains X"), phone numbers,
      emails and social handles;
  - enforce(text): removes the sentences that make such claims (an ad that says less is better than one
    that says something untrue) and reports each removal;
  - enforce_script(): runs that over every caption and voiceover line.

A fact counts as supplied when its key words appear in the fact sheet (prices and numbers are compared
as numbers, so "RWF 1,500" and "1500 Frw" match). Nothing here knows about any particular brand.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

FACT_FIELDS = ("features", "benefits", "prices", "selling_points", "specs")

_CURRENCY = r"(?:\$|€|£|¥|₦|USD|EUR|GBP|RWF|Frw|FRW|KES|UGX|TZS|NGN|ZAR|GHS|Ksh|Rs\.?|₹)"
_MONEY = re.compile(rf"(?i)(?:{_CURRENCY}\s?\d[\d,.\s]*\d|\d[\d,.\s]*\d?\s?{_CURRENCY}|{_CURRENCY}\s?\d)")
_NUMBERISH = re.compile(r"(?i)\b\d+(?:[.,]\d+)?\s?(?:%|x\b|percent\b|mg\b|g\b|kg\b|ml\b|l\b|cal\b|calories\b|kcal\b|"
                        r"hours?\b|hrs?\b|days?\b|years?\b|months?\b|minutes?\b|mins?\b|km\b|mph\b|gb\b|tb\b|mp\b)|"
                        r"\b\d+\s?/\s?\d+\b|#\s?1\b|\bno\.?\s?1\b|\bnumber one\b")
_RISKY_WORDS = [
    "guarantee", "guaranteed", "warranty", "best", "number one", "#1", "no.1", "cheapest", "lowest price",
    "clinically", "proven", "scientifically", "doctor", "dermatologist", "approved", "certified", "award",
    "award-winning", "organic", "natural", "all-natural", "sugar-free", "sugar free", "zero sugar", "no sugar",
    "gluten-free", "vegan", "halal", "kosher", "fat-free", "low-fat", "low calorie", "healthy", "cure", "cures",
    "heal", "heals", "treat", "treats", "prevent", "prevents", "weight loss", "lose weight", "boosts immunity",
    "detox", "free shipping", "free delivery", "refund", "money back", "risk-free", "safe", "100%", "unlimited",
    "lifetime", "fastest", "strongest", "most popular", "world's", "the only", "only one", "exclusive", "official", "eco-friendly",
    "sustainable", "waterproof", "handmade", "imported", "fresh", "authentic", "original", "premium",
]
_INGREDIENT = re.compile(r"(?i)\b(?:made (?:with|from)|contains?|packed with|infused with|full of|ingredients?:?|"
                         r"brewed with|crafted (?:with|from)|loaded with)\s+([a-z][a-z\s,&-]{2,60})")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
_EMAIL = re.compile(r"(?i)\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_HANDLE = re.compile(r"(?<![\w@])@[A-Za-z0-9_.]{2,30}\b")


def _num(s: str) -> Optional[float]:
    digits = re.sub(r"[^\d.,]", "", s).replace(" ", "")
    if not digits:
        return None
    if digits.count(",") and digits.count("."):
        digits = digits.replace(",", "")
    elif digits.count(",") and len(digits.split(",")[-1]) == 3:
        digits = digits.replace(",", "")
    else:
        digits = digits.replace(",", ".")
    try:
        return float(digits.strip("."))
    except ValueError:
        return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%#@+.\s-]", " ", (s or "").lower())).strip()


@dataclass
class FactSheet:
    facts: list = field(default_factory=list)         # [(id, field, text)]
    phones: list = field(default_factory=list)
    emails: list = field(default_factory=list)
    handles: list = field(default_factory=list)
    extra: str = ""                                    # name/description: words the advertiser used

    @classmethod
    def from_campaign(cls, c: dict) -> "FactSheet":
        raw = c.get("facts") or {}
        sheet = cls()
        n = 0
        for f in FACT_FIELDS:
            items = raw.get(f) or []
            if f == "features" and not items:
                items = c.get("target_features") or []      # the original single feature list
            for item in items:
                text = str(item).strip()
                if text:
                    n += 1
                    sheet.facts.append((f"F{n}", f, text[:300]))
        contact = c.get("contact") or {}
        sheet.phones = [str(contact.get("phone"))] if contact.get("phone") else []
        sheet.emails = [str(contact.get("email")).lower()] if contact.get("email") else []
        for s in (c.get("social_links") or []):
            m = re.search(r"@?([A-Za-z0-9_.]{2,30})/?$", str(s).strip())
            if m:
                sheet.handles.append("@" + m.group(1).lower())
        sheet.extra = " ".join([c.get("target_name") or "", c.get("brand_name") or "", c.get("target_description") or "",
                                str(contact.get("address") or "")])
        return sheet

    @property
    def corpus(self) -> str:
        return _norm(" ".join(t for _, _, t in self.facts) + " " + self.extra)

    def numbers(self) -> list[float]:
        out = []
        for _, _, t in self.facts:
            for m in re.finditer(r"\d[\d,.\s]*\d|\d", t):
                v = _num(m.group(0))
                if v is not None:
                    out.append(v)
        return out

    def prompt_block(self) -> str:
        if not self.facts:
            facts = "  (none supplied: make no specific claims about specs, ingredients, prices, results or guarantees)"
        else:
            facts = "\n".join(f"  {fid} [{f.replace('_', ' ')}]: {t}" for fid, f, t in self.facts)
        contact = []
        if self.phones:
            contact.append(f"phone {self.phones[0]}")
        if self.emails:
            contact.append(f"email {self.emails[0]}")
        if self.handles:
            contact.append("social " + ", ".join(self.handles))
        return ("VERIFIED FACTS (supplied by the advertiser; the ONLY claims the ad may make):\n" + facts + "\n"
                + (f"CONTACT (exact, only if useful): {'; '.join(contact)}\n" if contact else "")
                + "Anything not listed above is an assumption. Never state an assumption as fact: no invented "
                  "prices, discounts, specs, ingredients, health or results claims, guarantees, awards, rankings "
                  "or contact details. For each scene, list in fact_ids the ids of the facts its words rely on.\n")


def _backed_number(token: str, sheet: FactSheet) -> bool:
    v = _num(token)
    return v is not None and any(abs(v - n) < 1e-6 for n in sheet.numbers())


def claim_problems(text: str, sheet: FactSheet) -> list[str]:
    """Statements in [text] that no supplied fact backs."""
    if not text:
        return []
    probs, corpus = [], sheet.corpus
    for m in _MONEY.finditer(text):
        if not _backed_number(m.group(0), sheet):
            probs.append(f'price "{m.group(0).strip()}"')
    for m in _NUMBERISH.finditer(text):
        tok = m.group(0)
        if _norm(tok) in corpus or (re.search(r"\d", tok) and _backed_number(tok, sheet) and
                                    _norm(re.sub(r"[\d.,\s]", "", tok)) in corpus + " %"):
            continue
        probs.append(f'claim "{tok.strip()}"')
    low = _norm(text)
    for w in _RISKY_WORDS:
        if re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", low) and w not in corpus:
            probs.append(f'claim "{w}"')
    for m in _INGREDIENT.finditer(text):
        items = [i.strip() for i in re.split(r",|&|\band\b", m.group(1)) if i.strip()]
        for item in items:
            words = [w for w in _norm(item).split() if len(w) > 2]
            if words and not all(w in corpus for w in words[:2]):
                probs.append(f'ingredient "{item}"')
    for m in _PHONE.finditer(text):
        if re.sub(r"\D", "", m.group(0)) not in [re.sub(r"\D", "", p) for p in sheet.phones]:
            probs.append(f'phone "{m.group(0).strip()}"')
    for m in _EMAIL.finditer(text):
        if m.group(0).lower() not in sheet.emails:
            probs.append(f'email "{m.group(0)}"')
    for m in _HANDLE.finditer(text):
        if m.group(0).lower() not in sheet.handles:
            probs.append(f'social account "{m.group(0)}"')
    return list(dict.fromkeys(probs))


def enforce(text: str, sheet: FactSheet) -> tuple[str, list[str]]:
    """Drops each sentence that makes an unsupported claim. Returns (kept text, what was removed and why)."""
    if not text:
        return text or "", []
    kept, removed = [], []
    # A sentence ends at . ! or ? followed by a space (or the end), never at the dot in "kivu.com" or "2.5".
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        probs = claim_problems(sentence, sheet)
        if probs:
            removed.append(f'removed "{sentence.strip()}" ({", ".join(probs)} not in the supplied facts)')
        elif sentence.strip():
            kept.append(sentence.strip())
    return " ".join(kept), removed


def enforce_script(scenes: list[dict], sheet: FactSheet) -> tuple[list[dict], list[str]]:
    """Applies enforce() to every caption and voiceover line (the closing card is built from facts only).
    A scene left with nothing to say keeps its picture and stays silent rather than saying something false."""
    report = []
    for i, s in enumerate(scenes):
        for key in ("caption_text", "dialogue_or_vo"):
            if i == len(scenes) - 1 and key == "caption_text":
                continue
            fixed, removed = enforce(s.get(key) or "", sheet)
            s[key] = fixed
            report += [f"Scene {i + 1} {'caption' if key == 'caption_text' else 'voiceover'}: {r}" for r in removed]
    return scenes, report
