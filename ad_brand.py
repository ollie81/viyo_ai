"""The advertiser's own facts are the source of truth for every ad.

The brand name, website and call to action come only from the campaign the
advertiser filled in. Nothing the AI writes may add, change or replace them:

  - BrandFacts.from_campaign() reads them exactly as supplied (the website is
    validated, never "fixed" into another address);
  - find_domains() spots any website address in a piece of text, written
    ("ourollie.space", "https://x.y/z") or spoken ("ollie care dot com");
  - sanitize() replaces an address that isn't the advertiser's with the real
    one (or removes it when the campaign has no website), and puts the real
    brand name back where the AI wrote a made-up variant of it
    ("OllieCare" -> "Ollie");
  - sanitize_script() applies that to every caption, voiceover line and visual
    description before anything is rendered, and reports what it changed.

Nothing here is specific to one advertiser.
"""
from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional

# Written addresses: optional scheme and www, labels, a letters-only TLD, optional path.
_DOMAIN = re.compile(
    r"(?i)(?<![@\w.-])(?:https?://)?(?:www\.)?"
    r"((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24})"
    r"(?![\w-])(/[^\s,;!?\"')]*)?"
)
# Spoken addresses, as a voiceover would say them: "ollie care dot com".
_SPOKEN = re.compile(r"(?i)\b((?:[a-z0-9-]+\s+){0,2}[a-z0-9-]+)\s+dot\s+(com|net|org|io|app|ai|co|space|site|"
                     r"store|shop|online|tv|me|xyz|info|biz|dev|rw|uk|us)\b")
# Look like a domain to a regex but are file names or abbreviations.
_NOT_DOMAINS = {"png", "jpg", "jpeg", "gif", "webp", "mp4", "mov", "mp3", "wav", "pdf", "txt", "doc", "docx",
                "csv", "json", "html", "js", "py", "exe", "zip"}
_LEAD_WORDS = {"to", "visit", "at", "go", "on", "check", "out", "us", "the", "via", "from", "head", "find", "and",
               "or", "now", "today", "website", "site", "our"}
_DANGLING = re.compile(r"(?i)\b(?:visit|go to|head to|check out|find us at|at|on|via|from)\s*([.!?,]|$)")


def _norm_host(host: str) -> str:
    host = host.strip().lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


def parse_url(raw: str) -> tuple[str, str, str]:
    """(exact url as supplied, host without www, display form) — or ("", "", "") if it isn't a usable
    web address. The display form drops only the scheme and a trailing slash; the address itself is
    never altered."""
    raw = (raw or "").strip()
    if not raw:
        return "", "", ""
    parsed = urllib.parse.urlsplit(raw if "://" in raw else "https://" + raw)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or "." not in host or not re.fullmatch(r"[a-z0-9.-]+", host):
        return "", "", ""
    tld = host.rsplit(".", 1)[-1]
    if not re.fullmatch(r"[a-z]{2,24}", tld):
        return "", "", ""
    display = raw.split("://", 1)[-1].rstrip("/")
    return raw, _norm_host(host), display


@dataclass
class BrandFacts:
    name: str = ""                 # the brand: shown on the end card, never changed
    product: str = ""              # the product, when it has its own name different from the brand
    url: str = ""                  # exactly as the advertiser supplied it
    host: str = ""                 # for comparisons: lowercase, no www
    display_url: str = ""          # what's shown on screen: the address without "https://"
    cta: str = ""
    description: str = ""
    features: list = field(default_factory=list)
    issues: list = field(default_factory=list)

    @classmethod
    def from_campaign(cls, c: dict) -> "BrandFacts":
        brand = (c.get("brand_name") or "").strip()
        target = (c.get("target_name") or "").strip()
        f = cls(name=brand or target, product=target if brand and target.lower() != brand.lower() else "",
                cta=(c.get("cta_text") or "").strip(),
                description=(c.get("target_description") or "").strip(),
                features=[str(x) for x in (c.get("target_features") or [])])
        supplied = (c.get("destination_link") or "").strip()
        f.url, f.host, f.display_url = parse_url(supplied)
        if supplied and not f.url:
            f.issues.append(f'The website "{supplied}" isn\'t a valid web address, so no website is shown. '
                            "Fix it in the campaign to show it.")
        if not f.name:
            f.issues.append("The campaign has no brand or product name, so none is shown.")
        return f

    def allows(self, host: str) -> bool:
        return bool(self.host) and _norm_host(host) == self.host

    def is_exact(self, written: str) -> bool:
        """True when a written address is the advertiser's own, path included (scheme, www and a
        trailing slash don't matter). The right domain with a made-up page is not exact."""
        if not self.url:
            return False
        _, host, display = parse_url(written)
        strip = lambda d: re.sub(r"(?i)^www\.", "", d).rstrip("/").lower()   # noqa: E731
        return host == self.host and strip(display) == strip(self.display_url)


def _spoken_host(group: str, tld: str) -> tuple[str, str]:
    """(lead words to keep, host) from a spoken match like "to ollie care" + "com"."""
    words = group.split()
    while len(words) > 1 and words[0].lower() in _LEAD_WORDS:
        words = words[1:]
    lead = group[: len(group) - len(" ".join(words))] if words else group
    return lead, _norm_host("".join(words) + "." + tld)


def _spoken_allowed(group: str, tld: str, facts: "BrandFacts") -> bool:
    words = group.split()
    return any(facts.allows("".join(words[k:]) + "." + tld) for k in range(len(words)))


def find_domains(text: str) -> list[str]:
    """Every website address in [text], written or spoken, as a host name."""
    out = []
    for m in _DOMAIN.finditer(text or ""):
        host = m.group(1)
        if host.rsplit(".", 1)[-1].lower() in _NOT_DOMAINS:
            continue
        out.append(_norm_host(host))
    for m in _SPOKEN.finditer(text or ""):
        out.append(_spoken_host(m.group(1), m.group(2))[1])
    return out


# For text read off video frames (OCR), only realistic addresses count, so recognition noise like "ii.ll"
# never triggers a correction.
_OCR_TLDS = {"com", "net", "org", "io", "app", "ai", "co", "space", "site", "store", "shop", "online", "tv", "me",
             "xyz", "info", "biz", "dev", "rw", "uk", "us", "ca", "de", "fr", "ng", "ke", "za", "in", "ug", "tz",
             "gh", "eu", "au", "care", "health", "life", "live", "world", "page", "link"}


def ocr_foreign_domains(text: str, facts: "BrandFacts") -> list[str]:
    out = []
    for d in find_domains(text):
        label, _, tld = d.rpartition(".")
        if tld in _OCR_TLDS and len(label.split(".")[-1]) >= 3 and not facts.allows(d):
            out.append(d)
    return out


def foreign_domains(text: str, facts: BrandFacts) -> list[str]:
    """Addresses in [text] that aren't the advertiser's (by domain)."""
    bad = [d for d in find_domains(text) if not facts.allows(d)]
    for m in _SPOKEN.finditer(text or ""):
        if _spoken_allowed(m.group(1), m.group(2), facts):
            bad = [d for d in bad if d != _spoken_host(m.group(1), m.group(2))[1]]
    return bad


def _brand_variants(text: str, facts: BrandFacts) -> list[str]:
    """Made-up versions of the brand name, like "OllieCare" for "Ollie": the name with letters glued on.
    Words the advertiser itself uses (in the description or features) are not variants."""
    name = facts.name
    if not re.fullmatch(r"[A-Za-z0-9]{3,}", name or ""):
        return []
    own = (facts.description + " " + " ".join(facts.features)).lower()
    found = []
    for m in re.finditer(rf"(?i)\b{re.escape(name)}([A-Za-z]{{2,}})\b", text or ""):
        word = m.group(0)
        if word.lower() != name.lower() and word.lower() not in own and not m.group(1).lower() in ("s", "es"):
            found.append(word)
    return found


def sanitize(text: str, facts: BrandFacts, spoken: bool = False) -> tuple[str, list[str]]:
    """Returns (corrected text, what was changed). Invented addresses become the advertiser's real one,
    or are removed when there is none; invented brand-name variants become the real name."""
    if not text:
        return text or "", []
    changes, out = [], text

    def fix_spoken(m):
        if _spoken_allowed(m.group(1), m.group(2), facts):
            return m.group(0)
        lead, host = _spoken_host(m.group(1), m.group(2))
        changes.append(f'replaced spoken "{host}"' + (" with the real website" if facts.url else " (no website in the campaign)"))
        return lead + (_spoken(facts) if facts.url else "")

    # Spoken forms first, so the real address put back by the written pass isn't re-read as spoken.
    out = _SPOKEN.sub(fix_spoken, out)

    def fix_written(m):
        host = m.group(1)
        if host.rsplit(".", 1)[-1].lower() in _NOT_DOMAINS:
            return m.group(0)
        if facts.is_exact(m.group(0)):
            return _spoken(facts) if spoken else m.group(0)
        changes.append(f'replaced "{m.group(0)}"' + (f' with "{facts.display_url}"' if facts.url else " (no website in the campaign)"))
        return _spoken(facts) if (spoken and facts.url) else (facts.display_url if facts.url else "")

    out = _DOMAIN.sub(fix_written, out)
    for v in _brand_variants(out, facts):
        out = re.sub(rf"\b{re.escape(v)}\b", facts.name, out)
        changes.append(f'replaced "{v}" with "{facts.name}"')
    if changes:
        out = _DANGLING.sub(lambda m: m.group(1), out)
        out = re.sub(r"\s+([.!?,;:])", r"\1", re.sub(r"\s{2,}", " ", out)).strip(" ,")
    return out, changes


def _spoken(facts: BrandFacts) -> str:
    """How a voiceover says the real address: "ourollie dot space"."""
    return facts.display_url.replace(".", " dot ").replace("/", " slash ")


def sanitize_script(scenes: list[dict], facts: BrandFacts) -> tuple[list[dict], list[str]]:
    """Applies sanitize() to every scene's caption, voiceover and visual description. The final scene's
    caption becomes the advertiser's own call to action (or nothing, if they gave none) — it's shown on the
    end card, never rewritten by the AI."""
    report = []
    for i, s in enumerate(scenes):
        for key, spoken in (("caption_text", False), ("dialogue_or_vo", True), ("visual_description", False)):
            fixed, changes = sanitize(s.get(key) or "", facts, spoken=spoken)
            if key == "visual_description":
                # Image prompts never contain an address at all: generated images must not show text.
                fixed = _DOMAIN.sub("", fixed)
            s[key] = fixed
            report += [f"Scene {i + 1} {key.replace('_', ' ')}: {c}" for c in changes]
    if scenes:
        scenes[-1]["is_cta"] = True
        scenes[-1]["caption_text"] = facts.cta
    return scenes, report


def prompt_facts(facts: BrandFacts) -> str:
    """The exact facts block every script/hook prompt carries, with the rules that go with it."""
    site = facts.display_url or "NONE — this campaign has no website. Never mention or show any website."
    cta = facts.cta or "NONE supplied — end on the brand name. Never invent a call to action."
    product = f"PRODUCT NAME (exact spelling, never change it): {facts.product}\n" if facts.product else ""
    return (f"BRAND NAME (exact spelling, never change it): {facts.name or 'not given — do not invent one'}\n"
            f"{product}"
            f"WEBSITE (the only address that may ever appear or be spoken): {site}\n"
            f"CALL TO ACTION (exact words): {cta}\n"
            "Never invent, shorten, translate or vary the brand name, website, product names or call to action. "
            "Never put any website address in visual descriptions: on-screen text is added later from these "
            "exact facts.\n")


def describe_issues(facts: BrandFacts, changes: list[str]) -> Optional[str]:
    lines = list(facts.issues) + changes
    return "\n".join(lines) if lines else None
