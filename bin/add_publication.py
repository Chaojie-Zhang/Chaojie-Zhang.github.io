#!/usr/bin/env python3
"""Add a publication to _bibliography/papers.bib from a DOI.

Usage: python3 bin/add_publication.py <doi> [--summary "..."] [--selected]

Fetches Crossref metadata, generates a correctly formatted BibTeX entry
(authors, official journal abbreviation when available, protected capitals,
volume/pages, DOI), and inserts it at the top of papers.bib. Idempotent: a
DOI already present in the file is reported and skipped.
"""
import argparse
import json
import os
import re
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIB = os.environ.get("BIB_PATH", os.path.join(ROOT, "_bibliography", "papers.bib"))
MAILTO = "chaojiez@ucla.edu"

# preferred short names for journals Crossref abbreviates poorly or not at all
ABBR = {
    "physical review letters": "Phys. Rev. Lett.",
    "physical review accelerators and beams": "Phys. Rev. Accel. Beams",
    "physical review research": "Phys. Rev. Res.",
    "physical review e": "Phys. Rev. E",
    "physical review applied": "Phys. Rev. Appl.",
    "physics of plasmas": "Phys. Plasmas",
    "plasma physics and controlled fusion": "Plasma Phys. Control. Fusion",
    "nature communications": "Nat. Commun.",
    "nature physics": "Nat. Phys.",
    "nature photonics": "Nat. Photon.",
    "proceedings of the national academy of sciences": "PNAS",
    "science advances": "Sci. Adv.",
    "scientific reports": "Sci. Rep.",
    "communications physics": "Commun. Phys.",
    "new journal of physics": "New J. Phys.",
    "journal of plasma physics": "J. Plasma Phys.",
    "reviews of modern plasma physics": "Rev. Mod. Plasma Phys.",
    "optics express": "Opt. Express",
    "chinese physics c": "Chin. Phys. C",
}

TYPE_MAP = {
    "journal-article": "article",
    "proceedings-article": "inproceedings",
    "book-chapter": "incollection",
    "book": "book",
}

STOPWORDS = {"a", "an", "the", "on", "of", "in", "for", "and", "to", "with", "at", "from"}


def protect(title):
    """Wrap tokens with internal capitals, all-caps, or digits+caps in braces."""
    out = []
    for tok in title.split():
        core = re.sub(r"^\W+|\W+$", "", tok)
        needs = bool(re.search(r"[A-Z]", core[1:])) or (core.isupper() and len(core) >= 2)
        if needs and not tok.startswith("{"):
            tok = tok.replace(core, "{" + core + "}", 1)
        out.append(tok)
    return " ".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("doi")
    p.add_argument("--summary", default="")
    p.add_argument("--selected", action="store_true")
    a = p.parse_args()

    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", a.doi.strip())
    text = open(BIB).read()
    if re.search(re.escape(doi), text, re.I):
        print(f"Already present: an entry with DOI {doi} exists. Nothing to do.")
        return 0

    url = f"https://api.crossref.org/works/{urllib.request.quote(doi)}?mailto={MAILTO}"
    with urllib.request.urlopen(url, timeout=20) as r:
        msg = json.load(r)["message"]

    entrytype = TYPE_MAP.get(msg.get("type", ""), "article")
    title = protect(msg["title"][0].strip())
    authors = " and ".join(
        f"{au.get('given', '')} {au.get('family', '')}".strip()
        for au in msg.get("author", []))
    journal = (msg.get("container-title") or ["Unknown"])[0]
    short = (msg.get("short-container-title") or [""])
    abbr = ABBR.get(journal.lower()) or (short[0] if short and short[0] else journal)
    year = str((msg.get("published-print") or msg.get("published-online")
                or msg["issued"])["date-parts"][0][0])
    volume = msg.get("volume", "")
    pages = msg.get("page") or msg.get("article-number") or ""

    family = re.sub(r"[^a-z]", "", (msg.get("author") or [{}])[0].get("family", "anon").lower())
    word = next((re.sub(r"[^a-z]", "", w.lower()) for w in msg["title"][0].split()
                 if re.sub(r"[^a-z]", "", w.lower()) not in STOPWORDS
                 and len(re.sub(r"[^a-z]", "", w.lower())) > 2), "paper")
    key = f"{family}{year}{word}"
    while re.search(r"@\w+\{" + re.escape(key) + ",", text):
        key += "x"

    lines = [f"@{entrytype}{{{key},"]
    def field(name, value):
        if value:
            lines.append(f"  {name:<11} = {{{value}}},")
    field("abbr", abbr)
    field("bibtex_show", "true")
    if a.selected:
        field("selected", "true")
    field("title", title)
    field("author", authors)
    field("journal" if entrytype == "article" else "booktitle", journal)
    field("volume", volume)
    field("pages", pages)
    if a.summary:
        field("note", "Summary: " + a.summary.strip())
    field("year", year)
    field("doi", doi)
    lines[-1] = lines[-1].rstrip(",") + ","
    entry = "\n".join(lines) + "\n}\n\n"

    i = text.find("@")
    open(BIB, "w").write(text[:i] + entry + text[i:])
    print("Added entry:\n")
    print(entry)
    print(f"NOTE: no summary was provided." if not a.summary else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
