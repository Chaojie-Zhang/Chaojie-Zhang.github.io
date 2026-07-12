#!/usr/bin/env python3
"""Health check for _bibliography/papers.bib.

Local checks: unique keys, balanced braces, required fields.
Crossref checks (entries with a DOI): the DOI resolves, and its recorded
volume / first page / title agree with Crossref's record — the exact class
of data error that has bitten this bibliography before.

Exit code 1 on errors (fails the CI check); warnings only print.
"""
import json
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIB = os.path.join(ROOT, "_bibliography", "papers.bib")
MAILTO = "chaojiez@ucla.edu"


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def first_page(pages):
    return re.split(r"[-–—]+", pages)[0].strip().lstrip("0")


def main():
    text = open(BIB).read()
    errors, warnings = [], []
    entries = []
    keys = set()

    for chunk in re.split(r"(?=^@)", text, flags=re.M):
        m = re.match(r"@(\w+)\{([^,]+),", chunk)
        if not m:
            continue
        key = m.group(2)
        if key in keys:
            errors.append(f"duplicate key: {key}")
        keys.add(key)
        if chunk.count("{") != chunk.count("}"):
            errors.append(f"{key}: unbalanced braces")
        fields = dict(re.findall(r"^\s*(\w+)\s*=\s*\{(.*?)\},?\s*$", chunk, re.M))
        for req in ("title", "author", "year"):
            if not fields.get(req):
                errors.append(f"{key}: missing required field '{req}'")
        if m.group(1) == "article" and not fields.get("journal"):
            errors.append(f"{key}: @article without journal")
        if not fields.get("doi"):
            warnings.append(f"{key}: no DOI")
        entries.append((key, fields))

    print(f"{len(entries)} entries, {len([1 for _, f in entries if f.get('doi')])} with DOIs")

    for key, f in entries:
        doi = f.get("doi")
        if not doi:
            continue
        url = f"https://api.crossref.org/works/{urllib.request.quote(doi)}?mailto={MAILTO}"
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                msg = json.load(r)["message"]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                errors.append(f"{key}: DOI does not resolve on Crossref: {doi}")
            else:
                warnings.append(f"{key}: Crossref HTTP {e.code} (skipped)")
            continue
        except Exception as e:  # network blip: warn, don't fail CI
            warnings.append(f"{key}: Crossref unreachable ({e}); skipped")
            continue

        cr_title = (msg.get("title") or [""])[0]
        if cr_title and norm(re.sub(r"[{}]", "", f.get("title", "")))[:30] != norm(cr_title)[:30]:
            errors.append(f"{key}: title disagrees with Crossref for {doi}\n"
                          f"    bib     : {f.get('title', '')[:70]}\n"
                          f"    crossref: {cr_title[:70]}")
        cr_vol = msg.get("volume", "")
        if f.get("volume") and cr_vol and f["volume"].lstrip("0") != cr_vol.lstrip("0"):
            errors.append(f"{key}: volume {f['volume']} != Crossref {cr_vol}")
        cr_page = msg.get("page") or msg.get("article-number") or ""
        if f.get("pages") and cr_page and first_page(f["pages"]) != first_page(cr_page):
            errors.append(f"{key}: pages {f['pages']} != Crossref {cr_page}")
        time.sleep(0.3)

    print()
    for w in warnings:
        print("WARN :", w)
    for e in errors:
        print("ERROR:", e)
    print(f"\n{len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
