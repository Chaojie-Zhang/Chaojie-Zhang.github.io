#!/usr/bin/env python3
"""Match uploaded paper PDFs to publications and wire up their PDF buttons.

Scans assets/pdf/papers/ for PDFs not yet referenced in
_bibliography/papers.bib, matches each to a bib entry, renames the file to
<citekey>.pdf, and inserts `pdf = {papers/<citekey>.pdf}` into the entry.

Matching strategies, in order:
  1. DOI fragment  — any chunk of the entry's DOI appearing in the filename
                     (publisher downloads like s41467-024-48413-y.pdf or
                     072108_1_5.0207697.pdf)
  2. volume+pages  — both numbers appearing in the filename
                     (PhysRevLett.133.063201.pdf, Zhang_2024_PPCF_66_025013.pdf)
  3. title prefix  — normalized filename contains/starts the normalized title
                     (Zotero-style "Author et al. - 2024 - Title.pdf")

Safe to run repeatedly: already-referenced PDFs are skipped, ambiguous or
unmatched files are reported and left untouched.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIB = os.path.join(ROOT, "_bibliography", "papers.bib")
PAPERS = os.path.join(ROOT, "assets", "pdf", "papers")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def parse_entries(text):
    entries = []
    for chunk in re.split(r"(?=^@)", text, flags=re.M):
        m = re.match(r"@\w+\{([^,]+),", chunk)
        if not m:
            continue
        fields = dict(re.findall(r"^\s*(\w+)\s*=\s*\{(.*?)\},?\s*$", chunk, re.M))
        entries.append({
            "key": m.group(1),
            "chunk": chunk,
            "title": fields.get("title", ""),
            "doi": fields.get("doi", ""),
            "volume": fields.get("volume", ""),
            "pages": fields.get("pages", ""),
            "pdf": fields.get("pdf", ""),
        })
    return entries


def match(fname, entries):
    stem = os.path.splitext(fname)[0]
    nstem = norm(stem)
    hits = []
    for e in entries:
        # 1. DOI fragment: match on the DOI suffix (after the publisher prefix)
        if e["doi"]:
            suffix = e["doi"].split("/", 1)[-1]
            if len(norm(suffix)) >= 6 and norm(suffix) in nstem:
                hits.append((e, "doi"))
                continue
        # 2. volume + pages, as distinct number tokens in the filename
        if e["volume"] and e["pages"]:
            tokens = re.findall(r"\d+", stem)
            if e["volume"] in tokens and e["pages"].lstrip("0") in [t.lstrip("0") for t in tokens]:
                hits.append((e, "vol+pages"))
                continue
        # 3. title prefix (Zotero truncates titles, so prefix match)
        ntitle = norm(re.sub(r"[{}]", "", e["title"]))
        # strip "author et al year" prefix from the filename before comparing
        tail = re.sub(r"^.*?\d{4}", "", stem)
        ntail = norm(tail)
        if len(ntail) >= 25 and (ntail[:40] in ntitle or ntitle[:40] in ntail):
            hits.append((e, "title"))
    return hits


def main():
    text = open(BIB).read()
    entries = parse_entries(text)
    referenced = {e["pdf"].replace("papers/", "") for e in entries if e["pdf"]}

    pending = [f for f in sorted(os.listdir(PAPERS))
               if f.lower().endswith(".pdf") and f not in referenced]
    if not pending:
        print("Nothing to do: every PDF in assets/pdf/papers/ is already wired up.")
        return 0

    changed = False
    for fname in pending:
        hits = match(fname, entries)
        keys = {h[0]["key"] for h in hits}
        if not hits:
            print(f"UNMATCHED : {fname} — no bib entry found, left as is")
            continue
        if len(keys) > 1:
            print(f"AMBIGUOUS : {fname} — matches {sorted(keys)}, left as is")
            continue

        entry, how = hits[0]
        if entry["pdf"]:
            print(f"SKIP      : {fname} — {entry['key']} already has a PDF ({entry['pdf']})")
            continue

        newname = f"{entry['key']}.pdf"
        if fname != newname:
            subprocess.run(["git", "mv", fname, newname], cwd=PAPERS, check=True)
        text = re.sub(
            r"(@\w+\{" + re.escape(entry["key"]) + r",\n)",
            r"\1  pdf         = {papers/" + newname + "},\n",
            text, count=1)
        entry["pdf"] = f"papers/{newname}"
        changed = True
        print(f"MATCHED   : {fname} -> {newname}  [{entry['key']}, via {how}]")

    if changed:
        open(BIB, "w").write(text)
        print("papers.bib updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
