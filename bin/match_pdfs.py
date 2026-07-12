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


def pdf_metadata(path):
    """DOI and title from the PDF's info dictionary, if present (APS, IOP and
    many other publishers embed both) — lets files with meaningless names
    like cavity.pdf match without any external tools."""
    try:
        with open(path, "rb") as f:
            head = f.read(400_000)
            f.seek(max(0, os.path.getsize(path) - 400_000))
            tail = f.read()
        raw = (head + b"\n" + tail).decode("latin-1", "ignore")
    except OSError:
        return "", ""
    doi = ""
    m = re.search(r"doi:(10\.\d{4,}/[^\s)\"'<>]+)", raw)
    if m:
        doi = m.group(1)
    title = ""
    m = re.search(r"/Title\s*\(((?:[^()\\]|\\.){15,400}?)\)", raw)
    if m:
        title = re.sub(r"\\(.)", r"\1", m.group(1))
    return doi, title


def first_page(pages):
    return re.split(r"[-–—]+", pages)[0].strip()


def match(fname, path, entries):
    stem = os.path.splitext(fname)[0]
    nstem = norm(stem)
    meta_doi, meta_title = pdf_metadata(path)
    hits = []
    for e in entries:
        # 0. exact DOI from the PDF's own metadata
        if meta_doi and e["doi"] and norm(meta_doi) == norm(e["doi"]):
            hits.append((e, "pdf-doi"))
            continue
        # 1. DOI fragment in the filename (after the publisher prefix)
        if e["doi"]:
            suffix = e["doi"].split("/", 1)[-1]
            if len(norm(suffix)) >= 6 and norm(suffix) in nstem:
                hits.append((e, "doi"))
                continue
        # 2. volume + pages (first page of a range), as tokens in the filename
        if e["volume"] and e["pages"]:
            tokens = [t.lstrip("0") for t in re.findall(r"\d+", stem)]
            if e["volume"].lstrip("0") in tokens and first_page(e["pages"]).lstrip("0") in tokens:
                hits.append((e, "vol+pages"))
                continue
        # 3. title, from PDF metadata or the filename (Zotero truncates, so prefix)
        ntitle = norm(re.sub(r"[{}]", "", e["title"]))
        if meta_title and norm(meta_title)[:60] == ntitle[:60] and len(ntitle) >= 25:
            hits.append((e, "pdf-title"))
            continue
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
        hits = match(fname, os.path.join(PAPERS, fname), entries)
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
