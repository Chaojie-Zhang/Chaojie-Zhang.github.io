# chaojiezhang.me — owner's manual

Source of [chaojiezhang.me](https://chaojiezhang.me), built on the
[al-folio](https://github.com/alshedivat/al-folio) v1 theme.

**How deployment works:** push anything to `master` → GitHub Actions builds the
site (~4 min) → the result lands on the `gh-pages` branch → GitHub Pages serves
it. You never touch `gh-pages` directly. If a build fails, the live site simply
stays on the last good version — nothing breaks publicly.

> After a deploy, browsers/CDN may show the old page for up to ~10 minutes.
> Hard-refresh with **Cmd+Shift+R** if a change doesn't appear.

---

## The buttons (Actions tab → pick a workflow → "Run workflow")

| Button | When to use it |
|---|---|
| **Add publication (from DOI)** | A new paper is out. Paste the DOI (optionally a one-sentence summary and a "selected" checkbox for the home page). A complete BibTeX entry is generated from Crossref and the site redeploys. |
| **Match paper PDFs** | After uploading PDFs (any filename) to `assets/pdf/papers/`. Matches each to its publication by DOI / volume+pages / embedded PDF metadata, renames it, and adds the PDF button. Unmatchable files are listed in the run log, never guessed. |
| **New blog post (draft)** | Starting a post. Give it a title (+ optional tags/description); a correctly scaffolded draft appears in `_posts/`, invisible until you remove its `published: false` line. |
| **Review blog post** | Before publishing. Enter part of the post's filename; a full review (scientific accuracy, neutral tone, Scientific American style, reputation-risk scan) arrives as a GitHub issue. Requires the `ANTHROPIC_API_KEY` repo secret — without it, paste `docs/POST_REVIEW_PROMPT.md` plus the post into [claude.ai](https://claude.ai) for the same review. |
| **Update citation stats** | Monthly-ish. Read total citations + h-index from your [Google Scholar profile](https://scholar.google.com/citations?user=CBjsrOUAAAAJ) and type the two numbers; the CV line updates. (Manual by design: automated sources have your identity split or merged with namesakes.) |
| **Bibliography health check** | Runs by itself whenever `papers.bib` changes; verifies every DOI'd entry's title/volume/pages against Crossref. Run manually anytime for an audit. |
| **Broken link check** | Runs monthly by itself; sweeps the live site and opens a GitHub issue listing dead links. |

---

## Common tasks

### Publish a new paper (the full routine)
1. **Actions → Add publication (from DOI)** — paste the DOI, write the
   one-sentence summary, tick *selected* if it should be featured on the
   home page.
2. Upload the PDF (author-accepted/arXiv version) to `assets/pdf/papers/`
   (web UI: *Add file → Upload files*).
3. **Actions → Match paper PDFs**.
4. Optional — award/prize on the entry: edit `_bibliography/papers.bib` and add
   inside the entry:
   ```bibtex
   award_name  = {🏆 Some Prize},
   award       = {Longer description shown when the button is clicked.},
   ```

### Write a blog post
1. **Actions → New blog post (draft)** with your title.
2. Open the new file in `_posts/` in the web editor and write.
3. **Actions → Review blog post** with the filename — read the review issue
   and address the BLOCKER/MAJOR findings (or run the prompt manually, see
   `docs/POST_REVIEW_PROMPT.md`).
4. Delete the `published: false` line to go live.

Conventions (also in the draft template): inline math is `\( ... \)`, display
math is `$$ ... $$` — single `$` is *not* math, so dollar amounts are safe.
Images go in `assets/img/posts/` and are embedded with
`<img src="{{ '/assets/img/posts/FILE.png' | relative_url }}">`.

### Moderate comments
Comments are [Cusdis](https://cusdis.com/dashboard) — you get an email per
comment (Quick Approve link), or approve in the dashboard. Nothing appears
publicly until approved.

### Edit fixed pages
| Page | File |
|---|---|
| Home/about (bio, research highlights) | `_pages/about.md` |
| CV | `_pages/cv.md` |
| Talks | `_pages/talks.md` |
| Projects | `_projects/*.md` |
| News items on the home page | `_news/*.md` (copy an existing one) |
| Site identity, analytics, nav | `_config.yml` |

---

## Things worth knowing

- **Staging:** risky experiments go to the
  [website-preview](https://github.com/Chaojie-Zhang/website-preview) repo
  first (serves at [/website-preview/](https://chaojiezhang.me/website-preview/)).
  Note its `_config.yml` keeps `baseurl: /website-preview` — don't copy that
  line here.
- **Rollback:** the pre-redesign site lives in git history (commit `c4ddf28`).
  Any bad change: `git revert` the commit and push.
- **Local theme overrides** (files that shadow the theme's versions — edit these,
  not the gems): `_includes/header.liquid` (navbar), `_includes/footer.liquid`
  (also disables single-$ math), `_includes/plugins/al_comments.liquid` (Cusdis),
  `_layouts/about.liquid`, `_layouts/bib.liquid` (publication entries),
  `_sass/_themes.scss` (colors, fonts, all custom styling).
- **Publication entry extras** (fields the theme understands in `papers.bib`):
  `selected={true}` home-page feature · `note={Summary: ...}` the italic line ·
  `pdf={papers/file.pdf}` PDF button · `award_name`/`award` the 🏆 button ·
  `abbr={Phys. Rev. Lett.}` the journal chip.
