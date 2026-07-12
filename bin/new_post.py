#!/usr/bin/env python3
"""Scaffold a new blog post draft.

Usage: python3 bin/new_post.py "Post Title" [--tags "Tag One, Tag Two"] [--description "..."]

Creates _posts/YYYY-MM-DD-slug.md with the site's front-matter conventions
and published: false, so the draft is invisible until you remove that line.
"""
import argparse
import datetime
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("title")
    p.add_argument("--tags", default="")
    p.add_argument("--description", default="")
    a = p.parse_args()

    today = datetime.date.today()
    slug = re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", a.title.lower())).strip("-")
    slug = "-".join(slug.split("-")[:8])
    path = os.path.join(ROOT, "_posts", f"{today:%Y-%m-%d}-{slug}.md")
    if os.path.exists(path):
        print(f"Refusing to overwrite existing {path}")
        return 1

    tags = "".join(f"  - {t.strip()}\n" for t in a.tags.split(",") if t.strip())
    desc = a.description.strip() or "TODO: one-sentence description shown in the blog list and previews."
    body = f"""---
layout: post
title: '{a.title.replace("'", "''")}'
date: {today:%Y-%m-%d}
permalink: /posts/{today:%Y/%m}/{slug}/
description: "{desc}"
tags:
{tags or '  - TODO'}related_posts: true
published: false # remove this line to publish
---

TODO: write the post.

<!--
Conventions:
- Inline math: \\( E = mc^2 \\)   Display math: $$ ... $$
  (single $ is NOT math on this site, so dollar amounts are safe)
- Images: put files in assets/img/posts/ and embed with
  <img src="{{{{ '/assets/img/posts/FILE.png' | relative_url }}}}" alt="...">
- Comments are enabled automatically.
-->
"""
    open(path, "w").write(body)
    print(f"Created {os.path.relpath(path, ROOT)} (published: false)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
