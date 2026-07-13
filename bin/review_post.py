#!/usr/bin/env python3
"""Run the pre-publication review on a blog post via the Claude API.

Usage: ANTHROPIC_API_KEY=... python3 bin/review_post.py <post-slug-or-filename>

Finds the post in _posts/ (any unique filename substring works), sends it to
Claude with docs/POST_REVIEW_PROMPT.md, and writes the review to
review-output.md. Exits with a clear message if the API key is missing.
"""
import json
import os
import re
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("REVIEW_MODEL", "claude-sonnet-5")


def main():
    if len(sys.argv) != 2:
        print("usage: review_post.py <post-slug-or-filename>")
        return 2

    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("ANTHROPIC_API_KEY is not set.")
        print("Add it once: repo Settings -> Secrets and variables -> Actions ->")
        print("  New repository secret -> name ANTHROPIC_API_KEY (key from console.anthropic.com).")
        print("Until then, review manually: paste docs/POST_REVIEW_PROMPT.md plus")
        print("the post into claude.ai.")
        return 1

    needle = sys.argv[1].strip()
    posts = [f for f in sorted(os.listdir(os.path.join(ROOT, "_posts")))
             if needle in f and f.endswith(".md")]
    if len(posts) != 1:
        print(f"Expected exactly one post matching '{needle}', found: {posts or 'none'}")
        print("Available posts:")
        for f in sorted(os.listdir(os.path.join(ROOT, "_posts"))):
            print("  ", f)
        return 1

    post_path = os.path.join(ROOT, "_posts", posts[0])
    post = open(post_path).read()
    prompt_doc = open(os.path.join(ROOT, "docs", "POST_REVIEW_PROMPT.md")).read()
    # everything below the first horizontal rule is the prompt proper
    prompt = prompt_doc.split("\n---\n", 1)[1]

    body = json.dumps({
        "model": MODEL,
        "max_tokens": 8000,
        "messages": [{
            "role": "user",
            "content": prompt + "\n\n---\n\nThe draft post to review:\n\n" + post,
        }],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        })
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.load(r)

    review = "".join(b.get("text", "") for b in resp.get("content", []))
    header = f"# Review of `{posts[0]}`\n\n_Model: {MODEL}_\n\n"
    open(os.path.join(ROOT, "review-output.md"), "w").write(header + review)
    print(f"Review written ({len(review)} chars). Verdict line:")
    m = re.search(r"\*\*Verdict:?\*\*.*", review)
    print(m.group(0) if m else "(no verdict line found)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
