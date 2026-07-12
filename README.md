# chaojiezhang.me — preview

Preview of the redesigned personal website of [Chaojie Zhang](https://chaojiezhang.me), built on the [al-folio](https://github.com/alshedivat/al-folio) v1 theme.

- **Preview URL:** https://chaojie-zhang.github.io/website-preview/
- **Production repo:** [Chaojie-Zhang/Chaojie-Zhang.github.io](https://github.com/Chaojie-Zhang/Chaojie-Zhang.github.io)
- Deploys automatically via GitHub Actions (`.github/workflows/deploy.yml`) to the `gh-pages` branch on every push to `master`.

When promoting to production: set `baseurl: ""` and `url: https://chaojiezhang.me` in `_config.yml`, add the `CNAME` file, and push to the production repo.

## Local overrides of theme gems

- `_includes/plugins/al_comments.liquid` — replaces the al_comments (giscus/disqus) wrapper with [Cusdis](https://cusdis.com) accountless comments, synced to the site's dark/light theme.
