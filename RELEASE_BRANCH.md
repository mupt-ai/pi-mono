# Mupt release branch

This branch is intentionally separate from upstream contribution work.

## Branch roles

- `main` mirrors `upstream/main` exactly.
- `mupt-release` is `upstream/main` plus one release-only commit:
  - package rename/publish metadata for GitHub Packages
  - `.github/workflows/publish-packages.yml`
  - `scripts/version-github-packages.mjs`
- PR stack branches stay separate:
  - `avyay/upstream-pr1-core-stepped-loop`
  - `avyay/upstream-pr2-session-stepped-loop`
  - `avyay/upstream-pr3-session-snapshot`

Do not merge feature/code work into `mupt-release` permanently.

## Build a private release with the current PR stack

Use a disposable branch:

```bash
git fetch origin upstream

git switch -c mupt-release-build origin/mupt-release
git merge --no-ff origin/avyay/upstream-pr3-session-snapshot
npm run check
```

Release from `mupt-release-build`. Delete/recreate it as needed.

`origin/avyay/upstream-pr3-session-snapshot` is the top of the stack and includes PR1 and PR2.
