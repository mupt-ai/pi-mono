# Fork Changelog

This file tracks fork-only changes for `mupt-ai/pi-mono`.

Keep upstream package changelogs in `packages/*/CHANGELOG.md` aligned with `badlogic/pi-mono`. Record fork-specific changes here instead.

## [Unreleased]

### Added

- `packages/agent`: Added atomic loop-state APIs via `initializeLoopState()` and `stepLoop()` for externally orchestrated assistant/tool boundaries without changing the existing `Agent`, `agentLoop()`, or `runAgentLoopContinue()` behavior.
- `packages/coding-agent`: Added stepped session SDK APIs for initializing and advancing `AgentSession` prompt execution one atomic boundary at a time.
- `packages/agent`, `packages/coding-agent`: Externalized stepped provider execution so assistant/tool boundaries can be driven outside the built-in provider loop.

### Changed

- `packages/agent`, `packages/coding-agent`: Forked package publishing to GitHub Packages under `@mupt-ai/pi-agent-core` and `@mupt-ai/pi-coding-agent`, and added a dedicated publish workflow for those packages.
- `README.md`: Documented the distributed Pi fork changes for this branch.

### Fixed

- `package-lock.json`: Restored cross-platform optional package entries required by Linux CI and Tailwind/watcher prebuild resolution after upstream merges.
