<!-- OSS_WEEKEND_START -->
# 🏖️ OSS Weekend

**Issue tracker reopens Monday, April 13, 2026.**

OSS weekend runs Thursday, April 2, 2026 through Monday, April 13, 2026. New issues and PRs from unapproved contributors are auto-closed during this time. Approved contributors can still open issues and PRs if something is genuinely urgent, but please keep that to pressing matters only. For support, join [Discord](https://discord.com/invite/3cU7Bz4UPx).

> _Current focus: at the moment i'm deep in refactoring internals, and need to focus._
<!-- OSS_WEEKEND_END -->

# Distributed pi

This repo is a fork of [badlogic/pi-mono](https://github.com/badlogic/pi-mono). For the original project overview, package docs, installation, and standard workflow, see the upstream [README](https://github.com/badlogic/pi-mono/blob/main/README.md).

This README only documents the fork-specific changes here.

This fork is focused on distributed execution: assistant turns can be stepped externally, LLM requests can run off-device, and tool execution can also happen off-device and then be fed back into the agent loop.

## What Changed

### External provider execution

- Added `ProviderExecutionMode = "inline" | "external"` for stepped agent and session execution.
- Added `prepareAssistantProviderRequest()` to build the provider request without executing the LLM call inside the loop.
- Added `applyAssistantProviderResponse()` to apply normalized streamed assistant events back into the loop after the external caller finishes the provider request.
- Added `packages/agent/src/normalized-assistant-events.ts` so assistant streaming events can be transported across process or device boundaries.

### External session stepping APIs

- Added `createSessionStepRuntime()`.
- Added `initializeSessionLoopState()`.
- Added `stepSessionLoop()`.

### Off-device tool execution

- In external stepped mode, tool calls are surfaced as `toolExecutionRequests`.
- Tool results are returned to the loop with the `complete_tool_call` step command, so tools can run on another machine or service.

## External Flow

1. Call `createSessionStepRuntime({ providerExecutionMode: "external" })`.
2. Create loop state with `initializeSessionLoopState(...)`.
3. Advance the loop with `stepSessionLoop(...)`.
4. When the loop returns `preparedProviderRequest`, execute the LLM call externally.
5. Feed normalized provider events back with `complete_provider_response`.
6. When the loop returns `toolExecutionRequests`, execute those tools externally.
7. Feed tool results back with `complete_tool_call`.
