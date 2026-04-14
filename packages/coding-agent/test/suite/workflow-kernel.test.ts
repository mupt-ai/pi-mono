import type { AgentTool, NormalizedAssistantMessageEvent } from "@mupt-ai/pi-agent-core";
import { Type } from "@sinclair/typebox";
import { afterEach, describe, expect, it } from "vitest";
import {
	captureSessionLogSnapshot,
	captureWorkflowEnvironmentSnapshot,
	initializeWorkflowState,
	stepWorkflowState,
	type WorkflowState,
} from "../../src/index.js";
import { assistantMsg, userMsg } from "../utilities.js";
import { createHarness, getAssistantTexts, getUserTexts, type Harness } from "./harness.js";

function createUsage() {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function createTextEvents(text: string): NormalizedAssistantMessageEvent[] {
	return [
		{ type: "start" },
		{ type: "text_start", contentIndex: 0 },
		{ type: "text_delta", contentIndex: 0, delta: text },
		{ type: "text_end", contentIndex: 0 },
		{ type: "done", reason: "stop", usage: createUsage() },
	];
}

function createToolCallEvents(
	toolCallId: string,
	toolName: string,
	argumentsObject: Record<string, unknown>,
): NormalizedAssistantMessageEvent[] {
	return [
		{ type: "start" },
		{ type: "toolcall_start", contentIndex: 0, id: toolCallId, toolName },
		{ type: "toolcall_delta", contentIndex: 0, delta: JSON.stringify(argumentsObject) },
		{ type: "toolcall_end", contentIndex: 0 },
		{ type: "done", reason: "toolUse", usage: createUsage() },
	];
}

describe("workflow kernel", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		while (harnesses.length > 0) {
			harnesses.pop()?.cleanup();
		}
	});

	it("rehydrates a provider turn from snapshots across workflow steps", async () => {
		const harness = await createHarness({
			providerExecutionMode: "external",
			withConfiguredAuth: false,
		});
		harnesses.push(harness);

		let sessionLog = captureSessionLogSnapshot(harness.sessionManager);
		const environment = captureWorkflowEnvironmentSnapshot(harness.session);
		let state = await initializeWorkflowState("hi", environment, sessionLog);

		let result = await stepWorkflowState(state, sessionLog, { type: "prepare_prompt" });
		expect(result.nextAction).toBe("run_assistant_turn");
		state = result.state;
		sessionLog = result.sessionLog;

		result = await stepWorkflowState(state, sessionLog, { type: "run_assistant_turn" });
		expect(result.nextAction).toBe("complete_provider_response");
		expect(result.preparedProviderRequest).toMatchObject({
			context: {
				messages: [
					{
						role: "user",
					},
				],
			},
		});
		expect(result.state.pendingProviderRequest).toBeDefined();
		state = result.state;
		sessionLog = result.sessionLog;

		result = await stepWorkflowState(state, sessionLog, {
			type: "complete_provider_response",
			events: createTextEvents("ok"),
		});
		expect(result.nextAction).toBe("finalize_turn");
		state = result.state;
		sessionLog = result.sessionLog;

		result = await stepWorkflowState(state, sessionLog, { type: "finalize_turn" });
		expect(result.nextAction).toBe("run_post_turn_effects");
		state = result.state;
		sessionLog = result.sessionLog;

		result = await stepWorkflowState(state, sessionLog, { type: "run_post_turn_effects" });
		expect(result.nextAction).toBe("completed");
		expect(getUserTexts(harness)).toEqual([]);
		expect(getAssistantTexts(harness)).toEqual([]);
		expect(sessionLog.entries.filter((entry) => entry.type === "message").map((entry) => entry.message.role)).toEqual(
			["user", "assistant"],
		);
	});

	it("rehydrates external tool calls from snapshots", async () => {
		const echoTool: AgentTool = {
			name: "echo",
			label: "Echo",
			description: "Echo text back",
			parameters: Type.Object({ text: Type.String() }),
			execute: async (_toolCallId, params) => {
				const text = typeof params === "object" && params !== null && "text" in params ? String(params.text) : "";
				return {
					content: [{ type: "text", text: `echo:${text}` }],
					details: { text },
				};
			},
		};
		const harness = await createHarness({
			tools: [echoTool],
			providerExecutionMode: "external",
			withConfiguredAuth: false,
		});
		harnesses.push(harness);

		let sessionLog = captureSessionLogSnapshot(harness.sessionManager);
		const environment = captureWorkflowEnvironmentSnapshot(harness.session);
		let state = await initializeWorkflowState("start", environment, sessionLog);

		let result = await stepWorkflowState(state, sessionLog, { type: "prepare_prompt" });
		state = result.state;
		sessionLog = result.sessionLog;

		result = await stepWorkflowState(state, sessionLog, { type: "run_assistant_turn" });
		expect(result.nextAction).toBe("complete_provider_response");
		state = result.state;
		sessionLog = result.sessionLog;

		result = await stepWorkflowState(state, sessionLog, {
			type: "complete_provider_response",
			events: createToolCallEvents("tool-1", "echo", { text: "hello" }),
		});
		expect(result.nextAction).toBe("prepare_tool_calls");
		state = result.state;
		sessionLog = result.sessionLog;

		result = await stepWorkflowState(state, sessionLog, { type: "prepare_tool_calls" });
		expect(result.nextAction).toBe("complete_tool_call");
		expect(result.state.pendingToolCalls).toHaveLength(1);
		const request = result.toolExecutionRequests?.[0];
		expect(request?.toolName).toBe("echo");
		state = result.state;
		sessionLog = result.sessionLog;

		const toolResult = await echoTool.execute(request!.toolCallId, request!.args as { text: string });
		result = await stepWorkflowState(state, sessionLog, {
			type: "complete_tool_call",
			toolCallId: request!.toolCallId,
			result: toolResult,
			isError: false,
		});
		expect(result.nextAction).toBe("finalize_turn");
		state = result.state;
		sessionLog = result.sessionLog;

		result = await stepWorkflowState(state, sessionLog, { type: "finalize_turn" });
		expect(result.nextAction).toBe("run_assistant_turn");
		state = result.state;
		sessionLog = result.sessionLog;

		result = await stepWorkflowState(state, sessionLog, { type: "run_assistant_turn" });
		expect(result.nextAction).toBe("complete_provider_response");
		state = result.state;
		sessionLog = result.sessionLog;

		result = await stepWorkflowState(state, sessionLog, {
			type: "complete_provider_response",
			events: createTextEvents("done"),
		});
		state = result.state;
		sessionLog = result.sessionLog;

		result = await stepWorkflowState(state, sessionLog, { type: "finalize_turn" });
		state = result.state;
		sessionLog = result.sessionLog;

		result = await stepWorkflowState(state, sessionLog, { type: "run_post_turn_effects" });
		expect(result.nextAction).toBe("completed");
		expect(sessionLog.entries.filter((entry) => entry.type === "message").map((entry) => entry.message.role)).toEqual(
			["user", "assistant", "toolResult", "assistant"],
		);
	});

	it("prepares and completes compaction externally", async () => {
		const harness = await createHarness({
			settings: {
				compaction: {
					enabled: true,
					reserveTokens: 64,
					keepRecentTokens: 0,
				},
			},
		});
		harnesses.push(harness);

		harness.sessionManager.appendMessage(userMsg("first"));
		harness.sessionManager.appendMessage(assistantMsg("reply"));
		harness.sessionManager.appendMessage(userMsg("second"));
		harness.sessionManager.appendMessage(assistantMsg("more"));

		let sessionLog = captureSessionLogSnapshot(harness.sessionManager);
		const environment = captureWorkflowEnvironmentSnapshot(harness.session);
		const initialState = await initializeWorkflowState("next", environment, sessionLog);
		let state: WorkflowState = {
			...initialState,
			phase: "awaiting_compaction" as const,
			terminalStatus: "running" as const,
			compactionRequest: { reason: "threshold" as const, willRetry: false },
			lastAssistantMessage: assistantMsg("more"),
		};

		let result = await stepWorkflowState(state, sessionLog, { type: "prepare_compaction" });
		expect(result.nextAction).toBe("complete_compaction");
		expect(result.preparedCompactionRequest).toBeDefined();
		expect(result.state.pendingCompaction).toBeDefined();
		state = result.state;
		sessionLog = result.sessionLog;

		const request = result.preparedCompactionRequest!;
		result = await stepWorkflowState(state, sessionLog, {
			type: "complete_compaction",
			result: {
				summary: "Compacted history",
				firstKeptEntryId: request.preparation.firstKeptEntryId,
				tokensBefore: request.preparation.tokensBefore,
				details: { readFiles: [], modifiedFiles: [] },
			},
		});
		expect(result.nextAction).toBe("completed");
		expect(result.sessionOps).toEqual([
			{
				type: "append_compaction",
				summary: "Compacted history",
				firstKeptEntryId: request.preparation.firstKeptEntryId,
				tokensBefore: request.preparation.tokensBefore,
				details: { readFiles: [], modifiedFiles: [] },
				fromExtension: undefined,
			},
		]);
		expect(result.sessionLog.entries.at(-1)?.type).toBe("compaction");
	});

	it("rejects sessions with unsupported direct Agent hook mutations", async () => {
		const harness = await createHarness();
		harnesses.push(harness);

		harness.session.agent.beforeToolCall = async () => ({ block: true, reason: "skip" });

		expect(() => captureWorkflowEnvironmentSnapshot(harness.session)).toThrow(
			"Workflow snapshots do not support custom direct Agent hooks: beforeToolCall",
		);
	});
});
