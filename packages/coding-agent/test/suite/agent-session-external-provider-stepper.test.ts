import type { AgentTool, NormalizedAssistantMessageEvent } from "@mariozechner/pi-agent-core";
import { fauxAssistantMessage } from "@mariozechner/pi-ai";
import { Type } from "typebox";
import { afterEach, describe, expect, it } from "vitest";
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

describe("AgentSession external provider stepper", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		while (harnesses.length > 0) {
			harnesses.pop()?.cleanup();
		}
	});

	it("returns a prepared provider request and matches inline events after response application", async () => {
		const inlineHarness = await createHarness({
			fauxTokenSize: { min: 100, max: 100 },
		});
		harnesses.push(inlineHarness);
		inlineHarness.setResponses([fauxAssistantMessage("ok")]);

		const externalHarness = await createHarness({
			providerExecutionMode: "external",
			withConfiguredAuth: false,
		});
		harnesses.push(externalHarness);

		const inlineState = await inlineHarness.session.initializeSessionLoopState("hi");
		let inlineResult = await inlineHarness.session.stepSessionLoop(inlineState, { type: "prepare_prompt" });
		inlineResult = await inlineHarness.session.stepSessionLoop(inlineResult.state, { type: "run_assistant_turn" });

		let externalState = await externalHarness.session.initializeSessionLoopState("hi");
		const externalPrepare = await externalHarness.session.stepSessionLoop(externalState, { type: "prepare_prompt" });
		externalState = externalPrepare.state;

		const externalRun = await externalHarness.session.stepSessionLoop(externalState, { type: "run_assistant_turn" });
		expect(externalRun.nextAction).toBe("complete_provider_response");
		expect(externalRun.preparedProviderRequest).toMatchObject({
			context: {
				systemPrompt: externalHarness.session.systemPrompt,
				messages: [
					{
						role: "user",
					},
				],
			},
			options: {},
		});
		expect("apiKey" in (externalRun.preparedProviderRequest?.options ?? {})).toBe(false);
		expect("signal" in (externalRun.preparedProviderRequest?.options ?? {})).toBe(false);

		const externalComplete = await externalHarness.session.stepSessionLoop(externalRun.state, {
			type: "complete_provider_response",
			events: createTextEvents("ok"),
		});

		expect(externalComplete.nextAction).toBe("finalize_turn");
		expect([...externalRun.coreEvents, ...externalComplete.coreEvents].map((event) => event.type)).toEqual(
			inlineResult.coreEvents.map((event) => event.type),
		);
		expect([...externalRun.sessionEvents, ...externalComplete.sessionEvents]).toEqual(inlineResult.sessionEvents);

		let externalFinalize = await externalHarness.session.stepSessionLoop(externalComplete.state, {
			type: "finalize_turn",
		});
		expect(externalFinalize.nextAction).toBe("run_post_turn_effects");
		externalFinalize = await externalHarness.session.stepSessionLoop(externalFinalize.state, {
			type: "run_post_turn_effects",
		});
		expect(externalFinalize.nextAction).toBe("completed");
		expect(getUserTexts(externalHarness)).toEqual(["hi"]);
		expect(getAssistantTexts(externalHarness)).toEqual(["ok"]);
	});

	it("keeps external tool execution flow unchanged after applying a host-supplied provider response", async () => {
		const toolRuns: string[] = [];
		const echoTool: AgentTool = {
			name: "echo",
			label: "Echo",
			description: "Echo text back",
			parameters: Type.Object({ text: Type.String() }),
			execute: async (_toolCallId, params) => {
				const text = typeof params === "object" && params !== null && "text" in params ? String(params.text) : "";
				toolRuns.push(text);
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

		let state = await harness.session.initializeSessionLoopState("start");
		let result = await harness.session.stepSessionLoop(state, { type: "prepare_prompt" });
		state = result.state;

		result = await harness.session.stepSessionLoop(state, { type: "run_assistant_turn" });
		expect(result.nextAction).toBe("complete_provider_response");
		state = result.state;

		result = await harness.session.stepSessionLoop(state, {
			type: "complete_provider_response",
			events: createToolCallEvents("tool-1", "echo", { text: "hello" }),
		});
		expect(result.nextAction).toBe("prepare_tool_calls");
		state = result.state;

		result = await harness.session.stepSessionLoop(state, { type: "prepare_tool_calls" });
		expect(result.nextAction).toBe("complete_tool_call");
		expect(result.toolExecutionRequests).toHaveLength(1);
		const request = result.toolExecutionRequests?.[0];
		expect(request?.toolName).toBe("echo");
		state = result.state;

		const toolResult = await echoTool.execute(request!.toolCallId, request!.args as { text: string });
		result = await harness.session.stepSessionLoop(state, {
			type: "complete_tool_call",
			toolCallId: request!.toolCallId,
			result: toolResult,
			isError: false,
		});
		expect(result.nextAction).toBe("finalize_turn");
		state = result.state;

		result = await harness.session.stepSessionLoop(state, { type: "finalize_turn" });
		expect(result.nextAction).toBe("run_assistant_turn");
		state = result.state;

		result = await harness.session.stepSessionLoop(state, { type: "run_assistant_turn" });
		expect(result.nextAction).toBe("complete_provider_response");
		state = result.state;

		result = await harness.session.stepSessionLoop(state, {
			type: "complete_provider_response",
			events: createTextEvents("done"),
		});
		expect(result.nextAction).toBe("finalize_turn");
		state = result.state;

		result = await harness.session.stepSessionLoop(state, { type: "finalize_turn" });
		expect(result.nextAction).toBe("run_post_turn_effects");
		state = result.state;

		result = await harness.session.stepSessionLoop(state, { type: "run_post_turn_effects" });
		expect(result.nextAction).toBe("completed");
		expect(toolRuns).toEqual(["hello"]);
		expect(harness.session.messages.map((message) => message.role)).toEqual([
			"user",
			"assistant",
			"toolResult",
			"assistant",
		]);
	});
});
