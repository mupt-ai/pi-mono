import type { AgentTool } from "@mariozechner/pi-agent-core";
import { fauxAssistantMessage, fauxToolCall } from "@mariozechner/pi-ai";
import { Type } from "typebox";
import { afterEach, describe, expect, it } from "vitest";
import { createHarness, getAssistantTexts, getUserTexts, type Harness } from "./harness.js";

describe("AgentSession stepped execution", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		while (harnesses.length > 0) {
			harnesses.pop()?.cleanup();
		}
	});

	it("runs a plain prompt to completion through the stepped API", async () => {
		const harness = await createHarness();
		harnesses.push(harness);
		harness.setResponses([fauxAssistantMessage("hello")]);

		let state = await harness.session.initializeSessionLoopState("hi");

		let result = await harness.session.stepSessionLoop(state, { type: "prepare_prompt" });
		expect(result.nextAction).toBe("run_assistant_turn");
		state = result.state;

		result = await harness.session.stepSessionLoop(state, { type: "run_assistant_turn" });
		expect(result.nextAction).toBe("finalize_turn");
		state = result.state;

		result = await harness.session.stepSessionLoop(state, { type: "finalize_turn" });
		expect(result.nextAction).toBe("run_post_turn_effects");
		state = result.state;

		result = await harness.session.stepSessionLoop(state, { type: "run_post_turn_effects" });
		expect(result.nextAction).toBe("completed");
		expect(getUserTexts(harness)).toEqual(["hi"]);
		expect(getAssistantTexts(harness)).toEqual(["hello"]);
	});

	it("runs a tool turn and external tool execution through the stepped API", async () => {
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
		const harness = await createHarness({ tools: [echoTool] });
		harnesses.push(harness);
		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("echo", { text: "hello" }), { stopReason: "toolUse" }),
			fauxAssistantMessage("done"),
		]);

		let state = await harness.session.initializeSessionLoopState("start");
		let result = await harness.session.stepSessionLoop(state, { type: "prepare_prompt" });
		state = result.state;

		result = await harness.session.stepSessionLoop(state, { type: "run_assistant_turn" });
		expect(result.nextAction).toBe("prepare_tool_calls");
		state = result.state;

		result = await harness.session.stepSessionLoop(state, { type: "prepare_tool_calls" });
		expect(result.nextAction).toBe("complete_tool_call");
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
