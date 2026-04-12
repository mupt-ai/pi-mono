import {
	type AssistantMessage,
	type AssistantMessageEvent,
	type Context,
	EventStream,
	type Message,
	type Model,
	type UserMessage,
} from "@mariozechner/pi-ai";
import { Type } from "@sinclair/typebox";
import { afterEach, describe, expect, it, vi } from "vitest";
import { agentLoop, initializeLoopState, stepLoop } from "../src/agent-loop.js";
import type { NormalizedAssistantMessageEvent } from "../src/normalized-assistant-events.js";
import { streamProxy } from "../src/proxy.js";
import type {
	AgentContext,
	AgentEvent,
	AgentLoopConfig,
	AgentMessage,
	AgentTool,
	AgentToolResult,
	StepLoopRuntime,
	ToolExecutionRequest,
} from "../src/types.js";

class MockAssistantStream extends EventStream<AssistantMessageEvent, AssistantMessage> {
	constructor() {
		super(
			(event) => event.type === "done" || event.type === "error",
			(event) => {
				if (event.type === "done") return event.message;
				if (event.type === "error") return event.error;
				throw new Error("Unexpected event type");
			},
		);
	}
}

function createUsage(): AssistantMessage["usage"] {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function createModel(): Model<"openai-responses"> {
	return {
		id: "mock",
		name: "mock",
		api: "openai-responses",
		provider: "openai",
		baseUrl: "https://example.invalid",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 8192,
		maxTokens: 2048,
	};
}

function createAssistantMessage(
	content: AssistantMessage["content"],
	stopReason: AssistantMessage["stopReason"] = "stop",
): AssistantMessage {
	return {
		role: "assistant",
		content,
		api: "openai-responses",
		provider: "openai",
		model: "mock",
		usage: createUsage(),
		stopReason,
		timestamp: Date.now(),
	};
}

function createUserMessage(text: string): UserMessage {
	return {
		role: "user",
		content: text,
		timestamp: Date.now(),
	};
}

function identityConverter(messages: AgentMessage[]): Message[] {
	return messages.filter(
		(message) => message.role === "user" || message.role === "assistant" || message.role === "toolResult",
	) as Message[];
}

function normalizeMessages(messages: AgentMessage[]): AgentMessage[] {
	return messages.map((message) => {
		if (!("timestamp" in message)) {
			return message;
		}
		return { ...message, timestamp: 0 };
	});
}

function createStreamingTextStreamFn(text: string) {
	return () => {
		const stream = new MockAssistantStream();
		queueMicrotask(() => {
			const partial = createAssistantMessage([]);
			stream.push({ type: "start", partial });
			partial.content[0] = { type: "text", text: "" };
			stream.push({ type: "text_start", contentIndex: 0, partial: { ...partial } });
			(partial.content[0] as Extract<AssistantMessage["content"][number], { type: "text" }>).text += text;
			stream.push({ type: "text_delta", contentIndex: 0, delta: text, partial: { ...partial } });
			stream.push({ type: "text_end", contentIndex: 0, content: text, partial: { ...partial } });
			stream.push({
				type: "done",
				reason: "stop",
				message: createAssistantMessage([{ type: "text", text }]),
			});
		});
		return stream;
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

async function executeToolRequestExternally(
	request: ToolExecutionRequest,
	tools: AgentTool<any>[] | undefined,
): Promise<{ toolCallId: string; result: AgentToolResult<unknown>; isError: boolean }> {
	const tool = tools?.find((candidate) => candidate.name === request.toolName);
	if (!tool) {
		throw new Error(`Tool not found: ${request.toolName}`);
	}

	try {
		const result = await tool.execute(request.toolCallId, request.args as never);
		return {
			toolCallId: request.toolCallId,
			result,
			isError: false,
		};
	} catch (error) {
		return {
			toolCallId: request.toolCallId,
			result: {
				content: [{ type: "text", text: error instanceof Error ? error.message : String(error) }],
				details: {},
			},
			isError: true,
		};
	}
}

function createSseResponse(events: NormalizedAssistantMessageEvent[]): Response {
	const encoder = new TextEncoder();
	return new Response(
		new ReadableStream({
			start(controller) {
				for (const event of events) {
					controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n`));
				}
				controller.close();
			},
		}),
		{ status: 200, headers: { "Content-Type": "text/event-stream" } },
	);
}

afterEach(() => {
	vi.unstubAllGlobals();
	vi.restoreAllMocks();
});

describe("external provider stepper mode", () => {
	it("returns a prepared request and reproduces the inline transcript after response completion", async () => {
		const context: AgentContext = {
			systemPrompt: "You are helpful.",
			messages: [],
			tools: [],
		};
		const userPrompt: AgentMessage = createUserMessage("Hello");
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			reasoning: "low",
		};

		const baselineEvents: AgentEvent[] = [];
		const baselineStream = agentLoop([userPrompt], context, config, undefined, createStreamingTextStreamFn("Hi"));
		for await (const event of baselineStream) {
			baselineEvents.push(event);
		}
		const baselineMessages = await baselineStream.result();

		const runtime: StepLoopRuntime = {
			config,
			tools: context.tools,
			providerExecutionMode: "external",
		};

		let result = await stepLoop(
			initializeLoopState([userPrompt], context, config),
			{ type: "run_assistant_turn" },
			runtime,
		);
		expect(result.nextAction).toBe("complete_provider_response");
		expect(result.preparedProviderRequest).toMatchObject({
			model: createModel(),
			context: {
				systemPrompt: "You are helpful.",
				messages: [userPrompt],
			},
			options: {
				reasoning: "low",
			},
		});
		expect("apiKey" in (result.preparedProviderRequest?.options ?? {})).toBe(false);
		expect("signal" in (result.preparedProviderRequest?.options ?? {})).toBe(false);

		const stepEvents: AgentEvent[] = [...result.events];
		result = await stepLoop(
			result.state,
			{ type: "complete_provider_response", events: createTextEvents("Hi") },
			runtime,
		);
		stepEvents.push(...result.events);
		expect(result.nextAction).toBe("finalize_turn");

		result = await stepLoop(result.state, { type: "finalize_turn" }, runtime);
		stepEvents.push(...result.events);
		expect(result.nextAction).toBe("check_follow_up");

		result = await stepLoop(result.state, { type: "check_follow_up" }, runtime);
		stepEvents.push(...result.events);
		expect(result.nextAction).toBe("completed");
		expect(normalizeMessages(result.terminalMessages ?? [])).toEqual(normalizeMessages(baselineMessages));
		expect(stepEvents.map((event) => event.type)).toEqual(baselineEvents.map((event) => event.type));
	});

	it("transitions to tool preparation when a completed provider response includes tool calls", async () => {
		const toolSchema = Type.Object({ text: Type.String() });
		const tool: AgentTool<typeof toolSchema, { text: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo text back",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				return {
					content: [{ type: "text", text: `echo:${params.text}` }],
					details: { text: params.text },
				};
			},
		};
		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
		};
		const runtime: StepLoopRuntime = {
			config,
			tools: context.tools,
			providerExecutionMode: "external",
		};

		let result = await stepLoop(
			initializeLoopState([createUserMessage("Use the tool")], context, config),
			{ type: "run_assistant_turn" },
			runtime,
		);
		expect(result.nextAction).toBe("complete_provider_response");
		expect(result.preparedProviderRequest?.context.tools).toEqual([
			{
				name: "echo",
				description: "Echo text back",
				parameters: toolSchema,
			},
		]);

		result = await stepLoop(
			result.state,
			{
				type: "complete_provider_response",
				events: createToolCallEvents("tool-1", "echo", { text: "hello" }),
			},
			runtime,
		);
		expect(result.nextAction).toBe("prepare_tool_calls");

		result = await stepLoop(result.state, { type: "prepare_tool_calls" }, runtime);
		expect(result.nextAction).toBe("complete_tool_call");
		expect(result.toolExecutionRequests).toHaveLength(1);
		expect(result.toolExecutionRequests?.[0]).toMatchObject({
			toolCallId: "tool-1",
			toolName: "echo",
			args: { text: "hello" },
		});

		const execution = await executeToolRequestExternally(
			result.toolExecutionRequests?.[0] as ToolExecutionRequest,
			context.tools,
		);
		result = await stepLoop(
			result.state,
			{
				type: "complete_tool_call",
				toolCallId: execution.toolCallId,
				result: execution.result,
				isError: execution.isError,
			},
			runtime,
		);
		expect(result.nextAction).toBe("finalize_turn");
	});

	it("fails clearly when the normalized provider response is an error", async () => {
		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [],
		};
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
		};
		const runtime: StepLoopRuntime = {
			config,
			tools: context.tools,
			providerExecutionMode: "external",
		};

		let result = await stepLoop(
			initializeLoopState([createUserMessage("Hello")], context, config),
			{ type: "run_assistant_turn" },
			runtime,
		);
		result = await stepLoop(
			result.state,
			{
				type: "complete_provider_response",
				events: [{ type: "error", reason: "error", errorMessage: "upstream failed", usage: createUsage() }],
			},
			runtime,
		);

		expect(result.nextAction).toBe("failed");
		expect(result.state.phase).toBe("failed");
		expect(result.state.terminalStatus).toBe("failed");
		expect(result.terminalMessages?.at(-1)).toMatchObject({
			role: "assistant",
			stopReason: "error",
			errorMessage: "upstream failed",
		});
	});

	it("rejects complete_provider_response outside the provider-response phase", async () => {
		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [],
		};
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
		};

		await expect(
			stepLoop(
				initializeLoopState([createUserMessage("Hello")], context, config),
				{ type: "complete_provider_response", events: createTextEvents("Hi") },
				{ config, tools: context.tools, providerExecutionMode: "external" },
			),
		).rejects.toThrow("Cannot complete_provider_response while loop phase is awaiting_assistant");
	});
});

describe("streamProxy", () => {
	it("reconstructs streamed assistant content from normalized proxy events", async () => {
		const events = createTextEvents("proxy ok");
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => createSseResponse(events)),
		);

		const stream = streamProxy(createModel(), { messages: [], systemPrompt: "proxy" } satisfies Context, {
			authToken: "token",
			proxyUrl: "https://proxy.example.test",
		});

		const eventTypes: AssistantMessageEvent["type"][] = [];
		for await (const event of stream) {
			eventTypes.push(event.type);
		}
		const finalMessage = await stream.result();

		expect(eventTypes).toEqual(["start", "text_start", "text_delta", "text_end", "done"]);
		expect(finalMessage).toMatchObject({
			role: "assistant",
			content: [{ type: "text", text: "proxy ok" }],
			stopReason: "stop",
		});
	});
});
