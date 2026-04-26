/**
 * Agent loop that works with AgentMessage throughout.
 * Transforms to Message[] only at the LLM call boundary.
 */

import {
	type AssistantMessage,
	type Context,
	EventStream,
	streamSimple,
	type ToolResultMessage,
	validateToolArguments,
} from "@mariozechner/pi-ai";
import type {
	AgentContext,
	AgentEvent,
	AgentLoopConfig,
	AgentMessage,
	AgentTool,
	AgentToolCall,
	AgentToolResult,
	CompletedToolCallSnapshot,
	LoopState,
	PreparedToolCallSnapshot,
	StepCommand,
	StepLoopRuntime,
	StepResult,
	StreamFn,
	ToolExecutionRequest,
} from "./types.js";

export type AgentEventSink = (event: AgentEvent) => Promise<void> | void;

/**
 * Start an agent loop with a new prompt message.
 * The prompt is added to the context and events are emitted for it.
 */
export function agentLoop(
	prompts: AgentMessage[],
	context: AgentContext,
	config: AgentLoopConfig,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): EventStream<AgentEvent, AgentMessage[]> {
	const stream = createAgentStream();

	void runAgentLoop(
		prompts,
		context,
		config,
		async (event) => {
			stream.push(event);
		},
		signal,
		streamFn,
	).then((messages) => {
		stream.end(messages);
	});

	return stream;
}

/**
 * Continue an agent loop from the current context without adding a new message.
 * Used for retries - context already has user message or tool results.
 *
 * **Important:** The last message in context must convert to a `user` or `toolResult` message
 * via `convertToLlm`. If it doesn't, the LLM provider will reject the request.
 * This cannot be validated here since `convertToLlm` is only called once per turn.
 */
export function agentLoopContinue(
	context: AgentContext,
	config: AgentLoopConfig,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): EventStream<AgentEvent, AgentMessage[]> {
	if (context.messages.length === 0) {
		throw new Error("Cannot continue: no messages in context");
	}

	if (context.messages[context.messages.length - 1].role === "assistant") {
		throw new Error("Cannot continue from message role: assistant");
	}

	const stream = createAgentStream();

	void runAgentLoopContinue(
		context,
		config,
		async (event) => {
			stream.push(event);
		},
		signal,
		streamFn,
	).then((messages) => {
		stream.end(messages);
	});

	return stream;
}

export async function runAgentLoop(
	prompts: AgentMessage[],
	context: AgentContext,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): Promise<AgentMessage[]> {
	const newMessages: AgentMessage[] = [...prompts];
	const currentContext: AgentContext = {
		...context,
		messages: [...context.messages, ...prompts],
	};

	await emit({ type: "agent_start" });
	await emit({ type: "turn_start" });
	for (const prompt of prompts) {
		await emit({ type: "message_start", message: prompt });
		await emit({ type: "message_end", message: prompt });
	}

	await runLoop(currentContext, newMessages, config, signal, emit, streamFn);
	return newMessages;
}

export async function runAgentLoopContinue(
	context: AgentContext,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): Promise<AgentMessage[]> {
	if (context.messages.length === 0) {
		throw new Error("Cannot continue: no messages in context");
	}

	if (context.messages[context.messages.length - 1].role === "assistant") {
		throw new Error("Cannot continue from message role: assistant");
	}

	const newMessages: AgentMessage[] = [];
	const currentContext: AgentContext = { ...context };

	await emit({ type: "agent_start" });
	await emit({ type: "turn_start" });

	await runLoop(currentContext, newMessages, config, signal, emit, streamFn);
	return newMessages;
}

/**
 * Build the initial `LoopState` for a stepped agent loop.
 *
 * `prompts` are appended to `context.messages` and tracked as both
 * `newMessages` and `pendingPromptMessages` so the first step emits their
 * `message_start`/`message_end` events. The state starts in
 * `awaiting_assistant` with `firstTurn` set, ready for a `run_assistant_turn`
 * command.
 */
export function initializeLoopState(
	prompts: AgentMessage[],
	context: AgentContext,
	config?: Pick<AgentLoopConfig, "toolExecution">,
): LoopState {
	return {
		systemPrompt: context.systemPrompt,
		messages: [...context.messages, ...prompts],
		newMessages: [...prompts],
		phase: "awaiting_assistant",
		toolExecution: config?.toolExecution ?? "parallel",
		pendingPromptMessages: [...prompts],
		pendingSteeringMessages: [],
		pendingFollowUpMessages: [],
		pendingToolCalls: [],
		preparedToolCalls: [],
		executedToolCalls: [],
		completedToolResults: [],
		currentAssistantMessage: undefined,
		firstTurn: true,
		initialSteeringChecked: false,
		terminalStatus: "running",
	};
}

/**
 * Advance the stepped agent loop by exactly one command.
 *
 * The input `state` is cloned before mutation so the caller can keep the
 * previous snapshot. The dispatched step routine emits agent events, applies
 * them to the cloned state, and returns a `StepResult` with the next action
 * the host should perform plus any external work payloads (tool calls,
 * terminal messages).
 *
 * The legal `command.type` for each phase is enforced via `assertPhase()` in
 * the dispatched step routines — passing a command that does not match the
 * current phase throws.
 */
export async function stepLoop(
	state: LoopState,
	command: StepCommand,
	runtime: StepLoopRuntime,
	signal?: AbortSignal,
): Promise<StepResult> {
	const nextState = cloneLoopState(state);
	const events: AgentEvent[] = [];
	const emit: AgentEventSink = (event) => {
		events.push(event);
	};

	switch (command.type) {
		case "run_assistant_turn":
			return runAssistantTurnStep(nextState, runtime, signal, emit, events);
		case "prepare_tool_calls":
			return prepareToolCallsStep(nextState, runtime, signal, emit, events);
		case "complete_tool_call":
			return completeToolCallStep(nextState, command, runtime, signal, emit, events);
		case "finalize_turn":
			return finalizeTurnStep(nextState, runtime, emit, events);
		case "check_follow_up":
			return checkFollowUpStep(nextState, runtime, emit, events);
	}
}

async function runAssistantTurnStep(
	state: LoopState,
	runtime: StepLoopRuntime,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	events: AgentEvent[],
): Promise<StepResult> {
	assertPhase(state, "awaiting_assistant", "run_assistant_turn");

	if (state.firstTurn) {
		await emit({ type: "agent_start" });
	}
	await emit({ type: "turn_start" });
	await emitPendingPromptMessages(state, emit);

	if (!state.initialSteeringChecked) {
		state.pendingSteeringMessages = (await runtime.config.getSteeringMessages?.()) || [];
		state.initialSteeringChecked = true;
	}

	await injectPendingMessages(state, state.pendingSteeringMessages, emit);
	state.pendingSteeringMessages = [];
	await injectPendingMessages(state, state.pendingFollowUpMessages, emit);
	state.pendingFollowUpMessages = [];

	const context = createLoopContext(state, runtime.tools);

	let providerRequestPayload: unknown;
	const config = wrapConfigWithPayloadCapture(runtime.config, (payload) => {
		providerRequestPayload = payload;
	});

	state.phase = "assistant_streaming";
	const message = await streamAssistantResponse(context, config, signal, emit, runtime.streamFn);
	return advanceAfterAssistantMessage(state, message, emit, events, { providerRequestPayload });
}

async function prepareToolCallsStep(
	state: LoopState,
	runtime: StepLoopRuntime,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	events: AgentEvent[],
): Promise<StepResult> {
	assertPhase(state, "awaiting_tool_preflight", "prepare_tool_calls");

	const assistantMessage = state.currentAssistantMessage;
	if (!assistantMessage) {
		throw new Error("Cannot prepare tool calls without an assistant message");
	}

	if (state.pendingToolCalls.length === 0) {
		state.phase = "awaiting_turn_close";
		return createStepResult(state, events, "finalize_turn");
	}

	if (state.toolExecution === "parallel") {
		const toolExecutionRequests: ToolExecutionRequest[] = [];
		while (state.pendingToolCalls.length > 0) {
			const toolCall = state.pendingToolCalls.shift();
			if (!toolCall) {
				break;
			}

			await emit({
				type: "tool_execution_start",
				toolCallId: toolCall.id,
				toolName: toolCall.name,
				args: toolCall.arguments,
			});

			const preparation = await prepareToolCall(
				createLoopContext(state, runtime.tools),
				assistantMessage,
				toolCall,
				runtime.config,
				signal,
			);

			if (preparation.kind === "immediate") {
				const finalized: FinalizedToolCallOutcome = {
					toolCall,
					result: preparation.result,
					isError: preparation.isError,
				};
				await emitToolExecutionEnd(finalized, emit);
				const message = createToolResultMessage(finalized);
				await emitToolResultMessage(message, emit);
				state.completedToolResults.push(
					createCompletedToolCallSnapshot(toolCall, undefined, finalized.result, finalized.isError, message),
				);
				continue;
			}

			const prepared = createPreparedToolCallSnapshot(preparation);
			state.preparedToolCalls.push(prepared);
			toolExecutionRequests.push(createToolExecutionRequest(prepared));
		}

		if (toolExecutionRequests.length > 0) {
			state.phase = "awaiting_tool_execution";
			return createStepResult(state, events, "complete_tool_call", { toolExecutionRequests });
		}

		state.phase = "awaiting_turn_close";
		return createStepResult(state, events, "finalize_turn");
	}

	const toolCall = state.pendingToolCalls.shift();
	if (!toolCall) {
		state.phase = "awaiting_turn_close";
		return createStepResult(state, events, "finalize_turn");
	}

	await emit({
		type: "tool_execution_start",
		toolCallId: toolCall.id,
		toolName: toolCall.name,
		args: toolCall.arguments,
	});

	const preparation = await prepareToolCall(
		createLoopContext(state, runtime.tools),
		assistantMessage,
		toolCall,
		runtime.config,
		signal,
	);

	if (preparation.kind === "immediate") {
		const finalized: FinalizedToolCallOutcome = {
			toolCall,
			result: preparation.result,
			isError: preparation.isError,
		};
		await emitToolExecutionEnd(finalized, emit);
		const message = createToolResultMessage(finalized);
		await emitToolResultMessage(message, emit);
		state.completedToolResults.push(
			createCompletedToolCallSnapshot(toolCall, undefined, finalized.result, finalized.isError, message),
		);
		if (state.pendingToolCalls.length > 0) {
			return createStepResult(state, events, "prepare_tool_calls");
		}

		state.phase = "awaiting_turn_close";
		return createStepResult(state, events, "finalize_turn");
	}

	const prepared = createPreparedToolCallSnapshot(preparation);
	state.preparedToolCalls = [prepared];
	state.phase = "awaiting_tool_execution";
	return createStepResult(state, events, "complete_tool_call", {
		toolExecutionRequests: [createToolExecutionRequest(prepared)],
	});
}

async function completeToolCallStep(
	state: LoopState,
	command: Extract<StepCommand, { type: "complete_tool_call" }>,
	runtime: StepLoopRuntime,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	events: AgentEvent[],
): Promise<StepResult> {
	assertPhase(state, "awaiting_tool_execution", "complete_tool_call");

	const assistantMessage = state.currentAssistantMessage;
	if (!assistantMessage) {
		throw new Error("Cannot complete a tool call without an assistant message");
	}

	if (!state.preparedToolCalls.some((prepared) => prepared.toolCallId === command.toolCallId)) {
		throw new Error(`Unknown prepared tool call: ${command.toolCallId}`);
	}

	if (state.executedToolCalls.some((executed) => executed.toolCallId === command.toolCallId)) {
		throw new Error(`Tool call already completed: ${command.toolCallId}`);
	}

	state.executedToolCalls.push({
		toolCallId: command.toolCallId,
		result: command.result,
		isError: command.isError,
	});

	while (state.preparedToolCalls.length > 0) {
		const nextPrepared = state.preparedToolCalls[0];
		const executedIndex = state.executedToolCalls.findIndex(
			(executed) => executed.toolCallId === nextPrepared.toolCallId,
		);
		if (executedIndex === -1) {
			break;
		}

		const [prepared] = state.preparedToolCalls.splice(0, 1);
		const [executed] = state.executedToolCalls.splice(executedIndex, 1);
		const finalized = await finalizeExecutedToolCall(
			createLoopContext(state, runtime.tools),
			assistantMessage,
			createResolvedToolCall(prepared),
			executed,
			runtime.config,
			signal,
		);
		await emitToolExecutionEnd(finalized, emit);
		const message = createToolResultMessage(finalized);
		await emitToolResultMessage(message, emit);
		state.completedToolResults.push(
			createCompletedToolCallSnapshot(
				{
					type: "toolCall",
					id: prepared.toolCallId,
					name: prepared.toolName,
					arguments: prepared.rawArguments,
				},
				prepared.args,
				finalized.result,
				finalized.isError,
				message,
			),
		);
	}

	if (state.preparedToolCalls.length > 0) {
		return createStepResult(state, events, "complete_tool_call");
	}

	if (state.pendingToolCalls.length > 0) {
		state.phase = "awaiting_tool_preflight";
		return createStepResult(state, events, "prepare_tool_calls");
	}

	state.phase = "awaiting_turn_close";
	return createStepResult(state, events, "finalize_turn");
}

async function finalizeTurnStep(
	state: LoopState,
	runtime: StepLoopRuntime,
	emit: AgentEventSink,
	events: AgentEvent[],
): Promise<StepResult> {
	assertPhase(state, "awaiting_turn_close", "finalize_turn");

	const assistantMessage = state.currentAssistantMessage;
	if (!assistantMessage) {
		throw new Error("Cannot finalize a turn without an assistant message");
	}

	const hasToolCalls = assistantMessage.content.some((content) => content.type === "toolCall");
	const toolResultMessages = state.completedToolResults.map((result) => result.message);
	for (const result of toolResultMessages) {
		state.messages.push(result);
		state.newMessages.push(result);
	}

	await emit({
		type: "turn_end",
		message: assistantMessage,
		toolResults: toolResultMessages,
	});

	state.pendingSteeringMessages = (await runtime.config.getSteeringMessages?.()) || [];
	clearTurnState(state);

	if (hasToolCalls || state.pendingSteeringMessages.length > 0) {
		state.phase = "awaiting_assistant";
		return createStepResult(state, events, "run_assistant_turn");
	}

	state.phase = "awaiting_follow_up";
	return createStepResult(state, events, "check_follow_up");
}

async function checkFollowUpStep(
	state: LoopState,
	runtime: StepLoopRuntime,
	emit: AgentEventSink,
	events: AgentEvent[],
): Promise<StepResult> {
	assertPhase(state, "awaiting_follow_up", "check_follow_up");

	state.pendingFollowUpMessages = (await runtime.config.getFollowUpMessages?.()) || [];
	if (state.pendingFollowUpMessages.length > 0) {
		state.phase = "awaiting_assistant";
		return createStepResult(state, events, "run_assistant_turn");
	}

	state.phase = "completed";
	state.terminalStatus = "completed";
	await emit({ type: "agent_end", messages: state.newMessages });
	return createStepResult(state, events, "completed", { terminalMessages: state.newMessages });
}

function cloneLoopState(state: LoopState): LoopState {
	return {
		...state,
		messages: state.messages.slice(),
		newMessages: state.newMessages.slice(),
		pendingPromptMessages: state.pendingPromptMessages.slice(),
		pendingSteeringMessages: state.pendingSteeringMessages.slice(),
		pendingFollowUpMessages: state.pendingFollowUpMessages.slice(),
		pendingToolCalls: state.pendingToolCalls.slice(),
		preparedToolCalls: state.preparedToolCalls.map((prepared) => ({ ...prepared })),
		executedToolCalls: state.executedToolCalls.map((executed) => ({ ...executed })),
		completedToolResults: state.completedToolResults.map((completed) => ({ ...completed })),
	};
}

function createLoopContext(state: LoopState, tools: AgentTool<any>[] | undefined): AgentContext {
	return {
		systemPrompt: state.systemPrompt,
		messages: state.messages,
		tools,
	};
}

async function emitPendingPromptMessages(state: LoopState, emit: AgentEventSink): Promise<void> {
	for (const message of state.pendingPromptMessages) {
		await emit({ type: "message_start", message });
		await emit({ type: "message_end", message });
	}
	state.pendingPromptMessages = [];
}

async function injectPendingMessages(state: LoopState, messages: AgentMessage[], emit: AgentEventSink): Promise<void> {
	for (const message of messages) {
		await emit({ type: "message_start", message });
		await emit({ type: "message_end", message });
		state.messages.push(message);
		state.newMessages.push(message);
	}
}

function wrapConfigWithPayloadCapture(config: AgentLoopConfig, onPayload: (payload: unknown) => void): AgentLoopConfig {
	return {
		...config,
		onPayload: async (payload, model) => {
			onPayload(payload);
			return config.onPayload?.(payload, model);
		},
	};
}

function createPreparedToolCallSnapshot(prepared: PreparedToolCall): PreparedToolCallSnapshot {
	return {
		toolCallId: prepared.toolCall.id,
		toolName: prepared.toolCall.name,
		rawArguments: prepared.toolCall.arguments,
		args: prepared.args,
	};
}

function createToolExecutionRequest(prepared: PreparedToolCallSnapshot): ToolExecutionRequest {
	return {
		toolCallId: prepared.toolCallId,
		toolName: prepared.toolName,
		rawArguments: prepared.rawArguments,
		args: prepared.args,
	};
}

function createCompletedToolCallSnapshot(
	toolCall: AgentToolCall,
	args: unknown,
	result: AgentToolResult<unknown>,
	isError: boolean,
	message: ToolResultMessage<unknown>,
): CompletedToolCallSnapshot {
	return {
		toolCallId: toolCall.id,
		toolName: toolCall.name,
		rawArguments: toolCall.arguments,
		args,
		result,
		isError,
		message,
	};
}

function createResolvedToolCall(prepared: PreparedToolCallSnapshot): ResolvedToolCall {
	return {
		toolCall: {
			type: "toolCall",
			id: prepared.toolCallId,
			name: prepared.toolName,
			arguments: prepared.rawArguments,
		},
		args: prepared.args,
	};
}

function clearTurnState(state: LoopState): void {
	state.pendingToolCalls = [];
	state.preparedToolCalls = [];
	state.executedToolCalls = [];
	state.completedToolResults = [];
	state.currentAssistantMessage = undefined;
}

async function advanceAfterAssistantMessage(
	state: LoopState,
	message: AssistantMessage,
	emit: AgentEventSink,
	events: AgentEvent[],
	options: {
		providerRequestPayload?: unknown;
	} = {},
): Promise<StepResult> {
	state.currentAssistantMessage = message;
	state.newMessages.push(message);
	state.pendingToolCalls = message.content.filter((content) => content.type === "toolCall");
	state.preparedToolCalls = [];
	state.executedToolCalls = [];
	state.completedToolResults = [];
	state.firstTurn = false;

	if (message.stopReason === "error" || message.stopReason === "aborted") {
		state.phase = "failed";
		state.terminalStatus = "failed";
		await emit({ type: "turn_end", message, toolResults: [] });
		await emit({ type: "agent_end", messages: state.newMessages });
		return createStepResult(state, events, "failed", {
			providerRequestPayload: options.providerRequestPayload,
			terminalMessages: state.newMessages,
		});
	}

	if (state.pendingToolCalls.length > 0) {
		state.phase = "awaiting_tool_preflight";
		return createStepResult(state, events, "prepare_tool_calls", {
			providerRequestPayload: options.providerRequestPayload,
		});
	}

	state.phase = "awaiting_turn_close";
	return createStepResult(state, events, "finalize_turn", {
		providerRequestPayload: options.providerRequestPayload,
	});
}

function createStepResult(
	state: LoopState,
	events: AgentEvent[],
	nextAction: StepResult["nextAction"],
	options: {
		providerRequestPayload?: unknown;
		toolExecutionRequests?: ToolExecutionRequest[];
		terminalMessages?: AgentMessage[];
	} = {},
): StepResult {
	return {
		state,
		events,
		nextAction,
		providerRequestPayload: options.providerRequestPayload,
		toolExecutionRequests: options.toolExecutionRequests,
		terminalMessages: options.terminalMessages,
	};
}

function assertPhase(state: LoopState, phase: LoopState["phase"], command: StepCommand["type"]): void {
	if (state.phase !== phase) {
		throw new Error(`Cannot ${command} while loop phase is ${state.phase}`);
	}
}

function createAgentStream(): EventStream<AgentEvent, AgentMessage[]> {
	return new EventStream<AgentEvent, AgentMessage[]>(
		(event: AgentEvent) => event.type === "agent_end",
		(event: AgentEvent) => (event.type === "agent_end" ? event.messages : []),
	);
}

/**
 * Main loop logic shared by agentLoop and agentLoopContinue.
 */
async function runLoop(
	currentContext: AgentContext,
	newMessages: AgentMessage[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	streamFn?: StreamFn,
): Promise<void> {
	let firstTurn = true;
	// Check for steering messages at start (user may have typed while waiting)
	let pendingMessages: AgentMessage[] = (await config.getSteeringMessages?.()) || [];

	// Outer loop: continues when queued follow-up messages arrive after agent would stop
	while (true) {
		let hasMoreToolCalls = true;

		// Inner loop: process tool calls and steering messages
		while (hasMoreToolCalls || pendingMessages.length > 0) {
			if (!firstTurn) {
				await emit({ type: "turn_start" });
			} else {
				firstTurn = false;
			}

			// Process pending messages (inject before next assistant response)
			if (pendingMessages.length > 0) {
				for (const message of pendingMessages) {
					await emit({ type: "message_start", message });
					await emit({ type: "message_end", message });
					currentContext.messages.push(message);
					newMessages.push(message);
				}
				pendingMessages = [];
			}

			// Stream assistant response
			const message = await streamAssistantResponse(currentContext, config, signal, emit, streamFn);
			newMessages.push(message);

			if (message.stopReason === "error" || message.stopReason === "aborted") {
				await emit({ type: "turn_end", message, toolResults: [] });
				await emit({ type: "agent_end", messages: newMessages });
				return;
			}

			// Check for tool calls
			const toolCalls = message.content.filter((c) => c.type === "toolCall");

			const toolResults: ToolResultMessage[] = [];
			hasMoreToolCalls = false;
			if (toolCalls.length > 0) {
				const executedToolBatch = await executeToolCalls(currentContext, message, config, signal, emit);
				toolResults.push(...executedToolBatch.messages);
				hasMoreToolCalls = !executedToolBatch.terminate;

				for (const result of toolResults) {
					currentContext.messages.push(result);
					newMessages.push(result);
				}
			}

			await emit({ type: "turn_end", message, toolResults });

			pendingMessages = (await config.getSteeringMessages?.()) || [];
		}

		// Agent would stop here. Check for follow-up messages.
		const followUpMessages = (await config.getFollowUpMessages?.()) || [];
		if (followUpMessages.length > 0) {
			// Set as pending so inner loop processes them
			pendingMessages = followUpMessages;
			continue;
		}

		// No more messages, exit
		break;
	}

	await emit({ type: "agent_end", messages: newMessages });
}

/**
 * Stream an assistant response from the LLM.
 * This is where AgentMessage[] gets transformed to Message[] for the LLM.
 */
async function streamAssistantResponse(
	context: AgentContext,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	streamFn?: StreamFn,
): Promise<AssistantMessage> {
	// Apply context transform if configured (AgentMessage[] → AgentMessage[])
	let messages = context.messages;
	if (config.transformContext) {
		messages = await config.transformContext(messages, signal);
	}

	// Convert to LLM-compatible messages (AgentMessage[] → Message[])
	const llmMessages = await config.convertToLlm(messages);

	// Build LLM context
	const llmContext: Context = {
		systemPrompt: context.systemPrompt,
		messages: llmMessages,
		tools: context.tools,
	};

	const streamFunction = streamFn || streamSimple;

	// Resolve API key (important for expiring tokens)
	const resolvedApiKey =
		(config.getApiKey ? await config.getApiKey(config.model.provider) : undefined) || config.apiKey;

	const response = await streamFunction(config.model, llmContext, {
		...config,
		apiKey: resolvedApiKey,
		signal,
	});

	let partialMessage: AssistantMessage | null = null;
	let addedPartial = false;

	for await (const event of response) {
		switch (event.type) {
			case "start":
				partialMessage = event.partial;
				context.messages.push(partialMessage);
				addedPartial = true;
				await emit({ type: "message_start", message: { ...partialMessage } });
				break;

			case "text_start":
			case "text_delta":
			case "text_end":
			case "thinking_start":
			case "thinking_delta":
			case "thinking_end":
			case "toolcall_start":
			case "toolcall_delta":
			case "toolcall_end":
				if (partialMessage) {
					partialMessage = event.partial;
					context.messages[context.messages.length - 1] = partialMessage;
					await emit({
						type: "message_update",
						assistantMessageEvent: event,
						message: { ...partialMessage },
					});
				}
				break;

			case "done":
			case "error": {
				const finalMessage = await response.result();
				if (addedPartial) {
					context.messages[context.messages.length - 1] = finalMessage;
				} else {
					context.messages.push(finalMessage);
				}
				if (!addedPartial) {
					await emit({ type: "message_start", message: { ...finalMessage } });
				}
				await emit({ type: "message_end", message: finalMessage });
				return finalMessage;
			}
		}
	}

	const finalMessage = await response.result();
	if (addedPartial) {
		context.messages[context.messages.length - 1] = finalMessage;
	} else {
		context.messages.push(finalMessage);
		await emit({ type: "message_start", message: { ...finalMessage } });
	}
	await emit({ type: "message_end", message: finalMessage });
	return finalMessage;
}

/**
 * Execute tool calls from an assistant message.
 */
async function executeToolCalls(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallBatch> {
	const toolCalls = assistantMessage.content.filter((c) => c.type === "toolCall");
	const hasSequentialToolCall = toolCalls.some(
		(tc) => currentContext.tools?.find((t) => t.name === tc.name)?.executionMode === "sequential",
	);
	if (config.toolExecution === "sequential" || hasSequentialToolCall) {
		return executeToolCallsSequential(currentContext, assistantMessage, toolCalls, config, signal, emit);
	}
	return executeToolCallsParallel(currentContext, assistantMessage, toolCalls, config, signal, emit);
}

type ExecutedToolCallBatch = {
	messages: ToolResultMessage[];
	terminate: boolean;
};

async function executeToolCallsSequential(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallBatch> {
	const finalizedCalls: FinalizedToolCallOutcome[] = [];
	const messages: ToolResultMessage[] = [];

	for (const toolCall of toolCalls) {
		await emit({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		});

		const preparation = await prepareToolCall(currentContext, assistantMessage, toolCall, config, signal);
		let finalized: FinalizedToolCallOutcome;
		if (preparation.kind === "immediate") {
			finalized = {
				toolCall,
				result: preparation.result,
				isError: preparation.isError,
			};
		} else {
			const executed = await executePreparedToolCall(preparation, signal, emit);
			finalized = await finalizeExecutedToolCall(
				currentContext,
				assistantMessage,
				preparation,
				executed,
				config,
				signal,
			);
		}

		await emitToolExecutionEnd(finalized, emit);
		const toolResultMessage = createToolResultMessage(finalized);
		await emitToolResultMessage(toolResultMessage, emit);
		finalizedCalls.push(finalized);
		messages.push(toolResultMessage);
	}

	return {
		messages,
		terminate: shouldTerminateToolBatch(finalizedCalls),
	};
}

async function executeToolCallsParallel(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallBatch> {
	const finalizedCalls: FinalizedToolCallEntry[] = [];

	for (const toolCall of toolCalls) {
		await emit({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		});

		const preparation = await prepareToolCall(currentContext, assistantMessage, toolCall, config, signal);
		if (preparation.kind === "immediate") {
			const finalized = {
				toolCall,
				result: preparation.result,
				isError: preparation.isError,
			} satisfies FinalizedToolCallOutcome;
			await emitToolExecutionEnd(finalized, emit);
			finalizedCalls.push(finalized);
			continue;
		}

		finalizedCalls.push(async () => {
			const executed = await executePreparedToolCall(preparation, signal, emit);
			const finalized = await finalizeExecutedToolCall(
				currentContext,
				assistantMessage,
				preparation,
				executed,
				config,
				signal,
			);
			await emitToolExecutionEnd(finalized, emit);
			return finalized;
		});
	}

	const orderedFinalizedCalls = await Promise.all(
		finalizedCalls.map((entry) => (typeof entry === "function" ? entry() : Promise.resolve(entry))),
	);
	const messages: ToolResultMessage[] = [];
	for (const finalized of orderedFinalizedCalls) {
		const toolResultMessage = createToolResultMessage(finalized);
		await emitToolResultMessage(toolResultMessage, emit);
		messages.push(toolResultMessage);
	}

	return {
		messages,
		terminate: shouldTerminateToolBatch(orderedFinalizedCalls),
	};
}

/**
 * Live, in-memory state of a tool call after lookup, validation, and
 * `beforeToolCall`. Holds the resolved `AgentTool` reference because the
 * inline path passes this straight to `executePreparedToolCall`, which calls
 * `prepared.tool.execute(...)`. Not serializable — the stepped loop
 * captures a `PreparedToolCallSnapshot` instead.
 */
type PreparedToolCall = {
	kind: "prepared";
	toolCall: AgentToolCall;
	tool: AgentTool<any>;
	args: unknown;
};

/**
 * Subset of `PreparedToolCall` that `finalizeExecutedToolCall` actually reads:
 * the original `toolCall` block and the post-validation `args`. The stepped
 * path reconstructs this from a `PreparedToolCallSnapshot` at finalize time
 * (the live tool reference doesn't survive a process boundary, but isn't
 * needed once execute is done).
 */
type ResolvedToolCall = {
	toolCall: AgentToolCall;
	args: unknown;
};

type ImmediateToolCallOutcome = {
	kind: "immediate";
	result: AgentToolResult<any>;
	isError: boolean;
};

type ExecutedToolCallOutcome = {
	result: AgentToolResult<any>;
	isError: boolean;
};

type FinalizedToolCallOutcome = {
	toolCall: AgentToolCall;
	result: AgentToolResult<any>;
	isError: boolean;
};

type FinalizedToolCallEntry = FinalizedToolCallOutcome | (() => Promise<FinalizedToolCallOutcome>);

function shouldTerminateToolBatch(finalizedCalls: FinalizedToolCallOutcome[]): boolean {
	return finalizedCalls.length > 0 && finalizedCalls.every((finalized) => finalized.result.terminate === true);
}

function prepareToolCallArguments(tool: AgentTool<any>, toolCall: AgentToolCall): AgentToolCall {
	if (!tool.prepareArguments) {
		return toolCall;
	}
	const preparedArguments = tool.prepareArguments(toolCall.arguments);
	if (preparedArguments === toolCall.arguments) {
		return toolCall;
	}
	return {
		...toolCall,
		arguments: preparedArguments as Record<string, any>,
	};
}

async function prepareToolCall(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCall: AgentToolCall,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
): Promise<PreparedToolCall | ImmediateToolCallOutcome> {
	const tool = currentContext.tools?.find((t) => t.name === toolCall.name);
	if (!tool) {
		return {
			kind: "immediate",
			result: createErrorToolResult(`Tool ${toolCall.name} not found`),
			isError: true,
		};
	}

	try {
		const preparedToolCall = prepareToolCallArguments(tool, toolCall);
		const validatedArgs = validateToolArguments(tool, preparedToolCall);
		if (config.beforeToolCall) {
			const beforeResult = await config.beforeToolCall(
				{
					assistantMessage,
					toolCall,
					args: validatedArgs,
					context: currentContext,
				},
				signal,
			);
			if (beforeResult?.block) {
				return {
					kind: "immediate",
					result: createErrorToolResult(beforeResult.reason || "Tool execution was blocked"),
					isError: true,
				};
			}
		}
		return {
			kind: "prepared",
			toolCall,
			tool,
			args: validatedArgs,
		};
	} catch (error) {
		return {
			kind: "immediate",
			result: createErrorToolResult(error instanceof Error ? error.message : String(error)),
			isError: true,
		};
	}
}

async function executePreparedToolCall(
	prepared: PreparedToolCall,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallOutcome> {
	const updateEvents: Promise<void>[] = [];

	try {
		const result = await prepared.tool.execute(
			prepared.toolCall.id,
			prepared.args as never,
			signal,
			(partialResult) => {
				updateEvents.push(
					Promise.resolve(
						emit({
							type: "tool_execution_update",
							toolCallId: prepared.toolCall.id,
							toolName: prepared.toolCall.name,
							args: prepared.toolCall.arguments,
							partialResult,
						}),
					),
				);
			},
		);
		await Promise.all(updateEvents);
		return { result, isError: false };
	} catch (error) {
		await Promise.all(updateEvents);
		return {
			result: createErrorToolResult(error instanceof Error ? error.message : String(error)),
			isError: true,
		};
	}
}

async function finalizeExecutedToolCall(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	prepared: ResolvedToolCall,
	executed: ExecutedToolCallOutcome,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
): Promise<FinalizedToolCallOutcome> {
	let result = executed.result;
	let isError = executed.isError;

	if (config.afterToolCall) {
		try {
			const afterResult = await config.afterToolCall(
				{
					assistantMessage,
					toolCall: prepared.toolCall,
					args: prepared.args,
					result,
					isError,
					context: currentContext,
				},
				signal,
			);
			if (afterResult) {
				result = {
					content: afterResult.content ?? result.content,
					details: afterResult.details ?? result.details,
					terminate: afterResult.terminate ?? result.terminate,
				};
				isError = afterResult.isError ?? isError;
			}
		} catch (error) {
			result = createErrorToolResult(error instanceof Error ? error.message : String(error));
			isError = true;
		}
	}

	return {
		toolCall: prepared.toolCall,
		result,
		isError,
	};
}

function createErrorToolResult(message: string): AgentToolResult<any> {
	return {
		content: [{ type: "text", text: message }],
		details: {},
	};
}

async function emitToolExecutionEnd(finalized: FinalizedToolCallOutcome, emit: AgentEventSink): Promise<void> {
	await emit({
		type: "tool_execution_end",
		toolCallId: finalized.toolCall.id,
		toolName: finalized.toolCall.name,
		result: finalized.result,
		isError: finalized.isError,
	});
}

function createToolResultMessage(finalized: FinalizedToolCallOutcome): ToolResultMessage {
	return {
		role: "toolResult",
		toolCallId: finalized.toolCall.id,
		toolName: finalized.toolCall.name,
		content: finalized.result.content,
		details: finalized.result.details,
		isError: finalized.isError,
		timestamp: Date.now(),
	};
}

async function emitToolResultMessage(toolResultMessage: ToolResultMessage, emit: AgentEventSink): Promise<void> {
	await emit({ type: "message_start", message: toolResultMessage });
	await emit({ type: "message_end", message: toolResultMessage });
}
