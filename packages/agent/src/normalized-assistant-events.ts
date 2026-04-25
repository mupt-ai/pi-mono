import {
	type AssistantMessage,
	type AssistantMessageEvent,
	type Model,
	parseStreamingJson,
	type StopReason,
	type ToolCall,
} from "@mariozechner/pi-ai";

/**
 * Provider-agnostic streaming event for an assistant message.
 *
 * Hosts that run provider calls externally serialize the underlying SDK
 * stream into this normalized form, then feed it back into the agent loop.
 * `contentIndex` identifies the content block (text, thinking, or toolCall)
 * being updated. `done` and `error` are the only terminal event kinds.
 */
export type NormalizedAssistantMessageEvent =
	| { type: "start" }
	| { type: "text_start"; contentIndex: number }
	| { type: "text_delta"; contentIndex: number; delta: string }
	| { type: "text_end"; contentIndex: number; contentSignature?: string }
	| { type: "thinking_start"; contentIndex: number }
	| { type: "thinking_delta"; contentIndex: number; delta: string }
	| { type: "thinking_end"; contentIndex: number; contentSignature?: string }
	| { type: "toolcall_start"; contentIndex: number; id: string; toolName: string }
	| { type: "toolcall_delta"; contentIndex: number; delta: string }
	| { type: "toolcall_end"; contentIndex: number }
	| {
			type: "done";
			reason: Extract<StopReason, "stop" | "length" | "toolUse">;
			usage: AssistantMessage["usage"];
	  }
	| {
			type: "error";
			reason: Extract<StopReason, "aborted" | "error">;
			errorMessage?: string;
			usage: AssistantMessage["usage"];
	  };

/** Either a sync or async iterable of normalized events. Accepted by the loop and apply helpers. */
export type NormalizedAssistantMessageEventSource =
	| Iterable<NormalizedAssistantMessageEvent>
	| AsyncIterable<NormalizedAssistantMessageEvent>;

/**
 * In-progress reconstruction state for a normalized event stream.
 *
 * `message` is mutated as deltas arrive. `toolCallJson` accumulates raw JSON
 * fragments per content index so streaming arguments can be parsed
 * incrementally via `parseStreamingJson`.
 */
export interface NormalizedAssistantMessageState {
	message: AssistantMessage;
	toolCallJson: Map<number, string>;
}

/**
 * Build an empty `NormalizedAssistantMessageState` seeded with model identity
 * and zeroed usage/cost. Pass to `applyNormalizedAssistantMessageEvent` to
 * accumulate a streaming response into a complete `AssistantMessage`.
 */
export function createNormalizedAssistantMessageState(model: Model<any>): NormalizedAssistantMessageState {
	return {
		message: {
			role: "assistant",
			stopReason: "stop",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			timestamp: Date.now(),
		},
		toolCallJson: new Map(),
	};
}

/**
 * Apply one normalized event to the in-progress reconstruction state and
 * return the corresponding `AssistantMessageEvent` to forward to the agent.
 *
 * Mutates `state.message` in place: text/thinking deltas append to the
 * matching content block, tool-call deltas reparse accumulated JSON, and
 * `done`/`error` populate stop reason, usage, and any error message. Returns
 * `undefined` when an event has no agent-visible counterpart (currently only
 * `toolcall_end` against a non-toolCall block, which is silently dropped).
 *
 * Throws if a delta arrives for the wrong content type — that signals a bug
 * in the event source's normalization.
 */
export function applyNormalizedAssistantMessageEvent(
	normalizedEvent: NormalizedAssistantMessageEvent,
	state: NormalizedAssistantMessageState,
): AssistantMessageEvent | undefined {
	const partial = state.message;

	switch (normalizedEvent.type) {
		case "start":
			return { type: "start", partial };

		case "text_start":
			partial.content[normalizedEvent.contentIndex] = { type: "text", text: "" };
			return { type: "text_start", contentIndex: normalizedEvent.contentIndex, partial };

		case "text_delta": {
			const content = partial.content[normalizedEvent.contentIndex];
			if (content?.type !== "text") {
				throw new Error("Received text_delta for non-text content");
			}
			content.text += normalizedEvent.delta;
			return {
				type: "text_delta",
				contentIndex: normalizedEvent.contentIndex,
				delta: normalizedEvent.delta,
				partial,
			};
		}

		case "text_end": {
			const content = partial.content[normalizedEvent.contentIndex];
			if (content?.type !== "text") {
				throw new Error("Received text_end for non-text content");
			}
			content.textSignature = normalizedEvent.contentSignature;
			return {
				type: "text_end",
				contentIndex: normalizedEvent.contentIndex,
				content: content.text,
				partial,
			};
		}

		case "thinking_start":
			partial.content[normalizedEvent.contentIndex] = { type: "thinking", thinking: "" };
			return { type: "thinking_start", contentIndex: normalizedEvent.contentIndex, partial };

		case "thinking_delta": {
			const content = partial.content[normalizedEvent.contentIndex];
			if (content?.type !== "thinking") {
				throw new Error("Received thinking_delta for non-thinking content");
			}
			content.thinking += normalizedEvent.delta;
			return {
				type: "thinking_delta",
				contentIndex: normalizedEvent.contentIndex,
				delta: normalizedEvent.delta,
				partial,
			};
		}

		case "thinking_end": {
			const content = partial.content[normalizedEvent.contentIndex];
			if (content?.type !== "thinking") {
				throw new Error("Received thinking_end for non-thinking content");
			}
			content.thinkingSignature = normalizedEvent.contentSignature;
			return {
				type: "thinking_end",
				contentIndex: normalizedEvent.contentIndex,
				content: content.thinking,
				partial,
			};
		}

		case "toolcall_start":
			partial.content[normalizedEvent.contentIndex] = {
				type: "toolCall",
				id: normalizedEvent.id,
				name: normalizedEvent.toolName,
				arguments: {},
			};
			state.toolCallJson.set(normalizedEvent.contentIndex, "");
			return { type: "toolcall_start", contentIndex: normalizedEvent.contentIndex, partial };

		case "toolcall_delta": {
			const content = partial.content[normalizedEvent.contentIndex];
			if (content?.type !== "toolCall") {
				throw new Error("Received toolcall_delta for non-toolCall content");
			}
			const partialJson = `${state.toolCallJson.get(normalizedEvent.contentIndex) ?? ""}${normalizedEvent.delta}`;
			state.toolCallJson.set(normalizedEvent.contentIndex, partialJson);
			content.arguments = parseStreamingJson<Record<string, unknown>>(partialJson) || {};
			partial.content[normalizedEvent.contentIndex] = { ...content };
			return {
				type: "toolcall_delta",
				contentIndex: normalizedEvent.contentIndex,
				delta: normalizedEvent.delta,
				partial,
			};
		}

		case "toolcall_end": {
			const content = partial.content[normalizedEvent.contentIndex];
			if (content?.type !== "toolCall") {
				return undefined;
			}
			state.toolCallJson.delete(normalizedEvent.contentIndex);
			return {
				type: "toolcall_end",
				contentIndex: normalizedEvent.contentIndex,
				toolCall: content as ToolCall,
				partial,
			};
		}

		case "done":
			partial.stopReason = normalizedEvent.reason;
			partial.usage = normalizedEvent.usage;
			return { type: "done", reason: normalizedEvent.reason, message: partial };

		case "error":
			partial.stopReason = normalizedEvent.reason;
			partial.errorMessage = normalizedEvent.errorMessage;
			partial.usage = normalizedEvent.usage;
			return { type: "error", reason: normalizedEvent.reason, error: partial };
	}
}
