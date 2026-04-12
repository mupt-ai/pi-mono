import {
	type AssistantMessage,
	type AssistantMessageEvent,
	type Model,
	parseStreamingJson,
	type StopReason,
	type ToolCall,
} from "@mariozechner/pi-ai";

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

export type NormalizedAssistantMessageEventSource =
	| Iterable<NormalizedAssistantMessageEvent>
	| AsyncIterable<NormalizedAssistantMessageEvent>;

export interface NormalizedAssistantMessageState {
	message: AssistantMessage;
	toolCallJson: Map<number, string>;
}

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
