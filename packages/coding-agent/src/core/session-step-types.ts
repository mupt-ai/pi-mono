import type { AssistantMessage, ImageContent } from "@mariozechner/pi-ai";
import type {
	AgentEvent,
	AgentMessage,
	AgentToolResult,
	LoopState,
	ToolExecutionRequest,
} from "@mupt-ai/pi-agent-core";
import type { InputSource } from "./extensions/index.js";

export type SessionLoopPhase =
	| "preparing_prompt"
	| "awaiting_assistant"
	| "awaiting_tool_preflight"
	| "awaiting_tool_execution"
	| "awaiting_turn_close"
	| "awaiting_post_turn_effects"
	| "awaiting_compaction"
	| "completed"
	| "failed";

export type SessionLoopTerminalStatus = "running" | "completed" | "failed";

export type SessionLoopInput =
	| {
			kind: "text";
			text: string;
			images?: ImageContent[];
			expandPromptTemplates: boolean;
			source: InputSource;
	  }
	| {
			kind: "messages";
			messages: AgentMessage[];
	  };

export interface SessionQueueSnapshot {
	steering: AgentMessage[];
	followUp: AgentMessage[];
	steeringLabels: string[];
	followUpLabels: string[];
	pendingNextTurnMessages: AgentMessage[];
}

export interface SessionCompactionRequest {
	reason: "threshold" | "overflow";
	willRetry: boolean;
}

export interface SessionLoopState {
	phase: SessionLoopPhase;
	terminalStatus: SessionLoopTerminalStatus;
	input: SessionLoopInput;
	coreState?: LoopState;
	preparedPromptMessages: AgentMessage[];
	lastAssistantMessage?: AssistantMessage;
	queue: SessionQueueSnapshot;
	retryAttempt: number;
	overflowRecoveryAttempted: boolean;
	compactionRequest?: SessionCompactionRequest;
}

export type SessionPersistenceOp =
	| { type: "append_message"; message: AgentMessage }
	| {
			type: "append_custom_message";
			customType: string;
			content: unknown;
			display: boolean;
			details?: unknown;
	  }
	| {
			type: "append_compaction";
			summary: string;
			firstKeptEntryId: string;
			tokensBefore: number;
			details?: unknown;
			fromExtension?: boolean;
	  };

export type SessionStepCommand =
	| { type: "prepare_prompt" }
	| { type: "run_assistant_turn" }
	| { type: "prepare_tool_calls" }
	| { type: "complete_tool_call"; toolCallId: string; result: AgentToolResult<unknown>; isError: boolean }
	| { type: "finalize_turn" }
	| { type: "run_post_turn_effects" }
	| { type: "run_compaction" };

export type SessionStepNextAction =
	| "run_assistant_turn"
	| "prepare_tool_calls"
	| "complete_tool_call"
	| "finalize_turn"
	| "run_post_turn_effects"
	| "run_compaction"
	| "completed"
	| "failed";

export interface SessionStepResult {
	state: SessionLoopState;
	coreEvents: AgentEvent[];
	sessionEvents: unknown[];
	sessionOps: SessionPersistenceOp[];
	nextAction: SessionStepNextAction;
	providerRequestPayload?: unknown;
	toolExecutionRequests?: ToolExecutionRequest[];
	terminalMessages?: AgentMessage[];
}
