import type {
	AgentEvent,
	AgentMessage,
	AgentToolResult,
	LoopState,
	NormalizedAssistantMessageEventSource,
	PreparedProviderRequest,
	ToolExecutionRequest,
} from "@mariozechner/pi-agent-core";
import type { AssistantMessage, ImageContent } from "@mariozechner/pi-ai";
import type { InputSource } from "./extensions/index.js";

/**
 * Discrete phase of the AgentSession-level stepped loop.
 *
 * Wraps the lower-level core agent `LoopPhase` with session-only phases for
 * prompt preparation, post-turn extension effects, and compaction.
 */
export type SessionLoopPhase =
	| "preparing_prompt"
	| "awaiting_assistant"
	| "awaiting_provider_response"
	| "awaiting_tool_preflight"
	| "awaiting_tool_execution"
	| "awaiting_turn_close"
	| "awaiting_post_turn_effects"
	| "awaiting_compaction"
	| "completed"
	| "failed";

/** Terminal status reported by the session-level stepped loop. */
export type SessionLoopTerminalStatus = "running" | "completed" | "failed";

/**
 * Input that kicks off a session-level stepped loop.
 *
 * `text` is the normal user-typed path (slash commands and prompt templates
 * may still need expansion via `expandPromptTemplates`). `messages` is the
 * pre-built path used when callers already have AgentMessages in hand.
 */
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

/**
 * Serializable snapshot of the session's steering and follow-up queues.
 *
 * `steeringLabels` / `followUpLabels` carry the human-readable labels shown
 * in the UI for each queued message. `pendingNextTurnMessages` are messages
 * that have been claimed for the next turn but not yet handed to the core loop.
 */
export interface SessionQueueSnapshot {
	steering: AgentMessage[];
	followUp: AgentMessage[];
	steeringLabels: string[];
	followUpLabels: string[];
	pendingNextTurnMessages: AgentMessage[];
}

/** Pending compaction emitted by the stepped loop. `willRetry` means the original turn should retry after compacting. */
export interface SessionCompactionRequest {
	reason: "threshold" | "overflow";
	willRetry: boolean;
}

/**
 * Serializable carry-state for the session-level stepped loop.
 *
 * `coreState` is the embedded lower-level agent loop state (when active);
 * the surrounding fields capture session-only concerns like queues, retry
 * counts, and the most recent assistant message used for compaction.
 */
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

/**
 * Persistence operation the host should apply to the session log.
 *
 * Steps emit these instead of writing to disk directly so workflow runners
 * can persist them transactionally alongside the carry-state.
 */
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

/**
 * One-step command accepted by the session stepped loop.
 *
 * Most commands map onto the underlying core agent loop; `prepare_prompt`,
 * `run_post_turn_effects`, and `run_compaction` are session-only steps that
 * have no core-loop equivalent.
 */
export type SessionStepCommand =
	| { type: "prepare_prompt" }
	| { type: "run_assistant_turn" }
	| { type: "complete_provider_response"; events: NormalizedAssistantMessageEventSource }
	| { type: "prepare_tool_calls" }
	| { type: "complete_tool_call"; toolCallId: string; result: AgentToolResult<unknown>; isError: boolean }
	| { type: "finalize_turn" }
	| { type: "run_post_turn_effects" }
	| { type: "run_compaction" };

/** Next command the host should issue, or a terminal status when the session loop is done. */
export type SessionStepNextAction =
	| "run_assistant_turn"
	| "complete_provider_response"
	| "prepare_tool_calls"
	| "complete_tool_call"
	| "finalize_turn"
	| "run_post_turn_effects"
	| "run_compaction"
	| "completed"
	| "failed";

/**
 * Result of advancing the session stepped loop by one command.
 *
 * `coreEvents` are agent-level events forwarded to UI subscribers,
 * `sessionEvents` are session-level events (compaction, etc.), and
 * `sessionOps` are the persistence writes the host must apply. External work
 * payloads (provider request, tool calls, terminal messages) signal what the
 * host needs to resolve before the next step.
 */
export interface SessionStepResult {
	state: SessionLoopState;
	coreEvents: AgentEvent[];
	sessionEvents: unknown[];
	sessionOps: SessionPersistenceOp[];
	nextAction: SessionStepNextAction;
	preparedProviderRequest?: PreparedProviderRequest;
	providerRequestPayload?: unknown;
	toolExecutionRequests?: ToolExecutionRequest[];
	terminalMessages?: AgentMessage[];
}
