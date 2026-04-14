import { readFileSync } from "node:fs";
import type { AssistantMessage, Model, Transport } from "@mariozechner/pi-ai";
import {
	Agent,
	type AgentEvent,
	type AgentMessage,
	initializeLoopState,
	type PreparedProviderRequest,
	type ProviderExecutionMode,
	type ThinkingLevel,
	type ToolExecutionRequest,
} from "@mupt-ai/pi-agent-core";
import type { TSchema } from "@sinclair/typebox";
import { stripFrontmatter } from "../utils/frontmatter.js";
import { AgentSession, type AgentSessionEvent, type PromptOptions } from "./agent-session.js";
import { AuthStorage } from "./auth-storage.js";
import { type CompactionPreparation, type CompactionResult, prepareCompaction } from "./compaction/index.js";
import { createExtensionRuntime, type ToolDefinition } from "./extensions/index.js";
import { convertToLlm } from "./messages.js";
import { ModelRegistry } from "./model-registry.js";
import { expandPromptTemplate, type PromptTemplate } from "./prompt-templates.js";
import type { ResourceLoader } from "./resource-loader.js";
import { type SessionLogSnapshot, SessionManager } from "./session-manager.js";
import type {
	SessionCompactionRequest,
	SessionLoopInput,
	SessionLoopPhase,
	SessionLoopState,
	SessionLoopTerminalStatus,
	SessionPersistenceOp,
	SessionQueueSnapshot,
	SessionStepCommand,
	SessionStepResult,
} from "./session-step-types.js";
import type {
	CompactionSettings,
	ImageSettings,
	RetrySettings,
	Settings,
	ThinkingBudgetsSettings,
} from "./settings-manager.js";
import { SettingsManager } from "./settings-manager.js";
import type { Skill } from "./skills.js";
import type { SourceInfo } from "./source-info.js";

export interface WorkflowToolSnapshot {
	name: string;
	label: string;
	description: string;
	parameters: TSchema;
	promptSnippet?: string;
	promptGuidelines?: string[];
	prepareArguments?: (args: unknown) => unknown;
}

export interface WorkflowSkillSnapshot {
	name: string;
	description: string;
	location: string;
	baseDir: string;
	content: string;
	sourceInfo: SourceInfo;
	disableModelInvocation: boolean;
}

export interface WorkflowContextFileSnapshot {
	path: string;
	content: string;
}

export interface WorkflowSettingsSnapshot {
	transport?: Transport;
	steeringMode?: "all" | "one-at-a-time";
	followUpMode?: "all" | "one-at-a-time";
	compaction?: CompactionSettings;
	retry?: RetrySettings;
	images?: ImageSettings;
	shellCommandPrefix?: string;
	thinkingBudgets?: ThinkingBudgetsSettings;
}

export interface WorkflowEnvironmentSnapshot {
	cwd: string;
	model?: Model<any>;
	thinkingLevel?: ThinkingLevel;
	providerExecutionMode?: ProviderExecutionMode;
	activeToolNames: string[];
	tools: WorkflowToolSnapshot[];
	settings?: WorkflowSettingsSnapshot;
	systemPrompt?: string;
	appendSystemPrompt?: string[];
	promptTemplates?: PromptTemplate[];
	skills?: WorkflowSkillSnapshot[];
	agentsFiles?: WorkflowContextFileSnapshot[];
}

export type WorkflowInput = SessionLoopInput;
export type WorkflowQueueSnapshot = SessionQueueSnapshot;
export type WorkflowCompactionRequest = SessionCompactionRequest;
export type WorkflowPhase = SessionLoopPhase;
export type WorkflowTerminalStatus = SessionLoopTerminalStatus;

export interface PreparedCompactionRequest {
	model: Model<any>;
	reason: WorkflowCompactionRequest["reason"];
	willRetry: boolean;
	preparation: CompactionPreparation;
}

export interface WorkflowState {
	sessionId: string;
	environment: WorkflowEnvironmentSnapshot;
	phase: WorkflowPhase;
	terminalStatus: WorkflowTerminalStatus;
	input: WorkflowInput;
	coreState?: SessionLoopState["coreState"];
	preparedPromptMessages: AgentMessage[];
	lastAssistantMessage?: AssistantMessage;
	queue: WorkflowQueueSnapshot;
	retryAttempt: number;
	overflowRecoveryAttempted: boolean;
	compactionRequest?: WorkflowCompactionRequest;
	pendingProviderRequest?: PreparedProviderRequest;
	pendingToolCalls?: ToolExecutionRequest[];
	pendingCompaction?: PreparedCompactionRequest;
}

type WorkflowDelegatedCommand = Exclude<SessionStepCommand, { type: "run_compaction" }>;

export type WorkflowStepCommand =
	| WorkflowDelegatedCommand
	| { type: "prepare_compaction" }
	| { type: "complete_compaction"; result: CompactionResult<unknown>; fromExtension?: boolean };

export type WorkflowStepNextAction =
	| Exclude<SessionStepResult["nextAction"], "run_compaction">
	| "prepare_compaction"
	| "complete_compaction";

export interface WorkflowStepResult {
	state: WorkflowState;
	sessionLog: SessionLogSnapshot;
	coreEvents: AgentEvent[];
	sessionEvents: AgentSessionEvent[];
	sessionOps: SessionPersistenceOp[];
	nextAction: WorkflowStepNextAction;
	preparedProviderRequest?: PreparedProviderRequest;
	providerRequestPayload?: unknown;
	toolExecutionRequests?: ToolExecutionRequest[];
	preparedCompactionRequest?: PreparedCompactionRequest;
	terminalMessages?: AgentMessage[];
}

const UNSUPPORTED_EXTENSION_HANDLERS = [
	"input",
	"before_agent_start",
	"context",
	"before_provider_request",
	"session_before_compact",
	"tool_call",
	"tool_result",
];

function buildWorkflowSettingsSnapshot(settingsManager: SettingsManager): WorkflowSettingsSnapshot {
	return {
		transport: settingsManager.getTransport(),
		steeringMode: settingsManager.getSteeringMode(),
		followUpMode: settingsManager.getFollowUpMode(),
		compaction: settingsManager.getCompactionSettings(),
		retry: settingsManager.getRetrySettings(),
		images: {
			autoResize: settingsManager.getImageAutoResize(),
			blockImages: settingsManager.getBlockImages(),
		},
		shellCommandPrefix: settingsManager.getShellCommandPrefix(),
		thinkingBudgets: settingsManager.getThinkingBudgets(),
	};
}

function assertWorkflowSnapshotSupported(session: AgentSession): void {
	const runner = session.extensionRunner;
	if (!runner) {
		return;
	}

	const unsupportedFeatures: string[] = [];
	for (const eventType of UNSUPPORTED_EXTENSION_HANDLERS) {
		if (runner.hasHandlers(eventType)) {
			unsupportedFeatures.push(eventType);
		}
	}
	if (runner.getRegisteredCommands().length > 0) {
		unsupportedFeatures.push("commands");
	}

	if (unsupportedFeatures.length > 0) {
		throw new Error(
			`Workflow snapshots do not support dynamic extension features: ${unsupportedFeatures.join(", ")}`,
		);
	}
}

function captureWorkflowTools(session: AgentSession): WorkflowToolSnapshot[] {
	return session.getAllTools().map((tool) => {
		const definition = session.getToolDefinition(tool.name);
		if (!definition) {
			throw new Error(`Missing tool definition for ${tool.name}`);
		}
		return {
			name: definition.name,
			label: definition.label,
			description: definition.description,
			parameters: structuredClone(definition.parameters),
			promptSnippet: definition.promptSnippet,
			promptGuidelines: definition.promptGuidelines?.slice(),
			prepareArguments: definition.prepareArguments,
		};
	});
}

function captureWorkflowSkills(session: AgentSession): WorkflowSkillSnapshot[] {
	return session.resourceLoader.getSkills().skills.map((skill) => ({
		name: skill.name,
		description: skill.description,
		location: skill.filePath,
		baseDir: skill.baseDir,
		content: stripFrontmatter(readFileSync(skill.filePath, "utf-8")).trim(),
		sourceInfo: structuredClone(skill.sourceInfo),
		disableModelInvocation: skill.disableModelInvocation,
	}));
}

export function captureWorkflowEnvironmentSnapshot(
	session: AgentSession,
	options?: { providerExecutionMode?: ProviderExecutionMode },
): WorkflowEnvironmentSnapshot {
	assertWorkflowSnapshotSupported(session);

	return {
		cwd: session.sessionManager.getCwd(),
		model: session.model ? structuredClone(session.model) : undefined,
		thinkingLevel: session.thinkingLevel,
		providerExecutionMode: options?.providerExecutionMode ?? "external",
		activeToolNames: session.getActiveToolNames(),
		tools: captureWorkflowTools(session),
		settings: buildWorkflowSettingsSnapshot(session.settingsManager),
		systemPrompt: session.resourceLoader.getSystemPrompt(),
		appendSystemPrompt: session.resourceLoader.getAppendSystemPrompt(),
		promptTemplates: structuredClone(session.resourceLoader.getPrompts().prompts),
		skills: captureWorkflowSkills(session),
		agentsFiles: structuredClone(session.resourceLoader.getAgentsFiles().agentsFiles),
	};
}

export function captureSessionLogSnapshot(sessionManager: SessionManager): SessionLogSnapshot {
	return sessionManager.toSnapshot();
}

function createWorkflowResourceLoader(environment: WorkflowEnvironmentSnapshot): ResourceLoader {
	const runtime = createExtensionRuntime();
	const skills: Skill[] = (environment.skills ?? []).map((skill) => ({
		name: skill.name,
		description: skill.description,
		filePath: skill.location,
		baseDir: skill.baseDir,
		sourceInfo: structuredClone(skill.sourceInfo),
		disableModelInvocation: skill.disableModelInvocation,
	}));
	const prompts = structuredClone(environment.promptTemplates ?? []);
	const agentsFiles = structuredClone(environment.agentsFiles ?? []);
	const systemPrompt = environment.systemPrompt;
	const appendSystemPrompt = structuredClone(environment.appendSystemPrompt ?? []);

	return {
		getExtensions: () => ({ extensions: [], errors: [], runtime }),
		getSkills: () => ({ skills, diagnostics: [] }),
		getPrompts: () => ({ prompts, diagnostics: [] }),
		getThemes: () => ({ themes: [], diagnostics: [] }),
		getAgentsFiles: () => ({ agentsFiles }),
		getSystemPrompt: () => systemPrompt,
		getAppendSystemPrompt: () => appendSystemPrompt,
		extendResources: () => {},
		reload: async () => {},
	};
}

function createWorkflowToolDefinitions(
	tools: WorkflowEnvironmentSnapshot["tools"],
): Record<string, ToolDefinition<TSchema, unknown, unknown>> {
	return Object.fromEntries(
		tools.map((tool) => [
			tool.name,
			{
				name: tool.name,
				label: tool.label,
				description: tool.description,
				parameters: structuredClone(tool.parameters),
				promptSnippet: tool.promptSnippet,
				promptGuidelines: tool.promptGuidelines?.slice(),
				prepareArguments: tool.prepareArguments as
					| ToolDefinition<TSchema, unknown, unknown>["prepareArguments"]
					| undefined,
				execute: async () => {
					throw new Error(`Tool "${tool.name}" must be executed externally`);
				},
			} satisfies ToolDefinition<TSchema, unknown, unknown>,
		]),
	);
}

function createWorkflowSettingsManager(settingsSnapshot?: WorkflowSettingsSnapshot): SettingsManager {
	const settings: Partial<Settings> = {
		transport: settingsSnapshot?.transport,
		steeringMode: settingsSnapshot?.steeringMode,
		followUpMode: settingsSnapshot?.followUpMode,
		compaction: settingsSnapshot?.compaction,
		retry: settingsSnapshot?.retry,
		images: settingsSnapshot?.images,
		shellCommandPrefix: settingsSnapshot?.shellCommandPrefix,
		thinkingBudgets: settingsSnapshot?.thinkingBudgets,
	};
	return SettingsManager.inMemory(settings);
}

function createConvertToLlmWithBlockImages(settingsManager: SettingsManager) {
	return (messages: AgentMessage[]) => {
		const converted = convertToLlm(messages);
		if (!settingsManager.getBlockImages()) {
			return converted;
		}

		return converted.map((message) => {
			if (message.role === "user" || message.role === "toolResult") {
				const { content } = message;
				if (Array.isArray(content) && content.some((part) => part.type === "image")) {
					const filteredContent = content
						.map((part) =>
							part.type === "image" ? { type: "text" as const, text: "Image reading is disabled." } : part,
						)
						.filter(
							(part, index, parts) =>
								!(
									part.type === "text" &&
									part.text === "Image reading is disabled." &&
									index > 0 &&
									parts[index - 1]?.type === "text" &&
									parts[index - 1].text === "Image reading is disabled."
								),
						);
					return { ...message, content: filteredContent };
				}
			}
			return message;
		});
	};
}

function createWorkflowSession(environment: WorkflowEnvironmentSnapshot, sessionLog: SessionLogSnapshot): AgentSession {
	if (sessionLog.header.id === "") {
		throw new Error("Session snapshot is missing a session id");
	}

	const sessionManager = SessionManager.fromSnapshot(sessionLog);
	const settingsManager = createWorkflowSettingsManager(environment.settings);
	const authStorage = AuthStorage.inMemory();
	const modelRegistry = ModelRegistry.inMemory(authStorage);
	const agent = new Agent({
		initialState: {
			model: environment.model,
			thinkingLevel: environment.thinkingLevel ?? "off",
			systemPrompt: "",
			tools: [],
		},
		convertToLlm: createConvertToLlmWithBlockImages(settingsManager),
		sessionId: sessionManager.getSessionId(),
		steeringMode: settingsManager.getSteeringMode(),
		followUpMode: settingsManager.getFollowUpMode(),
		transport: settingsManager.getTransport(),
		thinkingBudgets: settingsManager.getThinkingBudgets(),
		maxRetryDelayMs: settingsManager.getRetrySettings().maxDelayMs,
	});

	const existingSession = sessionManager.buildSessionContext();
	agent.state.messages = existingSession.messages;
	if (environment.thinkingLevel === undefined) {
		agent.state.thinkingLevel = existingSession.thinkingLevel as ThinkingLevel;
	}

	return new AgentSession({
		agent,
		sessionManager,
		settingsManager,
		cwd: environment.cwd,
		modelRegistry,
		resourceLoader: createWorkflowResourceLoader(environment),
		initialActiveToolNames: environment.activeToolNames,
		baseToolDefinitionsOverride: createWorkflowToolDefinitions(environment.tools),
		providerExecutionMode: environment.providerExecutionMode ?? "external",
		compactionExecutionMode: "external",
	});
}

function expandWorkflowSkillCommand(text: string, skills: WorkflowSkillSnapshot[]): string {
	if (!text.startsWith("/skill:")) {
		return text;
	}

	const spaceIndex = text.indexOf(" ");
	const skillName = spaceIndex === -1 ? text.slice(7) : text.slice(7, spaceIndex);
	const args = spaceIndex === -1 ? "" : text.slice(spaceIndex + 1).trim();
	const skill = skills.find((candidate) => candidate.name === skillName);
	if (!skill) {
		return text;
	}

	const skillBlock = `<skill name="${skill.name}" location="${skill.location}">\nReferences are relative to ${skill.baseDir}.\n\n${skill.content}\n</skill>`;
	return args ? `${skillBlock}\n\n${args}` : skillBlock;
}

function preprocessWorkflowInput(input: WorkflowInput, environment: WorkflowEnvironmentSnapshot): WorkflowInput {
	if (input.kind !== "text" || !input.expandPromptTemplates) {
		return structuredClone(input);
	}

	let expandedText = expandWorkflowSkillCommand(input.text, environment.skills ?? []);
	expandedText = expandPromptTemplate(expandedText, environment.promptTemplates ?? []);

	return {
		...structuredClone(input),
		text: expandedText,
		expandPromptTemplates: false,
	};
}

function workflowStateToSessionLoopState(state: WorkflowState, inputOverride?: WorkflowInput): SessionLoopState {
	return {
		phase: state.phase,
		terminalStatus: state.terminalStatus,
		input: inputOverride ? structuredClone(inputOverride) : structuredClone(state.input),
		coreState: state.coreState ? structuredClone(state.coreState) : undefined,
		preparedPromptMessages: structuredClone(state.preparedPromptMessages),
		lastAssistantMessage: state.lastAssistantMessage ? structuredClone(state.lastAssistantMessage) : undefined,
		queue: structuredClone(state.queue),
		retryAttempt: state.retryAttempt,
		overflowRecoveryAttempted: state.overflowRecoveryAttempted,
		compactionRequest: state.compactionRequest ? structuredClone(state.compactionRequest) : undefined,
	};
}

function sessionLoopStateToWorkflowState(
	environment: WorkflowEnvironmentSnapshot,
	sessionId: string,
	loopState: SessionLoopState,
	overrides?: {
		pendingProviderRequest?: PreparedProviderRequest;
		pendingToolCalls?: ToolExecutionRequest[];
		pendingCompaction?: PreparedCompactionRequest;
	},
): WorkflowState {
	return {
		sessionId,
		environment,
		phase: loopState.phase,
		terminalStatus: loopState.terminalStatus,
		input: structuredClone(loopState.input),
		coreState: loopState.coreState ? structuredClone(loopState.coreState) : undefined,
		preparedPromptMessages: structuredClone(loopState.preparedPromptMessages),
		lastAssistantMessage: loopState.lastAssistantMessage
			? structuredClone(loopState.lastAssistantMessage)
			: undefined,
		queue: structuredClone(loopState.queue),
		retryAttempt: loopState.retryAttempt,
		overflowRecoveryAttempted: loopState.overflowRecoveryAttempted,
		compactionRequest: loopState.compactionRequest ? structuredClone(loopState.compactionRequest) : undefined,
		pendingProviderRequest: overrides?.pendingProviderRequest
			? structuredClone(overrides.pendingProviderRequest)
			: undefined,
		pendingToolCalls: overrides?.pendingToolCalls ? structuredClone(overrides.pendingToolCalls) : undefined,
		pendingCompaction: overrides?.pendingCompaction ? structuredClone(overrides.pendingCompaction) : undefined,
	};
}

function mapWorkflowNextAction(nextAction: SessionStepResult["nextAction"]): WorkflowStepNextAction {
	return nextAction === "run_compaction" ? "prepare_compaction" : nextAction;
}

function createContinuationLoopState(session: AgentSession): SessionLoopState["coreState"] {
	return initializeLoopState(
		[],
		{
			systemPrompt: session.agent.state.systemPrompt,
			messages: session.agent.state.messages,
			tools: session.agent.state.tools,
		},
		{ toolExecution: session.agent.toolExecution },
	);
}

function createCompactionEvents(
	result: CompactionResult<unknown> | undefined,
	request: WorkflowCompactionRequest,
): AgentSessionEvent[] {
	return [
		{
			type: "compaction_end",
			reason: request.reason,
			result,
			aborted: false,
			willRetry: request.willRetry,
		},
	];
}

export async function initializeWorkflowState(
	input: string | AgentMessage | AgentMessage[],
	environment: WorkflowEnvironmentSnapshot,
	sessionLog: SessionLogSnapshot,
	options?: PromptOptions,
): Promise<WorkflowState> {
	const session = createWorkflowSession(environment, sessionLog);
	try {
		const loopState = await session.initializeSessionLoopState(input, options);
		return sessionLoopStateToWorkflowState(environment, session.sessionId, loopState);
	} finally {
		session.dispose();
	}
}

export async function stepWorkflowState(
	state: WorkflowState,
	sessionLog: SessionLogSnapshot,
	command: WorkflowStepCommand,
	signal?: AbortSignal,
): Promise<WorkflowStepResult> {
	if (state.sessionId !== sessionLog.header.id) {
		throw new Error(`Workflow state session ${state.sessionId} does not match log ${sessionLog.header.id}`);
	}

	const session = createWorkflowSession(state.environment, sessionLog);
	try {
		if (command.type === "prepare_compaction") {
			if (state.phase !== "awaiting_compaction" || !state.compactionRequest) {
				throw new Error(`Cannot prepare_compaction while workflow phase is ${state.phase}`);
			}
			if (state.pendingCompaction) {
				return {
					state,
					sessionLog,
					coreEvents: [],
					sessionEvents: [],
					sessionOps: [],
					nextAction: "complete_compaction",
					preparedCompactionRequest: structuredClone(state.pendingCompaction),
				};
			}

			const request = state.compactionRequest;
			const sessionEvents: AgentSessionEvent[] = [{ type: "compaction_start", reason: request.reason }];
			const model = session.model;
			const preparation = model
				? prepareCompaction(session.sessionManager.getBranch(), session.settingsManager.getCompactionSettings())
				: undefined;
			if (!model || !preparation) {
				const nextState = sessionLoopStateToWorkflowState(state.environment, state.sessionId, {
					...workflowStateToSessionLoopState(state),
					phase: request.willRetry ? "failed" : "completed",
					terminalStatus: request.willRetry ? "failed" : "completed",
					compactionRequest: undefined,
					lastAssistantMessage: undefined,
				});
				return {
					state: nextState,
					sessionLog,
					coreEvents: [],
					sessionEvents: [...sessionEvents, ...createCompactionEvents(undefined, request)],
					sessionOps: [],
					nextAction: request.willRetry ? "failed" : "completed",
				};
			}

			const preparedCompactionRequest: PreparedCompactionRequest = {
				model: structuredClone(model),
				reason: request.reason,
				willRetry: request.willRetry,
				preparation: structuredClone(preparation),
			};
			return {
				state: {
					...state,
					pendingProviderRequest: undefined,
					pendingToolCalls: undefined,
					pendingCompaction: preparedCompactionRequest,
				},
				sessionLog,
				coreEvents: [],
				sessionEvents,
				sessionOps: [],
				nextAction: "complete_compaction",
				preparedCompactionRequest,
			};
		}

		if (command.type === "complete_compaction") {
			if (state.phase !== "awaiting_compaction" || !state.compactionRequest || !state.pendingCompaction) {
				throw new Error(`Cannot complete_compaction while workflow phase is ${state.phase}`);
			}

			const pendingCompaction = state.pendingCompaction;
			if (command.result.firstKeptEntryId !== pendingCompaction.preparation.firstKeptEntryId) {
				throw new Error(
					`Compaction result firstKeptEntryId ${command.result.firstKeptEntryId} does not match prepared request ${pendingCompaction.preparation.firstKeptEntryId}`,
				);
			}

			session.sessionManager.appendCompaction(
				command.result.summary,
				command.result.firstKeptEntryId,
				command.result.tokensBefore,
				command.result.details,
				command.fromExtension,
			);
			session.agent.state.messages = session.sessionManager.buildSessionContext().messages;

			const sessionOps: SessionPersistenceOp[] = [
				{
					type: "append_compaction",
					summary: command.result.summary,
					firstKeptEntryId: command.result.firstKeptEntryId,
					tokensBefore: command.result.tokensBefore,
					details: command.result.details,
					fromExtension: command.fromExtension,
				},
			];
			const hasQueuedMessages = state.queue.steering.length > 0 || state.queue.followUp.length > 0;
			const nextLoopState: SessionLoopState = {
				...workflowStateToSessionLoopState(state),
				compactionRequest: undefined,
				lastAssistantMessage: undefined,
				phase: state.compactionRequest.willRetry || hasQueuedMessages ? "awaiting_assistant" : "completed",
				terminalStatus: state.compactionRequest.willRetry || hasQueuedMessages ? "running" : "completed",
				coreState:
					state.compactionRequest.willRetry || hasQueuedMessages
						? createContinuationLoopState(session)
						: state.coreState
							? structuredClone(state.coreState)
							: undefined,
			};
			const nextAction: WorkflowStepNextAction =
				state.compactionRequest.willRetry || hasQueuedMessages ? "run_assistant_turn" : "completed";
			return {
				state: sessionLoopStateToWorkflowState(state.environment, state.sessionId, nextLoopState),
				sessionLog: captureSessionLogSnapshot(session.sessionManager),
				coreEvents: [],
				sessionEvents: createCompactionEvents(command.result, state.compactionRequest),
				sessionOps,
				nextAction,
			};
		}

		const preparedInput =
			command.type === "prepare_prompt" ? preprocessWorkflowInput(state.input, state.environment) : undefined;
		const sessionResult = await session.stepSessionLoop(
			workflowStateToSessionLoopState(state, preparedInput),
			command,
			signal,
		);
		const nextAction = mapWorkflowNextAction(sessionResult.nextAction);
		return {
			state: sessionLoopStateToWorkflowState(state.environment, state.sessionId, sessionResult.state, {
				pendingProviderRequest:
					nextAction === "complete_provider_response" ? sessionResult.preparedProviderRequest : undefined,
				pendingToolCalls: nextAction === "complete_tool_call" ? sessionResult.toolExecutionRequests : undefined,
			}),
			sessionLog: captureSessionLogSnapshot(session.sessionManager),
			coreEvents: sessionResult.coreEvents,
			sessionEvents: sessionResult.sessionEvents as AgentSessionEvent[],
			sessionOps: sessionResult.sessionOps,
			nextAction,
			preparedProviderRequest: sessionResult.preparedProviderRequest,
			providerRequestPayload: sessionResult.providerRequestPayload,
			toolExecutionRequests: sessionResult.toolExecutionRequests,
			terminalMessages: sessionResult.terminalMessages,
		};
	} finally {
		session.dispose();
	}
}
