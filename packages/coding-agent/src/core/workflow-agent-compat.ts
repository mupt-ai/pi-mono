import { Agent } from "@mupt-ai/pi-agent-core";

type WorkflowCompatibleAgentHooks = {
	convertToLlm: Agent["convertToLlm"];
	transformContext: Agent["transformContext"];
	onPayload: Agent["onPayload"];
	beforeToolCall: Agent["beforeToolCall"];
	afterToolCall: Agent["afterToolCall"];
};

type WorkflowCompatibleAgentHookName = keyof WorkflowCompatibleAgentHooks;

const defaultAgent = new Agent();
const defaultHooks = captureWorkflowCompatibleAgentHooks(defaultAgent);
const compatibleAgentHooks = new WeakMap<Agent, WorkflowCompatibleAgentHooks>();
const workflowCompatibleHookNames: WorkflowCompatibleAgentHookName[] = [
	"convertToLlm",
	"transformContext",
	"onPayload",
	"beforeToolCall",
	"afterToolCall",
];

function captureWorkflowCompatibleAgentHooks(agent: Agent): WorkflowCompatibleAgentHooks {
	return {
		convertToLlm: agent.convertToLlm,
		transformContext: agent.transformContext,
		onPayload: agent.onPayload,
		beforeToolCall: agent.beforeToolCall,
		afterToolCall: agent.afterToolCall,
	};
}

export function markWorkflowSnapshotCompatibleAgent(agent: Agent): void {
	compatibleAgentHooks.set(agent, captureWorkflowCompatibleAgentHooks(agent));
}

export function getWorkflowSnapshotUnsupportedDirectAgentHooks(agent: Agent): string[] {
	const currentHooks = captureWorkflowCompatibleAgentHooks(agent);
	const compatibleHooks = compatibleAgentHooks.get(agent) ?? defaultHooks;

	return workflowCompatibleHookNames.filter((hookName) => currentHooks[hookName] !== compatibleHooks[hookName]);
}
