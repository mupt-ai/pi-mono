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

/**
 * Record an Agent's current direct hooks as the workflow-compatible baseline.
 *
 * Workflow snapshots can't capture closures, so the kernel needs a way to
 * decide whether an Agent has been customized away from a known-good shape.
 * Call this right after constructing an Agent inside the workflow runtime
 * (or anywhere a host knows the current hooks are intentional and replayable).
 * Subsequent calls to `getWorkflowSnapshotUnsupportedDirectAgentHooks` compare
 * against this baseline instead of the default `new Agent()` hooks.
 */
export function markWorkflowSnapshotCompatibleAgent(agent: Agent): void {
	compatibleAgentHooks.set(agent, captureWorkflowCompatibleAgentHooks(agent));
}

/**
 * Return the names of any direct Agent hooks that have been overridden away
 * from the registered baseline (or, if unregistered, the default Agent hooks).
 *
 * Used by the workflow snapshot path to refuse capture when an Agent uses
 * custom `convertToLlm` / `transformContext` / `onPayload` / `beforeToolCall` /
 * `afterToolCall` closures, since those can't be serialized into a snapshot.
 */
export function getWorkflowSnapshotUnsupportedDirectAgentHooks(agent: Agent): string[] {
	const currentHooks = captureWorkflowCompatibleAgentHooks(agent);
	const compatibleHooks = compatibleAgentHooks.get(agent) ?? defaultHooks;

	return workflowCompatibleHookNames.filter((hookName) => currentHooks[hookName] !== compatibleHooks[hookName]);
}
