import { existsSync } from "fs";
import { join } from "path";

function hasVertexAdcCredentials(): boolean {
	const gacPath = process.env.GOOGLE_APPLICATION_CREDENTIALS;
	if (gacPath) {
		return existsSync(gacPath);
	}
	return existsSync(join(process.env.HOME ?? "", ".config", "gcloud", "application_default_credentials.json"));
}

/**
 * Local copy of pi-ai env auth resolution.
 *
 * coding-agent must not import getEnvApiKey from the pi-ai root export here,
 * because the package root is eagerly loaded by Agent Host's pi-worker and that
 * dependency edge has proven fragile in production bootstrap.
 */
export function getEnvApiKey(provider: string): string | undefined {
	if (provider === "github-copilot") {
		return process.env.COPILOT_GITHUB_TOKEN || process.env.GH_TOKEN || process.env.GITHUB_TOKEN;
	}

	if (provider === "anthropic") {
		return process.env.ANTHROPIC_OAUTH_TOKEN || process.env.ANTHROPIC_API_KEY;
	}

	if (provider === "google-vertex") {
		if (process.env.GOOGLE_CLOUD_API_KEY) {
			return process.env.GOOGLE_CLOUD_API_KEY;
		}

		const hasCredentials = hasVertexAdcCredentials();
		const hasProject = !!(process.env.GOOGLE_CLOUD_PROJECT || process.env.GCLOUD_PROJECT);
		const hasLocation = !!process.env.GOOGLE_CLOUD_LOCATION;
		if (hasCredentials && hasProject && hasLocation) {
			return "<authenticated>";
		}
	}

	if (provider === "amazon-bedrock") {
		if (
			process.env.AWS_PROFILE ||
			(process.env.AWS_ACCESS_KEY_ID && process.env.AWS_SECRET_ACCESS_KEY) ||
			process.env.AWS_BEARER_TOKEN_BEDROCK ||
			process.env.AWS_CONTAINER_CREDENTIALS_RELATIVE_URI ||
			process.env.AWS_CONTAINER_CREDENTIALS_FULL_URI ||
			process.env.AWS_WEB_IDENTITY_TOKEN_FILE
		) {
			return "<authenticated>";
		}
	}

	const envMap: Record<string, string> = {
		openai: "OPENAI_API_KEY",
		"azure-openai-responses": "AZURE_OPENAI_API_KEY",
		google: "GEMINI_API_KEY",
		groq: "GROQ_API_KEY",
		cerebras: "CEREBRAS_API_KEY",
		xai: "XAI_API_KEY",
		openrouter: "OPENROUTER_API_KEY",
		"vercel-ai-gateway": "AI_GATEWAY_API_KEY",
		zai: "ZAI_API_KEY",
		mistral: "MISTRAL_API_KEY",
		minimax: "MINIMAX_API_KEY",
		"minimax-cn": "MINIMAX_CN_API_KEY",
		huggingface: "HF_TOKEN",
		opencode: "OPENCODE_API_KEY",
		"opencode-go": "OPENCODE_API_KEY",
		"kimi-coding": "KIMI_API_KEY",
	};

	const envVar = envMap[provider];
	return envVar ? process.env[envVar] : undefined;
}
