import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
	resolve: {
		alias: [
			{
				find: "@mariozechner/pi-ai/env-api-keys",
				replacement: fileURLToPath(new URL("../ai/src/env-api-keys.ts", import.meta.url)),
			},
			{
				find: "@mariozechner/pi-ai/oauth",
				replacement: fileURLToPath(new URL("../ai/src/oauth.ts", import.meta.url)),
			},
			{
				find: "@mupt-ai/pi-agent-core",
				replacement: fileURLToPath(new URL("../agent/src/index.ts", import.meta.url)),
			},
			{
				find: "@mariozechner/pi-ai",
				replacement: fileURLToPath(new URL("../ai/src/index.ts", import.meta.url)),
			},
			{
				find: "@mariozechner/pi-tui",
				replacement: fileURLToPath(new URL("../tui/src/index.ts", import.meta.url)),
			},
		],
	},
	test: {
		globals: true,
		environment: "node",
		testTimeout: 30000, // 30 seconds for API calls
		server: {
			deps: {
				external: [/@silvia-odwyer\/photon-node/],
			},
		},
	},
});
