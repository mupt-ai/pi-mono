#!/usr/bin/env node

import { readFileSync, writeFileSync } from "fs";

const BUMP_TYPE = process.argv[2];
const ALLOWED_BUMPS = new Set(["patch", "minor"]);

if (!ALLOWED_BUMPS.has(BUMP_TYPE)) {
	console.error("Usage: node scripts/version-github-packages.mjs <patch|minor>");
	process.exit(1);
}

const agentPath = "packages/agent/package.json";
const codingAgentPath = "packages/coding-agent/package.json";

function readPackageJson(path) {
	return JSON.parse(readFileSync(path, "utf-8"));
}

function writePackageJson(path, data) {
	writeFileSync(path, `${JSON.stringify(data, null, "\t")}\n`);
}

function bumpVersion(version, bumpType) {
	const match = /^(\d+)\.(\d+)\.(\d+)$/.exec(version);
	if (!match) {
		throw new Error(`Unsupported version format: ${version}`);
	}

	const major = Number(match[1]);
	const minor = Number(match[2]);
	const patch = Number(match[3]);

	if (bumpType === "patch") {
		return `${major}.${minor}.${patch + 1}`;
	}

	return `${major}.${minor + 1}.0`;
}

const agentPackage = readPackageJson(agentPath);
const codingAgentPackage = readPackageJson(codingAgentPath);

if (agentPackage.version !== codingAgentPackage.version) {
	console.error(
		`Expected agent and coding-agent to share a version, got ${agentPackage.version} and ${codingAgentPackage.version}`,
	);
	process.exit(1);
}

const nextVersion = bumpVersion(agentPackage.version, BUMP_TYPE);

agentPackage.version = nextVersion;
codingAgentPackage.version = nextVersion;
codingAgentPackage.dependencies["@mupt-ai/pi-agent-core"] = `^${nextVersion}`;

writePackageJson(agentPath, agentPackage);
writePackageJson(codingAgentPath, codingAgentPackage);

console.log(`Updated GitHub Packages release version to ${nextVersion}`);
console.log(`  ${agentPackage.name}: ${nextVersion}`);
console.log(`  ${codingAgentPackage.name}: ${nextVersion}`);
