#!/usr/bin/env bun
import { parseArgs } from "util";

const { values } = parseArgs({
  args: Bun.argv.slice(2),
  options: {
    "api-url": { type: "string" },
    "session-id": { type: "string" },
    token: { type: "string" },
    cloud: { type: "boolean", default: false },
    help: { type: "boolean", short: "h", default: false },
  },
  strict: true,
});

if (values.help) {
  console.log(`lobster-chat — Lobster AI terminal interface

Usage:
  lobster-chat [options]

Options:
  --api-url <url>       Backend API URL (default: http://localhost:8000)
  --session-id <id>     Resume an existing session
  --token <token>       Authentication token for cloud mode
  --cloud               Connect to Omics-OS Cloud (app.omics-os.com)
  -h, --help            Show this help message`);
  process.exit(0);
}

const apiUrl = values["api-url"] ?? (values.cloud ? "https://app.omics-os.com/api/v1" : "http://localhost:8000");
const sessionId = values["session-id"];
const token = values.token;

console.log(`lobster-chat v0.1.0`);
console.log(`API: ${apiUrl}`);
if (sessionId) console.log(`Session: ${sessionId}`);
console.log("(App shell not yet wired — see Step 1.2)");
