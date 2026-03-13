#!/usr/bin/env bun
import React from "react";
import { render } from "ink";
import { parseArgs } from "util";
import { resolveConfig } from "./config.js";
import { App } from "./App.js";

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

const config = resolveConfig({
  apiUrl: values["api-url"],
  sessionId: values["session-id"],
  token: values.token,
  cloud: values.cloud,
});

render(<App config={config} />);
