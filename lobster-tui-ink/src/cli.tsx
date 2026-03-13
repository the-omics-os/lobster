#!/usr/bin/env bun
import React from "react";
import { render } from "ink";
import { readFileSync } from "fs";
import { parseArgs } from "util";
import { resolveConfig } from "./config.js";
import { App } from "./App.js";
import { InitWizard } from "./wizard/InitWizard.js";
import type { WizardManifest, WizardResult } from "./wizard/types.js";

const { values } = parseArgs({
  args: Bun.argv.slice(2),
  options: {
    "api-url": { type: "string" },
    "session-id": { type: "string" },
    token: { type: "string" },
    cloud: { type: "boolean", default: false },
    init: { type: "boolean", default: false },
    "manifest-file": { type: "string" },
    help: { type: "boolean", short: "h", default: false },
  },
  strict: true,
});

if (values.help) {
  console.log(`lobster-chat — Lobster AI terminal interface

Usage:
  lobster-chat [options]

Options:
  --api-url <url>           Backend API URL (default: http://localhost:8000)
  --session-id <id>         Resume an existing session
  --token <token>           Authentication token for cloud mode
  --cloud                   Connect to Omics-OS Cloud (app.omics-os.com)
  --init                    Run the init wizard
  --manifest-file <path>    Path to wizard manifest JSON file
  -h, --help                Show this help message`);
  process.exit(0);
}

if (values.init) {
  const manifestPath = values["manifest-file"];
  if (!manifestPath) {
    console.error("Error: --init requires --manifest-file <path>");
    process.exit(1);
  }

  let manifest: WizardManifest;
  try {
    manifest = JSON.parse(readFileSync(manifestPath, "utf-8")) as WizardManifest;
  } catch (e) {
    console.error(`Error reading manifest: ${e}`);
    process.exit(1);
  }

  function handleWizardComplete(result: WizardResult) {
    // Write result to stdout as JSON for the Python launcher to consume
    process.stdout.write(JSON.stringify(result) + "\n");
    process.exit(0);
  }

  render(<InitWizard manifest={manifest} onComplete={handleWizardComplete} />);
} else {
  const config = resolveConfig({
    apiUrl: values["api-url"],
    sessionId: values["session-id"],
    token: values.token,
    cloud: values.cloud,
  });

  render(<App config={config} />);
}
