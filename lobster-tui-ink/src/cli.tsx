import React from "react";
import { render } from "ink";
import { readFileSync, writeFileSync } from "fs";
import { parseArgs } from "util";
import { resolveConfig, readStoredCredentials } from "./config.js";
import { App } from "./App.js";
import { ThemeProvider } from "./hooks/useTheme.js";
import { InitWizard } from "./wizard/InitWizard.js";
import { CloudOnboarding } from "./onboarding/index.js";
import type { WizardManifest, WizardResult } from "./wizard/types.js";

// Prevent silent crashes from unhandled promise rejections
process.on("unhandledRejection", (reason) => {
  const msg = reason instanceof Error ? reason.stack ?? reason.message : String(reason);
  process.stderr.write(`\n[lobster-chat] Unhandled rejection: ${msg}\n`);
});
process.on("uncaughtException", (error) => {
  process.stderr.write(`\n[lobster-chat] Uncaught exception: ${error.stack ?? error.message}\n`);
  process.exit(1);
});

const { values } = parseArgs({
  args: process.argv.slice(2),
  options: {
    "api-url": { type: "string" },
    "session-id": { type: "string" },
    resume: { type: "string" },
    "resume-session": { type: "boolean", default: false },
    token: { type: "string" },
    cloud: { type: "boolean", default: false },
    "project-id": { type: "string" },
    "stream-endpoint": { type: "string" },
    init: { type: "boolean", default: false },
    "manifest-file": { type: "string" },
    "result-file": { type: "string" },
    help: { type: "boolean", short: "h", default: false },
  },
  strict: true,
  allowPositionals: true,
});

if (values.help) {
  console.log(`lobster-chat — Lobster AI terminal interface

Usage:
  lobster-chat [options]

Options:
  --api-url <url>           Backend API URL (default: http://localhost:8000)
  --session-id <id>         Resume an existing session
  --resume <id>             Alias for --session-id (reconnect after disconnect)
  --token <token>           Authentication token for cloud mode
  --cloud                   Connect to Omics-OS Cloud (app.omics-os.com)
  --project-id <id>         Associate session with a cloud project
  --stream-endpoint <url>   Custom stream base (e.g., https://stream.omics-os.com/api/v1)
  --init                    Run the init wizard
  --manifest-file <path>    Path to wizard manifest JSON file
  --result-file <path>      Write init wizard result JSON to a file
  -h, --help                Show this help message

Non-interactive queries:
  Use "lobster query" instead (runs directly in Python).`);
  process.exit(0);
}

if (values.init) {
  const manifestPath = values["manifest-file"];
  const resultFile = values["result-file"];
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
    const payload = JSON.stringify(result) + "\n";
    if (resultFile) {
      try {
        writeFileSync(resultFile, payload, "utf-8");
      } catch (e) {
        console.error(`Error writing result file: ${e}`);
        process.exit(1);
      }
    } else {
      process.stdout.write(payload);
    }
    process.exit(0);
  }

  render(
    <ThemeProvider>
      <InitWizard manifest={manifest} onComplete={handleWizardComplete} />
    </ThemeProvider>,
  );
} else {
  if (values.token) {
    console.error("Warning: --token passes secrets via CLI args (visible in ps). Use LOBSTER_TOKEN env var instead.");
  }

  const hasExplicitMode = values.cloud
    || values["api-url"]
    || values.token
    || process.env.LOBSTER_TOKEN
    || values["session-id"]
    || values.resume
    || values["resume-session"]
    || values["project-id"]
    || values["stream-endpoint"];
  const storedCreds = !hasExplicitMode ? readStoredCredentials() : undefined;

  if (!hasExplicitMode && !storedCreds) {
    render(
      <ThemeProvider>
        <CloudOnboarding />
      </ThemeProvider>,
    );
  } else {
    // --token or LOBSTER_TOKEN without --api-url implies cloud mode
    const hasToken = !!(values.token || process.env.LOBSTER_TOKEN);
    const effectiveCloud = values.cloud || (!values["api-url"] && (!!storedCreds || hasToken));

    const config = resolveConfig({
      apiUrl: values["api-url"],
      streamApiUrl: values["stream-endpoint"],
      sessionId: values["session-id"] ?? values.resume,
      projectId: values["project-id"],
      token: values.token,
      cloud: effectiveCloud,
      isResume: values["resume-session"] || Boolean(values.resume),
    });

    render(
      <ThemeProvider>
        <App config={config} />
      </ThemeProvider>,
    );
  }
}
