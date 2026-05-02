import React, { useState, useCallback, useEffect, useRef } from "react";
import { Box, Text } from "ink";
import { Select, PasswordInput } from "@inkjs/ui";
import { useTheme } from "../hooks/useTheme.js";
import { openInBrowser } from "../utils/openBrowser.js";
import { createServer, type Server } from "http";
import { randomBytes, timingSafeEqual } from "crypto";
import {
  writeFileSync, mkdirSync, existsSync, chmodSync, renameSync,
} from "fs";
import { join, dirname } from "path";
import { homedir } from "os";

const CREDENTIALS_PATH = join(homedir(), ".config", "omics-os", "credentials.json");
const CLOUD_ENDPOINT = "https://app.omics-os.com";
const LOGIN_TIMEOUT_MS = 120_000;
const MAX_CALLBACK_BODY_BYTES = 16 * 1024;
const CALLBACK_REQUEST_TIMEOUT_MS = 10_000;
const POST_LOGIN_DELAY_MS = 500;
const API_VALIDATE_TIMEOUT_MS = 15_000;

type LoginStep = "method" | "browser-waiting" | "api-key" | "validating" | "success" | "error";

interface Props {
  onSuccess: () => void;
  onBack: () => void;
}

function saveCredentials(creds: Record<string, unknown>) {
  const dir = dirname(CREDENTIALS_PATH);
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true, mode: 0o700 });
  }
  const tmpPath = CREDENTIALS_PATH + ".tmp";
  writeFileSync(tmpPath, JSON.stringify(creds, null, 2), { mode: 0o600 });
  chmodSync(tmpPath, 0o600);
  renameSync(tmpPath, CREDENTIALS_PATH);
  chmodSync(dir, 0o700);
}

function safeStateCompare(expected: string, actual: string): boolean {
  if (expected.length !== actual.length) return false;
  try {
    return timingSafeEqual(Buffer.from(expected), Buffer.from(actual));
  } catch {
    return false;
  }
}

export function CloudLogin({ onSuccess, onBack }: Props) {
  const theme = useTheme();
  const [step, setStep] = useState<LoginStep>("method");
  const [error, setError] = useState<string>();
  const serverRef = useRef<Server | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const stopBrowserLogin = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    if (serverRef.current) {
      serverRef.current.close();
      serverRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => { stopBrowserLogin(); };
  }, [stopBrowserLogin]);

  const startBrowserLogin = useCallback(() => {
    stopBrowserLogin();
    setStep("browser-waiting");
    const state = randomBytes(16).toString("hex");

    const corsHeaders: Record<string, string> = {
      "Access-Control-Allow-Origin": CLOUD_ENDPOINT,
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
      "Access-Control-Allow-Private-Network": "true",
      "Vary": "Origin",
    };

    const server = createServer((req, res) => {
      const origin = req.headers.origin;
      if (origin && origin !== CLOUD_ENDPOINT) {
        res.writeHead(403, { "Vary": "Origin" });
        res.end("Forbidden");
        return;
      }

      if (req.method === "OPTIONS" && req.url === "/callback") {
        res.writeHead(204, corsHeaders);
        res.end();
        return;
      }
      if (req.method === "POST" && req.url === "/callback") {
        let body = "";
        let bytes = 0;
        req.on("data", (chunk: Buffer) => {
          bytes += chunk.length;
          if (bytes > MAX_CALLBACK_BODY_BYTES) {
            res.writeHead(413, corsHeaders);
            res.end("Payload too large");
            req.destroy();
            return;
          }
          body += chunk.toString();
        });
        req.on("end", () => {
          if (bytes > MAX_CALLBACK_BODY_BYTES) return;
          try {
            const data = JSON.parse(body);
            if (typeof data.state !== "string" || !safeStateCompare(state, data.state)) {
              res.writeHead(400, corsHeaders);
              res.end("Invalid state");
              return;
            }
            if (!data.access_token || typeof data.access_token !== "string") {
              res.writeHead(400, corsHeaders);
              res.end("Missing access_token");
              setError("OAuth callback missing access_token");
              setStep("error");
              stopBrowserLogin();
              return;
            }
            const expiresIn = typeof data.expires_in === "number" && data.expires_in > 0
              ? data.expires_in : 3600;
            saveCredentials({
              auth_mode: "oauth",
              access_token: data.access_token,
              refresh_token: typeof data.refresh_token === "string" ? data.refresh_token : undefined,
              id_token: typeof data.id_token === "string" ? data.id_token : undefined,
              client_id: typeof data.client_id === "string" ? data.client_id : undefined,
              token_expiry: new Date(Date.now() + expiresIn * 1000).toISOString(),
              endpoint: CLOUD_ENDPOINT,
              user_id: typeof data.user_id === "string" ? data.user_id : "",
              email: typeof data.email === "string" ? data.email : "",
              tier: typeof data.tier === "string" ? data.tier : "free",
            });
            res.writeHead(200, { ...corsHeaders, "Content-Type": "text/html" });
            res.end("<html><body><h2>Login successful!</h2><p>You can close this tab.</p></body></html>");
            stopBrowserLogin();
            setStep("success");
            setTimeout(onSuccess, POST_LOGIN_DELAY_MS);
          } catch {
            res.writeHead(400, corsHeaders);
            res.end("Parse error");
            setError("Failed to parse callback");
            setStep("error");
            stopBrowserLogin();
          }
        });
      } else {
        res.writeHead(404);
        res.end();
      }
    });

    server.requestTimeout = CALLBACK_REQUEST_TIMEOUT_MS;
    server.headersTimeout = CALLBACK_REQUEST_TIMEOUT_MS;
    serverRef.current = server;

    server.listen(0, "127.0.0.1", () => {
      const addr = server.address();
      const port = typeof addr === "object" && addr ? addr.port : 0;
      if (!port) {
        setError("Could not start local callback server. Try API key login.");
        setStep("error");
        stopBrowserLogin();
        return;
      }
      const url = `${CLOUD_ENDPOINT}/auth/cli?port=${port}&state=${state}`;
      openInBrowser(url);
    });

    server.on("error", () => {
      setError("Could not start local callback server. Try API key login.");
      setStep("error");
      stopBrowserLogin();
    });

    timeoutRef.current = setTimeout(() => {
      setError("Login timed out. Try API key login instead.");
      setStep("error");
      stopBrowserLogin();
    }, LOGIN_TIMEOUT_MS);
  }, [onSuccess, stopBrowserLogin]);

  const handleApiKey = useCallback(async (key: string) => {
    if (!key.startsWith("omk_")) {
      setError("API key must start with 'omk_'. Get one at app.omics-os.com/settings/api-keys");
      setStep("error");
      return;
    }
    setStep("validating");
    try {
      const resp = await fetch(`${CLOUD_ENDPOINT}/api/v1/account`, {
        headers: { "X-API-Key": key },
        signal: AbortSignal.timeout(API_VALIDATE_TIMEOUT_MS),
      });
      if (!resp.ok) {
        setError(`Invalid API key (${resp.status}). Get one at app.omics-os.com/settings/api-keys`);
        setStep("error");
        return;
      }
      const data = (await resp.json()) as Record<string, unknown>;
      saveCredentials({
        auth_mode: "api_key",
        api_key: key,
        endpoint: CLOUD_ENDPOINT,
        user_id: typeof data.user_id === "string" ? data.user_id : "",
        email: typeof data.email === "string" ? data.email : "",
        tier: typeof data.tier === "string" ? data.tier : "free",
      });
      setStep("success");
      setTimeout(onSuccess, POST_LOGIN_DELAY_MS);
    } catch {
      setError("Network error. Check your internet connection.");
      setStep("error");
    }
  }, [onSuccess]);

  if (step === "method") {
    return (
      <Box flexDirection="column" paddingX={2} marginY={1}>
        <Text bold color={theme.primary}>Cloud Login</Text>
        <Text color={theme.textMuted}>Choose authentication method:</Text>
        <Box marginTop={1}>
          <Select
            options={[
              { label: "Browser login (recommended)", value: "browser" },
              { label: "Paste API key", value: "api-key" },
              { label: "← Back", value: "back" },
            ]}
            onChange={(value) => {
              if (value === "browser") startBrowserLogin();
              else if (value === "api-key") setStep("api-key");
              else onBack();
            }}
          />
        </Box>
      </Box>
    );
  }

  if (step === "browser-waiting") {
    return (
      <Box flexDirection="column" paddingX={2} marginY={1}>
        <Text bold color={theme.primary}>Cloud Login</Text>
        <Text color={theme.info}>Opening browser...</Text>
        <Text color={theme.textMuted}>
          Complete the login in your browser. Waiting for callback...
        </Text>
      </Box>
    );
  }

  if (step === "api-key") {
    return (
      <Box flexDirection="column" paddingX={2} marginY={1}>
        <Text bold color={theme.primary}>Cloud Login — API Key</Text>
        <Text color={theme.textMuted}>
          Get your key at app.omics-os.com/settings/api-keys
        </Text>
        <Box marginTop={1}>
          <Text>API Key: </Text>
          <PasswordInput
            placeholder="omk_..."
            onSubmit={(value) => { handleApiKey(value); }}
          />
        </Box>
      </Box>
    );
  }

  if (step === "validating") {
    return (
      <Box flexDirection="column" paddingX={2} marginY={1}>
        <Text color={theme.info}>Validating...</Text>
      </Box>
    );
  }

  if (step === "success") {
    return (
      <Box flexDirection="column" paddingX={2} marginY={1}>
        <Text bold color={theme.success}>Login successful!</Text>
        <Text color={theme.textMuted}>Starting Lobster AI...</Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" paddingX={2} marginY={1}>
      <Text bold color={theme.error}>Login failed</Text>
      <Text color={theme.textMuted}>{error}</Text>
      <Box marginTop={1}>
        <Select
          options={[
            { label: "Try again", value: "retry" },
            { label: "Use API key instead", value: "api-key" },
            { label: "← Back", value: "back" },
          ]}
          onChange={(value) => {
            setError(undefined);
            if (value === "retry") startBrowserLogin();
            else if (value === "api-key") setStep("api-key");
            else onBack();
          }}
        />
      </Box>
    </Box>
  );
}
