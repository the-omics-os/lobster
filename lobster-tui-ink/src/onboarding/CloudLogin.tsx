import React, { useState, useCallback, useEffect, useRef } from "react";
import { Box, Text } from "ink";
import { Select, PasswordInput } from "@inkjs/ui";
import { useTheme } from "../hooks/useTheme.js";
import { openInBrowser } from "../utils/openBrowser.js";
import { createServer, type Server } from "http";
import { randomBytes } from "crypto";
import { writeFileSync, mkdirSync, existsSync, chmodSync } from "fs";
import { join, dirname } from "path";
import { homedir } from "os";

const CREDENTIALS_PATH = join(homedir(), ".config", "omics-os", "credentials.json");
const CLOUD_ENDPOINT = "https://app.omics-os.com";

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
  writeFileSync(CREDENTIALS_PATH, JSON.stringify(creds, null, 2), { mode: 0o600 });
  chmodSync(CREDENTIALS_PATH, 0o600);
  chmodSync(dir, 0o700);
}

export function CloudLogin({ onSuccess, onBack }: Props) {
  const theme = useTheme();
  const [step, setStep] = useState<LoginStep>("method");
  const [error, setError] = useState<string>();
  const serverRef = useRef<Server | null>(null);

  useEffect(() => {
    return () => {
      serverRef.current?.close();
    };
  }, []);

  const startBrowserLogin = useCallback(() => {
    setStep("browser-waiting");
    const port = 18765 + Math.floor(Math.random() * 35);
    const state = randomBytes(16).toString("hex");

    const corsHeaders: Record<string, string> = {
      "Access-Control-Allow-Origin": CLOUD_ENDPOINT,
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
      "Access-Control-Allow-Private-Network": "true",
    };

    const server = createServer((req, res) => {
      if (req.method === "OPTIONS" && req.url === "/callback") {
        res.writeHead(204, corsHeaders);
        res.end();
        return;
      }
      if (req.method === "POST" && req.url === "/callback") {
        let body = "";
        req.on("data", (chunk: Buffer) => { body += chunk.toString(); });
        req.on("end", () => {
          try {
            const data = JSON.parse(body);
            if (data.state !== state) {
              res.writeHead(400, corsHeaders);
              res.end("Invalid state");
              return;
            }
            if (!data.access_token || typeof data.access_token !== "string") {
              res.writeHead(400, corsHeaders);
              res.end("Missing access_token");
              setError("OAuth callback missing access_token");
              setStep("error");
              return;
            }
            const expiresIn = typeof data.expires_in === "number" ? data.expires_in : 3600;
            saveCredentials({
              auth_mode: "oauth",
              access_token: data.access_token,
              refresh_token: data.refresh_token,
              id_token: data.id_token,
              client_id: data.client_id,
              token_expiry: new Date(Date.now() + expiresIn * 1000).toISOString(),
              endpoint: CLOUD_ENDPOINT,
              user_id: data.user_id,
              email: data.email,
              tier: data.tier,
            });
            res.writeHead(200, { ...corsHeaders, "Content-Type": "text/html" });
            res.end("<html><body><h2>Login successful!</h2><p>You can close this tab.</p></body></html>");
            server.close();
            serverRef.current = null;
            setStep("success");
            setTimeout(onSuccess, 500);
          } catch {
            res.writeHead(400, corsHeaders);
            res.end("Parse error");
            setError("Failed to parse callback");
            setStep("error");
          }
        });
      } else {
        res.writeHead(404);
        res.end();
      }
    });

    serverRef.current = server;

    server.listen(port, "127.0.0.1", () => {
      const url = `${CLOUD_ENDPOINT}/auth/cli?port=${port}&state=${state}`;
      openInBrowser(url);
    });

    server.on("error", () => {
      setError("Could not start local callback server. Try API key login.");
      setStep("error");
    });

    setTimeout(() => {
      if (serverRef.current === server) {
        server.close();
        serverRef.current = null;
        setError("Login timed out. Try API key login instead.");
        setStep("error");
      }
    }, 120_000);
  }, [onSuccess]);

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
        signal: AbortSignal.timeout(15_000),
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
        user_id: data.user_id ?? "",
        email: data.email ?? "",
        tier: data.tier ?? "free",
      });
      setStep("success");
      setTimeout(onSuccess, 500);
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
          Complete the login in your browser. Waiting for callback... (120s timeout)
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
