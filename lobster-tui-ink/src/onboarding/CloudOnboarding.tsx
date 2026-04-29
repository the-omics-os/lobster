import React, { useState } from "react";
import { Box, Text } from "ink";
import { useTheme } from "../hooks/useTheme.js";
import { ModeChooser } from "./ModeChooser.js";
import { CloudLogin } from "./CloudLogin.js";
import { App } from "../App.js";
import { resolveConfig } from "../config.js";

type Step = "choose" | "login" | "ready" | "local-guidance";

export function CloudOnboarding() {
  const theme = useTheme();
  const [step, setStep] = useState<Step>("choose");

  if (step === "choose") {
    return (
      <ModeChooser
        onSelect={(selected) => {
          if (selected === "cloud") {
            setStep("login");
          } else {
            setStep("local-guidance");
          }
        }}
      />
    );
  }

  if (step === "login") {
    return (
      <CloudLogin
        onSuccess={() => setStep("ready")}
        onBack={() => setStep("choose")}
      />
    );
  }

  if (step === "ready") {
    const config = resolveConfig({ cloud: true });
    return <App config={config} />;
  }

  return (
    <Box flexDirection="column" paddingX={2} marginY={1}>
      <Text bold color={theme.primary}>Local Mode Setup</Text>
      <Text color={theme.textMuted}>
        Lobster AI local mode requires Python. Run these commands:
      </Text>
      <Box marginTop={1} flexDirection="column">
        <Text>  1. curl -LsSf https://astral.sh/uv/install.sh | sh</Text>
        <Text>  2. uv tool install 'lobster-ai[full]'</Text>
        <Text>  3. lobster init</Text>
        <Text>  4. lobster chat</Text>
      </Box>
      <Text color={theme.textMuted}>
        Or install via pip: pip install 'lobster-ai[full]'
      </Text>
    </Box>
  );
}
