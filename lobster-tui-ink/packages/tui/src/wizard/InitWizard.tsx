import React, { useState, useCallback } from "react";
import { Box, Text } from "ink";
import {
  Select,
  MultiSelect,
  TextInput,
  PasswordInput,
  ConfirmInput,
} from "@inkjs/ui";
import type {
  WizardManifest,
  WizardResult,
  ProviderDef,
  CredentialField,
  AuthMethodDef,
} from "./types.js";
import { theme } from "../theme.js";

type WizardStep =
  | "packages"
  | "provider"
  | "authMethod"
  | "credentials"
  | "model"
  | "optionalKeys";

const STEP_LABELS: Record<WizardStep, string> = {
  packages: "Packages",
  provider: "Provider",
  authMethod: "Auth",
  credentials: "Credentials",
  model: "Model",
  optionalKeys: "Options",
};

const STEPS: WizardStep[] = [
  "packages",
  "provider",
  "authMethod",
  "credentials",
  "model",
  "optionalKeys",
];

interface WizardState {
  selectedPackages: string[];
  provider: ProviderDef | null;
  credentials: Record<string, string>;
  model: string | null;
  profile: string | null;
  oauthAuthenticated: boolean;
}

interface InitWizardProps {
  manifest: WizardManifest;
  onComplete: (result: WizardResult) => void;
}

export function InitWizard({ manifest, onComplete }: InitWizardProps) {
  const [step, setStep] = useState<WizardStep>("packages");
  const [state, setState] = useState<WizardState>({
    selectedPackages: [],
    provider: null,
    credentials: {},
    model: null,
    profile: null,
    oauthAuthenticated: false,
  });

  const stepIndex = STEPS.indexOf(step);

  const handlePackagesSubmit = useCallback((values: string[]) => {
    setState((s) => ({ ...s, selectedPackages: values }));
    setStep("provider");
  }, []);

  const handleProviderSelect = useCallback(
    (value: string) => {
      const provider = manifest.providers.find((p) => p.name === value);
      if (!provider) return;
      setState((s) => ({ ...s, provider, oauthAuthenticated: false }));

      const hasOAuth = provider.auth_methods?.some((am) => am.type === "oauth");
      if (hasOAuth && provider.auth_methods.length > 1) {
        // Provider supports multiple auth methods — let user choose
        setStep("authMethod");
      } else if (provider.credentials.length === 0) {
        // Skip credentials (e.g. Ollama), go to model step
        advanceToModelStep(provider);
      } else {
        setStep("credentials");
      }
    },
    [manifest.providers]
  );

  const handleAuthMethodSelect = useCallback(
    (method: string) => {
      if (method === "oauth") {
        // OAuth selected — mark it and skip credential form
        setState((s) => ({ ...s, oauthAuthenticated: true, credentials: {} }));
        if (state.provider) {
          advanceToModelStep(state.provider);
        }
      } else {
        // API key selected — proceed to credential form
        setState((s) => ({ ...s, oauthAuthenticated: false }));
        setStep("credentials");
      }
    },
    [state.provider]
  );

  function advanceToModelStep(provider: ProviderDef) {
    if (provider.model_selection === "managed") {
      // Skip model step entirely for managed providers
      setStep("optionalKeys");
    } else {
      setStep("model");
    }
  }

  const handleCredentialsComplete = useCallback(
    (creds: Record<string, string>) => {
      setState((s) => ({ ...s, credentials: creds }));
      if (state.provider) {
        advanceToModelStep(state.provider);
      }
    },
    [state.provider]
  );

  const handleModelComplete = useCallback(
    (model: string | null, profile: string | null) => {
      setState((s) => ({ ...s, model, profile }));
      setStep("optionalKeys");
    },
    []
  );

  const handleOptionalKeysComplete = useCallback(
    (optionalKeys: Record<string, string>, smartStd: boolean) => {
      if (!state.provider) return;
      onComplete({
        selectedPackages: state.selectedPackages,
        provider: state.provider.name,
        credentials: state.credentials,
        model: state.model,
        profile: state.profile,
        optionalKeys,
        smartStandardization: smartStd,
        ...(state.oauthAuthenticated ? { oauthAuthenticated: true } : {}),
      });
    },
    [state, onComplete]
  );

  return (
    <Box flexDirection="column" paddingX={1}>
      <Box marginBottom={1}>
        <Text bold color={theme.primary}>
          Lobster AI
        </Text>
        <Text> — Setup Wizard</Text>
      </Box>

      <StepIndicator current={stepIndex} />

      {step === "packages" && (
        <PackageStep manifest={manifest} onSubmit={handlePackagesSubmit} />
      )}

      {step === "provider" && (
        <ProviderStep manifest={manifest} onSelect={handleProviderSelect} />
      )}

      {step === "authMethod" && state.provider && (
        <AuthMethodStep
          provider={state.provider}
          onSelect={handleAuthMethodSelect}
        />
      )}

      {step === "credentials" && state.provider && (
        <CredentialStep
          provider={state.provider}
          onComplete={handleCredentialsComplete}
        />
      )}

      {step === "model" && state.provider && (
        <ModelStep
          provider={state.provider}
          manifest={manifest}
          onComplete={handleModelComplete}
        />
      )}

      {step === "optionalKeys" && (
        <OptionalKeysStep onComplete={handleOptionalKeysComplete} />
      )}
    </Box>
  );
}

function StepIndicator({ current }: { current: number }) {
  return (
    <Box marginBottom={1} gap={1}>
      {STEPS.map((s, i) => (
        <Text
          key={s}
          color={i === current ? theme.info : i < current ? theme.success : theme.textMuted}
          bold={i === current}
        >
          {i < current ? "\u2713" : i === current ? "\u25CF" : "\u25CB"}{" "}
          {STEP_LABELS[s]}
        </Text>
      ))}
    </Box>
  );
}

/* --- Step 1: Agent Packages --- */

function PackageStep({
  manifest,
  onSubmit,
}: {
  manifest: WizardManifest;
  onSubmit: (values: string[]) => void;
}) {
  const options = manifest.agent_packages.map((pkg) => ({
    label: `${pkg.package_name}${pkg.experimental ? " [experimental]" : ""} \u2014 ${pkg.description}`,
    value: pkg.package_name,
  }));

  const defaults = manifest.agent_packages
    .filter((p) => p.published && !p.experimental)
    .map((p) => p.package_name);

  return (
    <Box flexDirection="column">
      <Text bold>Select agent packages to install:</Text>
      <Text color={theme.textMuted}>Space to toggle, Enter to confirm</Text>
      <Box marginTop={1}>
        <MultiSelect
          options={options}
          defaultValue={defaults}
          onSubmit={onSubmit}
        />
      </Box>
    </Box>
  );
}

/* --- Step 2: Provider Selection --- */

function ProviderStep({
  manifest,
  onSelect,
}: {
  manifest: WizardManifest;
  onSelect: (value: string) => void;
}) {
  const options = manifest.providers.map((p) => ({
    label: `${p.display_name} \u2014 ${p.description}`,
    value: p.name,
  }));

  return (
    <Box flexDirection="column">
      <Text bold>Select your LLM provider:</Text>
      <Box marginTop={1}>
        <Select options={options} onChange={onSelect} />
      </Box>
    </Box>
  );
}

/* --- Step 2b: Auth Method Selection (OAuth vs API key) --- */

function AuthMethodStep({
  provider,
  onSelect,
}: {
  provider: ProviderDef;
  onSelect: (method: string) => void;
}) {
  const methods = provider.auth_methods ?? [];
  const options = methods.map((am) => ({
    label: am.label,
    value: am.type,
  }));
  const defaultMethod = methods.find((am) => am.is_default)?.type;

  return (
    <Box flexDirection="column">
      <Text bold>How would you like to authenticate with {provider.display_name}?</Text>
      <Box marginTop={1}>
        <Select options={options} defaultValue={defaultMethod} onChange={onSelect} />
      </Box>
      <Text color={theme.textMuted} dimColor>
        OAuth lets you use your existing Claude Pro/Max subscription.
      </Text>
    </Box>
  );
}

/* --- Step 3: Credentials --- */

function CredentialStep({
  provider,
  onComplete,
}: {
  provider: ProviderDef;
  onComplete: (creds: Record<string, string>) => void;
}) {
  const allFields = [
    ...provider.credentials.filter((c) => c.required),
    ...provider.credentials.filter((c) => !c.required),
  ];

  const [fieldIndex, setFieldIndex] = useState(0);
  const [values, setValues] = useState<Record<string, string>>({});

  const currentField: CredentialField | undefined = allFields[fieldIndex];

  const handleSubmit = useCallback(
    (value: string) => {
      const field = allFields[fieldIndex];
      if (!field) return;

      const next = { ...values, [field.key]: value };
      setValues(next);

      if (fieldIndex + 1 < allFields.length) {
        setFieldIndex(fieldIndex + 1);
      } else {
        onComplete(next);
      }
    },
    [fieldIndex, allFields, values, onComplete]
  );

  if (!currentField) return null;

  return (
    <Box flexDirection="column">
      <Text bold>Enter credentials for {provider.display_name}:</Text>
      <Text color={theme.textMuted}>
        ({fieldIndex + 1}/{allFields.length}){" "}
        {!currentField.required && "[optional \u2014 Enter to skip] "}
      </Text>
      <Box marginTop={1}>
        <Text>{currentField.label}: </Text>
        {currentField.secret ? (
          <PasswordInput
            placeholder={currentField.key}
            onSubmit={handleSubmit}
          />
        ) : (
          <TextInput
            placeholder={currentField.key}
            onSubmit={handleSubmit}
          />
        )}
      </Box>
      {currentField.help_url && (
        <Text color={theme.textMuted} dimColor>
          Get key: {currentField.help_url}
        </Text>
      )}
    </Box>
  );
}

/* --- Step 3b: Model / Profile Selection --- */

function ModelStep({
  provider,
  manifest,
  onComplete,
}: {
  provider: ProviderDef;
  manifest: WizardManifest;
  onComplete: (model: string | null, profile: string | null) => void;
}) {
  const mode = provider.model_selection;

  if (mode === "profile") {
    const options = provider.profiles.map((p) => ({
      label: `${p.display_name} \u2014 ${p.description}`,
      value: p.name,
    }));
    const defaultProfile = provider.profiles.find((p) => p.is_default)?.name;

    return (
      <Box flexDirection="column">
        <Text bold>Select model profile:</Text>
        <Box marginTop={1}>
          <Select
            options={options}
            defaultValue={defaultProfile}
            onChange={(value) => onComplete(null, value)}
          />
        </Box>
      </Box>
    );
  }

  if (mode === "explicit") {
    const options = provider.models.map((m) => ({
      label: `${m.display_name}${m.description ? ` \u2014 ${m.description}` : ""}`,
      value: m.name,
    }));
    const defaultModel = provider.models.find((m) => m.is_default)?.name;

    return (
      <Box flexDirection="column">
        <Text bold>Select model:</Text>
        <Box marginTop={1}>
          <Select
            options={options}
            defaultValue={defaultModel}
            onChange={(value) => onComplete(value, null)}
          />
        </Box>
      </Box>
    );
  }

  if (mode === "local") {
    return (
      <LocalModelStep
        manifest={manifest}
        onComplete={(model) => onComplete(model, null)}
      />
    );
  }

  // "managed" — should not reach here (skipped in parent)
  return null;
}

function LocalModelStep({
  manifest,
  onComplete,
}: {
  manifest: WizardManifest;
  onComplete: (model: string) => void;
}) {
  const [useCustom, setUseCustom] = useState(false);
  const ollamaModels = manifest.ollama_status.models;

  if (!manifest.ollama_status.available) {
    return (
      <Box flexDirection="column">
        <Text color={theme.warning}>
          Ollama not detected. Enter a model name manually:
        </Text>
        <Box marginTop={1}>
          <TextInput
            placeholder="e.g. llama3:8b"
            onSubmit={(v) => onComplete(v)}
          />
        </Box>
      </Box>
    );
  }

  if (useCustom) {
    return (
      <Box flexDirection="column">
        <Text bold>Enter custom model name:</Text>
        <Box marginTop={1}>
          <TextInput
            placeholder="e.g. llama3:8b"
            onSubmit={(v) => onComplete(v)}
          />
        </Box>
      </Box>
    );
  }

  const options = [
    ...ollamaModels.map((m) => ({ label: m, value: m })),
    { label: "[Enter custom model name]", value: "__custom__" },
  ];

  return (
    <Box flexDirection="column">
      <Text bold>Select local Ollama model:</Text>
      <Text color={theme.textMuted}>
        {ollamaModels.length} model{ollamaModels.length !== 1 ? "s" : ""}{" "}
        detected
      </Text>
      <Box marginTop={1}>
        <Select
          options={options}
          onChange={(value) => {
            if (value === "__custom__") {
              setUseCustom(true);
            } else {
              onComplete(value);
            }
          }}
        />
      </Box>
    </Box>
  );
}

/* --- Step 4: Optional Keys --- */

const OPTIONAL_KEYS = [
  { key: "NCBI_API_KEY", label: "NCBI API Key (for PubMed queries)", secret: true },
  { key: "OMICS_OS_CLOUD_KEY", label: "Omics-OS Cloud API Key", secret: true },
];

function OptionalKeysStep({
  onComplete,
}: {
  onComplete: (keys: Record<string, string>, smartStd: boolean) => void;
}) {
  const [fieldIndex, setFieldIndex] = useState(0);
  const [values, setValues] = useState<Record<string, string>>({});
  const [phase, setPhase] = useState<"keys" | "smartStd">("keys");

  const handleKeySubmit = useCallback(
    (value: string) => {
      const field = OPTIONAL_KEYS[fieldIndex];
      if (!field) return;

      const next = value ? { ...values, [field.key]: value } : values;
      setValues(next);

      if (fieldIndex + 1 < OPTIONAL_KEYS.length) {
        setFieldIndex(fieldIndex + 1);
      } else {
        setPhase("smartStd");
      }
    },
    [fieldIndex, values]
  );

  if (phase === "smartStd") {
    return (
      <Box flexDirection="column">
        <Text bold>Enable smart standardization?</Text>
        <Text color={theme.textMuted}>
          Automatically standardize column names and data formats (Y/n)
        </Text>
        <Box marginTop={1}>
          <ConfirmInput
            defaultChoice="confirm"
            onConfirm={() => onComplete(values, true)}
            onCancel={() => onComplete(values, false)}
          />
        </Box>
      </Box>
    );
  }

  const currentField = OPTIONAL_KEYS[fieldIndex];
  if (!currentField) return null;

  return (
    <Box flexDirection="column">
      <Text bold>Optional configuration:</Text>
      <Text color={theme.textMuted}>
        ({fieldIndex + 1}/{OPTIONAL_KEYS.length}) Press Enter to skip
      </Text>
      <Box marginTop={1}>
        <Text>{currentField.label}: </Text>
        {currentField.secret ? (
          <PasswordInput
            placeholder="(optional)"
            onSubmit={handleKeySubmit}
          />
        ) : (
          <TextInput
            placeholder="(optional)"
            onSubmit={handleKeySubmit}
          />
        )}
      </Box>
    </Box>
  );
}
