import React, { useState, useCallback } from "react";
import { Box, Text } from "ink";
import { Select, MultiSelect, TextInput, PasswordInput } from "@inkjs/ui";
import type {
  WizardManifest,
  WizardResult,
  ProviderDef,
  CredentialField,
} from "./types.js";

type WizardStep = "packages" | "provider" | "credentials";

const STEP_LABELS: Record<WizardStep, string> = {
  packages: "Agent Packages",
  provider: "LLM Provider",
  credentials: "Credentials",
};

const STEPS: WizardStep[] = ["packages", "provider", "credentials"];

interface InitWizardProps {
  manifest: WizardManifest;
  onComplete: (result: WizardResult) => void;
}

export function InitWizard({ manifest, onComplete }: InitWizardProps) {
  const [step, setStep] = useState<WizardStep>("packages");
  const [selectedPackages, setSelectedPackages] = useState<string[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<ProviderDef | null>(
    null
  );

  const stepIndex = STEPS.indexOf(step);

  const handlePackagesSubmit = useCallback(
    (values: string[]) => {
      setSelectedPackages(values);
      setStep("provider");
    },
    []
  );

  const handleProviderSelect = useCallback(
    (value: string) => {
      const provider = manifest.providers.find((p) => p.name === value);
      if (provider) {
        setSelectedProvider(provider);
        if (provider.credentials.length === 0) {
          // No credentials needed (e.g. Ollama)
          onComplete({
            selectedPackages,
            provider: provider.name,
            credentials: {},
          });
        } else {
          setStep("credentials");
        }
      }
    },
    [manifest.providers, selectedPackages, onComplete]
  );

  const handleCredentialsComplete = useCallback(
    (creds: Record<string, string>) => {
      if (!selectedProvider) return;
      onComplete({
        selectedPackages,
        provider: selectedProvider.name,
        credentials: creds,
      });
    },
    [selectedPackages, selectedProvider, onComplete]
  );

  return (
    <Box flexDirection="column" paddingX={1}>
      <Box marginBottom={1}>
        <Text bold color="red">
          Lobster AI
        </Text>
        <Text> — Setup Wizard</Text>
      </Box>

      <StepIndicator current={stepIndex} />

      {step === "packages" && (
        <PackageStep
          manifest={manifest}
          onSubmit={handlePackagesSubmit}
        />
      )}

      {step === "provider" && (
        <ProviderStep
          manifest={manifest}
          onSelect={handleProviderSelect}
        />
      )}

      {step === "credentials" && selectedProvider && (
        <CredentialStep
          provider={selectedProvider}
          onComplete={handleCredentialsComplete}
        />
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
          color={i === current ? "cyan" : i < current ? "green" : "gray"}
          bold={i === current}
        >
          {i < current ? "✓" : i === current ? "●" : "○"} {STEP_LABELS[s]}
        </Text>
      ))}
    </Box>
  );
}

function PackageStep({
  manifest,
  onSubmit,
}: {
  manifest: WizardManifest;
  onSubmit: (values: string[]) => void;
}) {
  const options = manifest.agent_packages.map((pkg) => ({
    label: `${pkg.package_name}${pkg.experimental ? " [experimental]" : ""} — ${pkg.description}`,
    value: pkg.package_name,
  }));

  // Pre-select published, non-experimental packages
  const defaults = manifest.agent_packages
    .filter((p) => p.published && !p.experimental)
    .map((p) => p.package_name);

  return (
    <Box flexDirection="column">
      <Text bold>Select agent packages to install:</Text>
      <Text color="gray">Space to toggle, Enter to confirm</Text>
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

function ProviderStep({
  manifest,
  onSelect,
}: {
  manifest: WizardManifest;
  onSelect: (value: string) => void;
}) {
  const options = manifest.providers.map((p) => ({
    label: `${p.display_name} — ${p.description}`,
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

interface CredentialStepProps {
  provider: ProviderDef;
  onComplete: (creds: Record<string, string>) => void;
}

function CredentialStep({ provider, onComplete }: CredentialStepProps) {
  const fields = provider.credentials.filter((c) => c.required);
  const optionalFields = provider.credentials.filter((c) => !c.required);
  const allFields = [...fields, ...optionalFields];

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
      <Text bold>
        Enter credentials for {provider.display_name}:
      </Text>
      <Text color="gray">
        ({fieldIndex + 1}/{allFields.length}){" "}
        {!currentField.required && "[optional — Enter to skip] "}
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
        <Text color="gray" dimColor>
          Get key: {currentField.help_url}
        </Text>
      )}
    </Box>
  );
}
