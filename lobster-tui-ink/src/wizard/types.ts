/** TypeScript mirror of lobster/ui/wizard/manifest.py dataclasses. */

export interface CredentialField {
  key: string;
  label: string;
  secret: boolean;
  required: boolean;
  env_var: string | null;
  help_url: string | null;
}

export interface ModelDef {
  name: string;
  display_name: string;
  description: string;
  is_default: boolean;
  context_window: number;
  input_cost_per_million: number;
  output_cost_per_million: number;
}

export interface ProfileDef {
  name: string;
  display_name: string;
  description: string;
  is_default: boolean;
}

export interface AgentPackageDef {
  package_name: string;
  description: string;
  agents: string[];
  published: boolean;
  experimental: boolean;
}

export interface OllamaStatus {
  available: boolean;
  models: string[];
  error: string | null;
}

export type ModelSelection = "explicit" | "profile" | "local" | "managed";

export interface ProviderDef {
  name: string;
  display_name: string;
  description: string;
  model_selection: ModelSelection;
  credentials: CredentialField[];
  models: ModelDef[];
  profiles: ProfileDef[];
}

export interface WizardManifest {
  providers: ProviderDef[];
  agent_packages: AgentPackageDef[];
  ollama_status: OllamaStatus;
}

export interface WizardResult {
  selectedPackages: string[];
  provider: string;
  credentials: Record<string, string>;
  model: string | null;
  profile: string | null;
  optionalKeys: Record<string, string>;
  smartStandardization: boolean;
}
