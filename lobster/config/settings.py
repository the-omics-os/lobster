"""
Application settings and configuration.

This module centralizes all configuration settings for the application,
including the new professional agent configuration system.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import dotenv_values, find_dotenv, load_dotenv

from lobster.config.supervisor_config import SupervisorConfig

logger = logging.getLogger(__name__)

# The catalog reads these from the workspace .env itself, so that a value set in
# the shell wins over the workspace (LOBSTER_LAMIN=0 lobster chat records nothing).
# Loading a .env must therefore never leave them in os.environ.
_SHELL_ONLY_KEYS = ("LOBSTER_LAMIN", "LOBSTER_LAMIN_INSTANCE")

# Where Settings records what a .env load set: by a project's workspace, or at process start.
_PROJECT_KEYS = "_project_applied"
_START_KEYS = "_start_applied"


def _is_project_workspace(workspace: Path) -> bool:
    """A workspace holding a ``project.json`` (a regular file, never a link) is a project.

    Same test as lobster.core.runtime.workspace._is_project_workspace; the literal is kept
    here because settings must not import the runtime. Keep the two in step.
    """
    project_file = workspace / "project.json"
    return (
        not workspace.is_symlink()
        and project_file.is_file()
        and not project_file.is_symlink()
    )


class Settings:
    """
    Application settings with environment variable support.

    This class manages application-wide settings with fallbacks
    and environment variable overrides for easier configuration
    in different environments, especially in containers.

    Credential Loading Priority (highest first):
    1. Workspace credentials (workspace/.env, else ./.env; a project reads only
       workspace/.env then the .env beside its workspace folder); these override the
       environment
    2. Environment variables (explicit exports)
    3. A .env found upward from the installed package (only when no workspace .env)
    4. Global credentials (~/.config/lobster/credentials.env)

    The last two only fill variables that are still unset. A workspace ``.env`` loaded by
    ``reload_credentials`` is undone when the next workspace is loaded; for a project
    workspace so is what ./.env and the upward file gave this process at its start.
    """

    def __init__(self):
        """Initialize application settings."""
        # Load credentials in priority order (lowest first, so higher priority overrides)
        self._load_credentials()

        # hackathon
        self.LINKUP_API_KEY = os.environ.get("LINKUP_API_KEY", "")

        # CDK variables (used by lobster-cloud deployment)
        self.STACK_NAME = "LobsterStack"
        self.CDK_DEPLY_ACCOUNT = os.environ.get("CDK_DEPLOY_ACCOUNT", "")
        # AWS Fargate CPU/Memory options summary:
        # - 256 (.25 vCPU): 512 MiB, 1 GB, 2 GB (Linux)
        # - 512 (.5 vCPU): 1 GB, 2 GB, 3 GB, 4 GB (Linux)
        # - 1024 (1 vCPU): 2-8 GB (Linux, Windows)
        # - 2048 (2 vCPU): 4-16 GB (Linux, Windows, 1 GB steps)
        # - 4096 (4 vCPU): 8-30 GB (Linux, Windows, 1 GB steps)
        # - 8192 (8 vCPU, Linux 1.4.0+): 16-60 GB (4 GB steps)
        # - 16384 (16 vCPU, Linux 1.4.0+): 32-120 GB (8 GB steps)
        # See AWS docs for full details.
        self.MEMORY = 24576
        self.CPU = 8192
        # Initialize supervisor configuration
        self.supervisor_config = SupervisorConfig.from_env()

        # Base directories
        self.BASE_DIR = Path(__file__).resolve().parent.parent

        # API keys
        self.AWS_BEDROCK_ACCESS_KEY = os.environ.get("AWS_BEDROCK_ACCESS_KEY", "")
        self.AWS_BEDROCK_SECRET_ACCESS_KEY = os.environ.get(
            "AWS_BEDROCK_SECRET_ACCESS_KEY", ""
        )
        self.ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
        self.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
        self.NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
        self.NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "")
        # Multi-key support for parallelization (NCBI_API_KEY_1, NCBI_API_KEY_2, etc.)
        self._ncbi_api_keys = self._load_ncbi_api_keys()

        # Ollama settings (local LLM support)
        self.OLLAMA_BASE_URL = os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.OLLAMA_DEFAULT_MODEL = os.environ.get(
            "OLLAMA_DEFAULT_MODEL", "gpt-oss:20b"
        )

        # Logging settings
        self.LOG_LEVEL = os.environ.get("LOBSTER_LOG_LEVEL", "WARNING").upper()

        # Git automation settings (for sync scripts and CI/CD)
        self.GIT_USER_NAME = os.environ.get("GIT_USER_NAME", "")
        self.GIT_USER_EMAIL = os.environ.get("GIT_USER_EMAIL", "")

        # AWS region (fallback for backward compatibility)
        self.REGION = os.environ.get("AWS_REGION", "us-east-1")

        # Web server settings: reserved, no current consumer. `lobster serve`
        # takes --host/--port (localhost:8000 by default); these only mirror it.
        self.PORT = int(os.environ.get("PORT", "8000"))
        # Localhost by default; containers set HOST=0.0.0.0 explicitly.
        self.HOST = os.environ.get("HOST", "127.0.0.1")
        self.DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

        # Data processing settings
        self.MAX_FILE_SIZE_MB = int(os.environ.get("LOBSTER_MAX_FILE_SIZE_MB", "500"))
        self.DEFAULT_CLUSTER_RESOLUTION = float(
            os.environ.get("LOBSTER_CLUSTER_RESOLUTION", "0.5")
        )

        # SSL/HTTPS settings
        self.SSL_VERIFY = os.environ.get("LOBSTER_SSL_VERIFY", "true").lower() == "true"
        self.SSL_CERT_PATH = os.environ.get("LOBSTER_SSL_CERT_PATH", None)

        # Configure global SSL context for all urllib requests (including Bio.Entrez)
        self._configure_global_ssl_context()

    def _configure_global_ssl_context(self) -> None:
        """
        Configure global SSL context for urllib (used by Bio.Entrez, etc.).

        This patches ssl._create_default_https_context to use certifi's certificate
        bundle by default, which fixes SSL issues on systems without proper root
        certificates (macOS, corporate proxies, etc.).

        Called once during Settings initialization to ensure all network requests
        use proper SSL handling.
        """
        import ssl

        # If SSL verification is disabled, use unverified context
        if not self.SSL_VERIFY:
            logger.warning(
                "SSL certificate verification DISABLED globally. "
                "Set LOBSTER_SSL_VERIFY=true to re-enable (recommended)."
            )
            ssl._create_default_https_context = ssl._create_unverified_context
            return

        # If custom cert path is specified, use it
        if self.SSL_CERT_PATH:
            if os.path.exists(self.SSL_CERT_PATH):
                logger.info(
                    f"Using custom SSL certificate bundle: {self.SSL_CERT_PATH}"
                )
                original_context = ssl._create_default_https_context
                cert_path = self.SSL_CERT_PATH

                def _create_custom_context():
                    ctx = original_context()
                    ctx.load_verify_locations(cafile=cert_path)
                    return ctx

                ssl._create_default_https_context = _create_custom_context
                return
            else:
                logger.warning(
                    f"Custom SSL cert path not found: {self.SSL_CERT_PATH}. "
                    "Falling back to certifi."
                )

        # Default: use certifi's certificate bundle for reliable SSL
        try:
            import certifi

            def _create_certifi_context():
                return ssl.create_default_context(cafile=certifi.where())

            ssl._create_default_https_context = _create_certifi_context
            logger.debug(
                "Configured global SSL context with certifi certificate bundle"
            )
        except ImportError:
            logger.debug("certifi not available, using system certificates")

    def get_all_settings(self) -> Dict[str, Any]:
        """
        Get all settings as a dictionary.

        Returns:
            dict: All settings
        """
        settings_dict = {}
        for attr in dir(self):
            if not attr.startswith("_") and not callable(getattr(self, attr)):
                settings_dict[attr] = getattr(self, attr)
        return settings_dict

    def get_setting(self, name: str, default: Any = None) -> Any:
        """
        Get a specific setting.

        Args:
            name: Setting name
            default: Default value if setting doesn't exist

        Returns:
            Value of the setting or default
        """
        return getattr(self, name, default)

    def _load_credentials(self, workspace_path: Path = None) -> None:
        """
        Load credentials from the ``.env`` files, in this order (highest first):

        1. The workspace ``.env``, else ``./.env`` (also when no workspace is given).
           For a project (a ``.lobster_workspace`` holding ``project.json``) the
           candidates are only the workspace ``.env`` then the ``.env`` beside the
           workspace folder, where ``lobster init`` writes the key, and step 3 below is
           skipped: ``./.env`` and the upward file belong to the server's folder or
           venv, not to a project, so what they gave this process at its start is
           undone first. All override the environment.
        2. Real environment variables (explicit exports).
        3. A ``.env`` found upward from the installed package, read only when
           step 1 found no workspace ``.env``. It never overrides anything.
        4. Global credentials (~/.config/lobster/credentials.env). Never overrides.

        Args:
            workspace_path: Optional explicit workspace path. If None, uses CWD.

        Example:
            >>> settings._load_credentials()  # Load from CWD
            >>> settings._load_credentials(Path("/path/to/workspace"))  # Explicit workspace

        LOBSTER_LAMIN and LOBSTER_LAMIN_INSTANCE keep their shell values (or stay
        unset): the catalog reads the workspace .env itself, below the shell. Neither
        file loaded here (global credentials.env, workspace .env) can set them; the
        global setting is ``lamin_enabled`` in providers.json.
        """
        shell = {k: os.environ[k] for k in _SHELL_ONLY_KEYS if k in os.environ}
        try:
            self._load_credential_files(workspace_path)
        finally:
            for key in _SHELL_ONLY_KEYS:
                if key in shell:
                    os.environ[key] = shell[key]
                else:
                    os.environ.pop(key, None)

    def _load_credential_files(self, workspace_path: Path = None) -> None:
        from lobster.config.global_config import (
            get_global_credentials_path,
            global_credentials_exist,
        )

        if workspace_path:
            workspace_env = workspace_path / ".env"
        else:
            workspace_env = Path.cwd() / ".env"
        cwd_env = Path.cwd() / ".env"
        # `lobster init` writes a project's key beside its workspace folder, so that file
        # is the project's own, found from any working folder. Same rank as workspace_env.
        # The folder is named like the runtime names it, so a stray project.json in some
        # other folder cannot make its parent (say $HOME) a credentials source.
        project_env = None
        if workspace_path:
            lexical = Path(os.path.abspath(workspace_path))  # "." and ".." resolve
            if lexical.name == ".lobster_workspace" and _is_project_workspace(lexical):
                project_env = lexical.parent / ".env"
        # A project reads ONLY its own files: the upward search starts in the folder of
        # the installed package (a venv inside project A's folder) or, under a tracer, in
        # the cwd, and ./.env is the folder the server was started in; either would hand
        # project A's file to project B, and neither is ever undone. What the process
        # already has from its start (shell, the import-time load) is the server's own.
        is_project = project_env is not None
        # Undo what the last project loaded; for a project also what this process got at
        # its own start from ./.env and the upward file. Any other load re-reads those
        # files by the old rules, so nothing changes for it.
        self._undo_credentials(_PROJECT_KEYS)
        if is_project:
            self._undo_credentials(_START_KEYS)
        candidates = (
            (workspace_env, project_env) if is_project else (workspace_env, cwd_env)
        )

        # override=False keeps what is already set (a real export, or a .env loaded
        # above it), so the .env next to the installation only fills gaps. It is
        # read before the global file because it has always ranked above it.
        if not is_project and not workspace_env.exists():
            upward = find_dotenv()
            if upward:
                before = {k: os.environ.get(k) for k in dotenv_values(upward)}
                load_dotenv(upward, override=False)
                for key, old in before.items():
                    if old is None and key in os.environ:  # only the gaps it filled
                        self._remember(_START_KEYS, key, os.environ[key], None)
                logger.debug(f"Loaded credentials from {upward}")

        if global_credentials_exist():
            global_creds_path = get_global_credentials_path()
            load_dotenv(global_creds_path, override=False)
            logger.debug(f"Loaded global credentials from {global_creds_path}")

        for env_file in candidates:
            if env_file.exists():
                # A reload for a named workspace is the project's; the no-workspace load
                # is the server's own start.
                self._load_workspace_credentials(
                    env_file, _PROJECT_KEYS if workspace_path else _START_KEYS
                )
                logger.debug(f"Loaded workspace credentials from {env_file}")
                break

    def _load_workspace_credentials(self, env_file: Path, store: str) -> None:
        """Load an overriding ``.env``; remember what each key replaced so a later load
        can put it back (one process may serve several projects in turn)."""
        names = [k for k, v in dotenv_values(env_file).items() if v is not None]
        before = {k: os.environ.get(k) for k in names}
        load_dotenv(env_file, override=True)
        for key in names:
            if key in os.environ:
                self._remember(store, key, os.environ[key], before[key])

    def _remember(self, store: str, key: str, applied: str, previous) -> None:
        """Record that we set ``key`` to ``applied`` over ``previous`` (None = absent).

        A key we already hold a record for, still at the value we set, keeps its original
        ``previous``: reading the same file again must not make the shell's value forgotten.
        """
        held = self.__dict__.setdefault(store, {})
        if key in held and held[key][0] == previous:
            previous = held[key][1]
        held[key] = (applied, previous)

    def _undo_credentials(self, store: str) -> None:
        """Put back what our loads replaced (the shell's value, the global file's, or
        nothing), so the next project does not inherit it.

        A variable that no longer holds the value we set was changed by someone since: left
        alone. The global file and the shell are never recorded: every project shares them.
        """
        for key, (applied, previous) in self.__dict__.get(store, {}).items():
            if os.environ.get(key) != applied:
                continue
            if previous is None:
                del os.environ[key]
            else:
                os.environ[key] = previous
        self.__dict__[store] = {}

    def reload_credentials(self, workspace_path: Path) -> None:
        """
        Reload credentials for a specific workspace.

        Call this when switching workspaces to ensure correct credentials are loaded.
        What the previous workspace's ``.env`` set is undone first, so a long-lived process
        serving several projects never lends one project's keys to the next.

        Args:
            workspace_path: Path to the workspace directory

        Example:
            >>> settings.reload_credentials(Path("/new/workspace"))
        """
        self._load_credentials(workspace_path)
        # Refresh API key attributes
        self.AWS_BEDROCK_ACCESS_KEY = os.environ.get("AWS_BEDROCK_ACCESS_KEY", "")
        self.AWS_BEDROCK_SECRET_ACCESS_KEY = os.environ.get(
            "AWS_BEDROCK_SECRET_ACCESS_KEY", ""
        )
        self.ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
        self.NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
        self._ncbi_api_keys = self._load_ncbi_api_keys()
        logger.info(f"Reloaded credentials for workspace: {workspace_path}")

    def _load_ncbi_api_keys(self) -> List[str]:
        """
        Load all NCBI API keys from environment variables.

        Supports multiple keys for parallelization:
        - NCBI_API_KEY (primary, backward compatible)
        - NCBI_API_KEY_1, NCBI_API_KEY_2, ... NCBI_API_KEY_9

        Returns:
            List of API keys found in environment
        """
        keys = []
        # Primary key (backward compatible)
        primary = os.environ.get("NCBI_API_KEY", "")
        if primary:
            keys.append(primary)
        # Additional keys for parallelization
        for i in range(1, 10):
            key = os.environ.get(f"NCBI_API_KEY_{i}", "")
            if key:
                keys.append(key)
        return keys

    def get_ncbi_api_key(self, index: int = 0) -> str:
        """
        Get NCBI API key by index (for parallelization).

        Uses round-robin if index exceeds available keys.

        Args:
            index: Worker index (0-based)

        Returns:
            API key string, or empty string if no keys configured
        """
        if not self._ncbi_api_keys:
            return ""
        return self._ncbi_api_keys[index % len(self._ncbi_api_keys)]

    def get_all_ncbi_api_keys(self) -> List[str]:
        """
        Get all available NCBI API keys for parallel workers.

        Returns:
            List of all configured API keys
        """
        return self._ncbi_api_keys.copy()

    def get_ncbi_key_count(self) -> int:
        """
        Get the number of NCBI API keys available.

        Returns:
            Number of keys (useful for determining parallelization factor)
        """
        return len(self._ncbi_api_keys)

    def get_agent_llm_params(self, agent_name: str) -> Dict[str, Any]:
        """
        Get LLM parameters for a specific agent.

        Resolution order (via agent_defaults):
        1. Environment variable overrides
        2. Workspace config.toml agent_settings
        3. Defaults (temperature unset = model default, thinking=disabled)

        Never fails for unknown agents — they get sensible defaults.

        Args:
            agent_name: Name of the agent (e.g., 'supervisor', 'transcriptomics_expert')

        Returns:
            Dictionary of LLM initialization parameters

        Note:
            Provider/model information is resolved by ConfigResolver, not Settings.
        """
        from lobster.config.agent_defaults import get_agent_params

        workspace_path = getattr(self, "_workspace_path", None)
        return get_agent_params(agent_name, workspace_path=workspace_path)

    def get_assistant_llm_params(self, agent_name: str) -> Dict[str, Any]:
        """
        Get LLM parameters for a specific assistant.

        Args:
            agent_name: Name of the assistant (e.g., 'data_expert_assistant')

        Returns:
            Dictionary of LLM initialization parameters
        """
        return self.get_agent_llm_params(agent_name)

    def get_supervisor_config(self) -> SupervisorConfig:
        """
        Get supervisor configuration.

        Returns:
            SupervisorConfig: Supervisor configuration instance
        """
        return self.supervisor_config


# Model pricing configuration (USD per million tokens)
# Dynamically loaded from ProviderRegistry - single source of truth
# Pricing is defined in each provider's ModelInfo objects:
#   - lobster/config/providers/anthropic_provider.py
#   - lobster/config/providers/bedrock_provider.py
#   - lobster/config/providers/gemini_provider.py
#   - lobster/config/providers/ollama_provider.py (free - pricing=0.0)

# Cache for lazy-loaded pricing (avoids circular imports)
_MODEL_PRICING_CACHE: Dict[str, Any] = None


def get_model_pricing() -> Dict[str, Any]:
    """
    Get model pricing dictionary from all registered providers.

    Uses lazy loading to avoid circular imports. Pricing is defined
    in each provider's ModelInfo objects (single source of truth).

    Returns:
        Dict mapping model names to pricing info:
        {
            "model-name": {
                "input_per_million": float,
                "output_per_million": float,
                "display_name": str
            }
        }

    Example:
        >>> pricing = get_model_pricing()
        >>> gemini_cost = pricing.get("gemini-3-pro-preview")
        >>> if gemini_cost:
        ...     print(f"Input: ${gemini_cost['input_per_million']}/M tokens")
    """
    global _MODEL_PRICING_CACHE

    if _MODEL_PRICING_CACHE is None:
        from lobster.config.providers.registry import ProviderRegistry

        _MODEL_PRICING_CACHE = ProviderRegistry.get_all_models_with_pricing()

    return _MODEL_PRICING_CACHE


def reset_model_pricing_cache() -> None:
    """
    Reset the pricing cache (for testing or after provider changes).

    This forces the next call to get_model_pricing() to reload from providers.
    """
    global _MODEL_PRICING_CACHE
    _MODEL_PRICING_CACHE = None


def register_model_pricing(models) -> int:
    """Merge priced ``ModelInfo`` entries into the live pricing table, in place.

    Pricing is built from static catalogs at import (no network). Providers with a
    live catalog call this after an explicit fetch so those models are priced too.
    ``MODEL_PRICING`` is updated in place, so consumers holding a reference see it.
    Existing entries are kept. Returns the number of entries added.
    """
    table = get_model_pricing()
    added = 0
    for model in models:
        cost = getattr(model, "input_cost_per_million", None)
        if not cost or cost <= 0 or model.name in table:
            continue
        table[model.name] = {
            "input_per_million": cost,
            "output_per_million": getattr(model, "output_cost_per_million", None)
            or 0.0,
            "display_name": model.display_name,
        }
        added += 1
    return added


# Backward compatibility: MODEL_PRICING as module-level dict
# Populated on first access via get_model_pricing()
MODEL_PRICING: Dict[str, Any] = get_model_pricing()


# Default pricing for unknown models (use Sonnet as reasonable default)
DEFAULT_PRICING = {
    "input_per_million": 3.0,
    "output_per_million": 15.0,
}


def calculate_token_cost(
    input_tokens: int,
    output_tokens: int,
    model_id: str = None,
) -> float:
    """
    Calculate cost in USD for token usage.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model_id: Model identifier (optional, uses default if not found)

    Returns:
        Cost in USD
    """
    pricing = (
        MODEL_PRICING.get(model_id, DEFAULT_PRICING) if model_id else DEFAULT_PRICING
    )

    input_cost = (input_tokens / 1_000_000) * pricing.get(
        "input_per_million", DEFAULT_PRICING["input_per_million"]
    )
    output_cost = (output_tokens / 1_000_000) * pricing.get(
        "output_per_million", DEFAULT_PRICING["output_per_million"]
    )

    return input_cost + output_cost


# Create singleton instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get the application settings.

    Returns:
        Settings: Application settings
    """
    return settings
