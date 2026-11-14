# Local Python Setup with an Omics-OS Cloud Account

The non-interactive Cloud init flags below require **Lobster AI 1.2.2 or later**.
Use `[full]` for the default analysis install, not a provider-only/core-only
install that can leave no specialist agents. `[full]` is a bundle, not every
optional domain; check the agents needed for the task. If `uv` is absent, obtain
permission to install it first; do not switch to npm or bootstrap with bare pip.

For a **fresh approved project/environment** (POSIX example):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python 'lobster-ai[full]~=1.2.2'
.venv/bin/python -m lobster --version
```

Do not replace an existing `.venv` without permission. In an already approved
Python environment, reuse its interpreter instead. These explicit Python commands
avoid a Python/npm binary collision. A `uv tool` install is another option in the
[CLI reference](cli-reference.md#installation-and-local-configuration).

Get an authorized key from `https://app.omics-os.com/account` through the
human account owner. For machine setup, prefer a securely injected
`OMICS_OS_API_KEY`. If the user instead supplies a key file, set `OMICS_OS_KEY_FILE`
to **that authorized path** and use `export OMICS_OS_API_KEY="$(<"$OMICS_OS_KEY_FILE")"`
in Bash; do not print it, source the file as code, or enable shell tracing.

```bash
(
  set -e
  umask 077
  : "${OMICS_OS_API_KEY:?Supply an authorized Omics-OS Cloud API key securely}"
  .venv/bin/python -m lobster init --non-interactive --use-omics-os \
    --skip-extras --skip-ssl-test </dev/null
  .venv/bin/python -m lobster status
  .venv/bin/python -m lobster agents list
)
```

This selects the **local** gateway provider without requiring an Anthropic,
OpenAI or personal AWS key. No npm install/login is needed. `--use-omics-os`
looks up the nonblank Python environment key, then the active saved Python
platform profile. If neither supplies a usable credential, workspace-mode init
tries CWD `.env`, then global Python `credentials.env`; `--global` skips the CWD
file. See the [lookup details](cli-reference.md#local-python-with-an-omics-os-cloud-account).
`--omics-os-key <key>` takes precedence, including explicit blank input failing
without fallback, but can expose the key in process arguments. Do not combine Cloud
setup with another provider's flags: the other explicit provider takes priority.
`--cloud-key` is a different legacy supplement, not gateway selection.

Read the exit status **and output**. Missing credentials fail nonzero without a
hidden prompt. Existing different-provider configuration requires approved
`--force`; "already configured" means no rewrite, not a new key rotation or an
authentication check. `--force` can replace credentials and creates a configuration
backup; see the [write/overwrite details](cli-reference.md#cloud-init-writes-and-overwrite).
Do not call internal credential/config APIs to bypass an unsupported CLI; check
`init --help` and upgrade the Python install with permission instead.

**Saved, not checked:** init persists configuration but does not verify account
access, key validity or inference. In Python `status`, inspect `Initialization`,
`Provider` and `Configuration Source`: expect provider `omics-os`, but
`Account Access: Not checked (configuration only)` is **not authentication proof**.
`--skip-extras --skip-ssl-test` skips init's optional installs/SSL test, not all
possible network activity (a saved OAuth token may refresh). Inspect the agent
inventory; "No agents installed" is not an analysis-ready setup. Stop on an
unexpected provider, missing task agents or setup errors.

For **environment-only local setup**, omit init and supply both
`LOBSTER_LLM_PROVIDER=omics-os` and `OMICS_OS_API_KEY` securely. This does not
persist provider credentials. Workspace/global provider settings outrank the
provider environment variable; use a clean approved configuration or an explicit
`--provider omics-os` on a later query/chat, not a silent config overwrite.

A human can use Python `lobster init` and paste a key in the masked Cloud prompt.
The npm device flow is a human-approved alternative for managed Cloud sessions,
not non-interactive Python setup; configure Python separately. See
[CLI setup details](cli-reference.md#local-python-with-an-omics-os-cloud-account).

**Optional live verification requires separate permission:** `config-test --json`
invokes a paid LLM test, may contact NCBI and writes a workspace probe. If approved,
use the Python command, require exit 0, JSON `valid: true` and
`checks.llm_provider.provider: "omics-os"`, and retain failure diagnostics
privately. Do not run it merely to claim setup success.
