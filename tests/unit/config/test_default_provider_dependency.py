"""The default provider (Omics-OS Cloud) must start on a plain `lobster-ai` install.

It builds its chat model through bedrock_builder, which imports langchain_aws
and boto3. When those lived only in the `bedrock` extra, `lobster chat` on a
plain install failed with "No module named 'langchain_aws'".
"""

import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"


def _core_dependencies():
    data = tomllib.loads(PYPROJECT.read_text())
    return {
        re.split(r"[<>=!~\[ ;]", dep, maxsplit=1)[0].lower()
        for dep in data["project"]["dependencies"]
    }


def test_plain_install_carries_what_the_default_provider_imports():
    deps = _core_dependencies()
    assert "langchain-aws" in deps
    assert "boto3" in deps
