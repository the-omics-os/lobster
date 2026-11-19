"""The cloud product is Omics-OS Cloud, in every string the engine ships.

`lobster cloud ...` stays: it is a command, not the product name.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OLD_NAME = re.compile(r"Lobster Cloud|LOBSTER CLOUD")


def test_no_shipped_string_says_lobster_cloud():
    files = [
        *(ROOT / "lobster").rglob("*.py"),
        *(ROOT / "lobster" / "i18n").glob("*.json"),
        *(ROOT / "lobster-tui" / "internal" / "i18n" / "catalogs").glob("*.json"),
    ]
    hits = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in files
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if OLD_NAME.search(line)
    ]
    assert hits == []
