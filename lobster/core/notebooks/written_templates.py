"""The code templates Lobster writes today for tools whose earlier templates ran values.

Each of these tools once put a value into its cell as code: rendered raw
(``results_df.to_{{ output_format }}``) or put into the template with an
f-string before Jinja read it (file paths, ID types, sample names). A stored
step keeps the template it was recorded with, so a session recorded before the
fix would export the value as code again. The exporter leaves out a step of a
listed operation whose template is not one of the ones below.

Each tool's tests pin its builder's template to its hash here: change a
template, change its hash in the same commit.
"""

import hashlib
from typing import Any, Dict, FrozenSet, Optional, Tuple

# operation -> (tool, sha256 of each code_template its builder writes).
# A DE export also keeps the exact template 1.3.0 wrote, which keeps every value
# text, so a session recorded with 1.3.0 still shows its export step.
WRITTEN_TEMPLATES: Dict[str, Tuple[str, FrozenSet[str]]] = {
    "de_results.export": (
        "export_de_results",
        frozenset(
            {
                "3436b122e11534962de34275282f97f617662a84ff54463732e104c170363bdc",
                "aff7d99fdfe066a84667009ea59ee8744bc50425e85c6262e4a903b2a1425bfd",  # 1.3.0
            }
        ),
    ),
    "de_results.extract_and_export": (
        "extract_and_export_de_results",
        frozenset(
            {
                "f7cb7a579bf58852a0d42c5505fdd33be60bde8b931785810443bb3dd6c0e1f8",
                "35440cc8c975149226867dad1bf76d30c615af8f555c781dc9510c745b10729a",  # 1.3.0
            }
        ),
    ),
    "proteomics.affinity.cross_platform_concordance": (
        "assess_cross_platform_concordance",
        frozenset({"fc3f351f8594fa637a1de12a19eb6a15b717081869707a5c153df6fb05e4cbbd"}),
    ),
    "import_bulk_counts": (
        "import_bulk_counts",
        frozenset({"086d0cdb37563ff892bd5233260e028f6ffb499387a7de216c5f0104b12716f8"}),
    ),
    "merge_sample_metadata": (
        "merge_sample_metadata",
        frozenset(
            {
                "b1184ff19f003f5e310f1e465092eb3527ddc992e71de018a6f223a574d023ec",
                "abc2379075ee84497c8b002374b0ade813b2cf340918d124cbaa12da1d5ca42f",  # 1.3.0
            }
        ),
    ),
    "convert_gene_identifiers": (
        "convert_gene_identifiers",
        frozenset({"092845efd28d3e9ecd43e1e70e37f58c30e52dab375efaf6ac9939b84306a614"}),
    ),
    "anndata.concat": (
        "concatenate_samples",
        frozenset({"81e90e9747bff3aa7d1aea00c98fd568dac85eb44913761be84fcc822d6d67e5"}),
    ),
}


def template_hash(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def outdated_tool(ir: Any) -> Optional[str]:
    """The tool of a listed step whose template this Lobster does not write
    (a template that is not text included), else None."""
    operation = getattr(ir, "operation", None)
    listed = WRITTEN_TEMPLATES.get(operation) if isinstance(operation, str) else None
    if listed is None:
        return None
    tool, hashes = listed
    template = getattr(ir, "code_template", None)
    if isinstance(template, str) and template_hash(template) in hashes:
        return None
    return tool
