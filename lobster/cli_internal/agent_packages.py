"""The official ``lobster-*`` expert packages that ``lobster init`` and ``lobster agents install`` offer.

Standard library only, so a caller that only lists them (an MCP overview) does not
pay for the init wizard. Descriptions are plain English; the places that show them
translate them with ``h()``.
"""

# What every init UI ticks by default; the Go wizard marks the same two.
DEFAULT_AGENT_PACKAGES = ("lobster-research", "lobster-transcriptomics")

# (package_name, description, agents_provided, published_on_pypi, experimental)
AVAILABLE_AGENT_PACKAGES = [
    (
        "lobster-research",
        "Literature search & data discovery",
        ["research_agent", "data_expert_agent"],
        True,
        False,
    ),
    (
        "lobster-transcriptomics",
        "Single-cell & bulk RNA-seq analysis",
        ["transcriptomics_expert", "annotation_expert", "de_analysis_expert"],
        True,
        False,
    ),
    (
        "lobster-visualization",
        "Data visualization & plotting",
        ["visualization_expert_agent"],
        True,
        False,
    ),
    (
        "lobster-genomics",
        "Genomics/DNA analysis (VCF, GWAS, clinical variants)",
        ["genomics_expert", "variant_analysis_expert"],
        True,
        False,
    ),
    (
        "lobster-proteomics",
        "Mass spec & affinity proteomics analysis",
        [
            "proteomics_expert",
            "proteomics_de_analysis_expert",
            "biomarker_discovery_expert",
        ],
        True,
        False,
    ),
    (
        "lobster-metabolomics",
        "LC-MS, GC-MS, and NMR metabolomics analysis",
        ["metabolomics_expert"],
        True,
        False,
    ),
    (
        "lobster-ml",
        "Machine learning, feature selection, and survival analysis",
        [
            "machine_learning_expert",
            "feature_selection_expert",
            "survival_analysis_expert",
        ],
        True,
        True,
    ),
    (
        "lobster-drug-discovery",
        "Drug discovery, cheminformatics, and translational strategy",
        [
            "drug_discovery_expert",
            "cheminformatics_expert",
            "clinical_dev_expert",
            "pharmacogenomics_expert",
        ],
        True,
        True,
    ),
    (
        "lobster-metadata",
        "Metadata filtering & standardization",
        ["metadata_assistant"],
        True,
        True,
    ),
    (
        "lobster-structural-viz",
        "Protein structure visualization (PyMOL, PDB)",
        ["protein_structure_visualization_expert"],
        True,
        True,
    ),
]
