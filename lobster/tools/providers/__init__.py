"""
Publication providers for literature search and dataset discovery.

This package contains different provider implementations for accessing
various publication and dataset sources.
"""

from lobster.tools.providers.base_provider import (
    BasePublicationProvider,
    DatasetMetadata,
    DatasetType,
    PublicationMetadata,
    PublicationSource,
)

# Biomolecule database providers (peptides)
from lobster.tools.providers.biomolecule_provider import (
    BaseBiomoleculeProvider,
    BiomoleculeSource,
    PeptideMetadata,
)
from lobster.tools.providers.biorxiv_medrxiv_config import BioRxivMedRxivConfig
from lobster.tools.providers.biorxiv_medrxiv_provider import BioRxivMedRxivProvider
from lobster.tools.providers.dbaasp_provider import DBAASPProvider, DBAASPProviderConfig
from lobster.tools.providers.geo_provider import GEOProvider, GEOProviderConfig
from lobster.tools.providers.iedb_provider import IEDBProvider, IEDBProviderConfig
from lobster.tools.providers.massive_provider import (
    MassIVEProvider,
    MassIVEProviderConfig,
)
from lobster.tools.providers.peptipedia_provider import (
    PeptipediaProvider,
    PeptipediaProviderConfig,
)
from lobster.tools.providers.pubmed_provider import PubMedProvider, PubMedProviderConfig

# PRIDE provider is PREMIUM-only (proteomics)
# Import conditionally to avoid breaking FREE tier
try:
    from lobster.tools.providers.pride_provider import (
        PRIDEProvider,
        PRIDEProviderConfig,
    )

    _PRIDE_AVAILABLE = True
except ImportError:
    _PRIDE_AVAILABLE = False
    PRIDEProvider = None  # type: ignore
    PRIDEProviderConfig = None  # type: ignore

__all__ = [
    # Base classes
    "BasePublicationProvider",
    "PublicationSource",
    "DatasetType",
    "PublicationMetadata",
    "DatasetMetadata",
    # Biomolecule base classes
    "BaseBiomoleculeProvider",
    "BiomoleculeSource",
    "PeptideMetadata",
    # BioRxiv/MedRxiv provider
    "BioRxivMedRxivProvider",
    "BioRxivMedRxivConfig",
    # GEO provider
    "GEOProvider",
    "GEOProviderConfig",
    # PubMed provider
    "PubMedProvider",
    "PubMedProviderConfig",
    # MassIVE provider
    "MassIVEProvider",
    "MassIVEProviderConfig",
    # Peptide database providers
    "DBAASPProvider",
    "DBAASPProviderConfig",
    "IEDBProvider",
    "IEDBProviderConfig",
    "PeptipediaProvider",
    "PeptipediaProviderConfig",
]

# Add PRIDE to exports if available (PREMIUM tier)
if _PRIDE_AVAILABLE:
    __all__.extend(["PRIDEProvider", "PRIDEProviderConfig"])
