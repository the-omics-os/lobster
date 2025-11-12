"""
Research Agent for literature discovery and dataset identification.

This agent specializes in searching scientific literature, discovering datasets,
and providing comprehensive research context using the modular publication service
architecture with DataManagerV2 integration.
"""

import json
import re
from datetime import date
from typing import List

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.research_agent_assistant import ResearchAgentAssistant
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.metadata_validation_service import MetadataValidationService

# Phase 1: New providers for two-tier access
from lobster.tools.providers.abstract_provider import AbstractProvider
from lobster.tools.providers.base_provider import DatasetType, PublicationSource
from lobster.tools.providers.webpage_provider import WebpageProvider
from lobster.tools.content_access_service import ContentAccessService
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def research_agent(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "research_agent",
    handoff_tools: List = None,
):
    """Create research agent using DataManagerV2 and modular publication service."""

    settings = get_settings()
    model_params = settings.get_agent_llm_params("research_agent")
    llm = create_llm("research_agent", model_params)

    if callback_handler and hasattr(llm, "with_config"):
        llm = llm.with_config(callbacks=[callback_handler])

    # Initialize content access service (Phase 2 complete)
    content_access_service = ContentAccessService(data_manager=data_manager)

    # Initialize research agent assistant for PDF resolution
    research_assistant = ResearchAgentAssistant()

    # Initialize metadata validation service (Phase 2: extracted from ResearchAgentAssistant)
    metadata_validator = MetadataValidationService(data_manager=data_manager)

    # Define tools
    @tool
    def search_literature(
        query: str, max_results: int = 5, sources: str = "pubmed", filters: str = None
    ) -> str:
        """
        Search for scientific literature across multiple sources.

        Args:
            query: Search query string
            max_results: Number of results to retrieve (default: 5, range: 1-20)
            sources: Publication sources to search (default: "pubmed", options: "pubmed,biorxiv,medrxiv")
            filters: Optional search filters as JSON string (e.g., '{"date_range": {"start": "2020", "end": "2024"}}')
        """
        try:
            # Parse sources
            source_list = []
            if sources:
                for source in sources.split(","):
                    source = source.strip().lower()
                    if source == "pubmed":
                        source_list.append(PublicationSource.PUBMED)
                    elif source == "biorxiv":
                        source_list.append(PublicationSource.BIORXIV)
                    elif source == "medrxiv":
                        source_list.append(PublicationSource.MEDRXIV)

            # Parse filters if provided
            filter_dict = None
            if filters:
                import json

                try:
                    filter_dict = json.loads(filters)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid filters JSON: {filters}")

            results = content_access_service.search_literature(
                query=query,
                max_results=max_results,
                sources=source_list if source_list else None,
                filters=filter_dict,
            )

            logger.info(
                f"Literature search completed for: {query[:50]}... (max_results={max_results})"
            )
            return results

        except Exception as e:
            logger.error(f"Error searching literature: {e}")
            return f"Error searching literature: {str(e)}"

    @tool
    def discover_related_studies(
        identifier: str, research_topic: str = None, max_results: int = 5
    ) -> str:
        """
        Discover studies related to a given publication or research topic.

        Args:
            identifier: Publication identifier (DOI or PMID) to find related studies
            research_topic: Optional research topic to focus the search
            max_results: Number of results to retrieve (default: 5)
        """
        try:
            # First get metadata from the source publication
            metadata = content_access_service.extract_metadata(identifier)

            if isinstance(metadata, str):
                return f"Could not extract metadata for {identifier}: {metadata}"

            # Build search query based on metadata and research topic
            search_terms = []

            # Extract key terms from title
            if metadata.title:
                # Simple keyword extraction (could be enhanced with NLP)
                title_words = re.findall(r"\b[a-zA-Z]{4,}\b", metadata.title.lower())
                # Filter common words and take meaningful terms
                meaningful_terms = [
                    w
                    for w in title_words
                    if w not in ["study", "analysis", "using", "with", "from", "data"]
                ]
                search_terms.extend(meaningful_terms[:3])

            # Add research topic if provided
            if research_topic:
                search_terms.append(research_topic)

            # Build search query
            search_query = " ".join(search_terms[:5])  # Limit to avoid too broad search

            if not search_query.strip():
                search_query = "related studies"

            results = content_access_service.search_literature(
                query=search_query, max_results=max_results
            )

            # Add context header
            context_header = f"## Related Studies for {identifier}\n"
            context_header += f"**Source publication**: {metadata.title[:100]}...\n"
            context_header += f"**Search strategy**: {search_query}\n"
            if research_topic:
                context_header += f"**Research focus**: {research_topic}\n"
            context_header += "\n"

            logger.info(f"Related studies search completed for {identifier}")
            return context_header + results

        except Exception as e:
            logger.error(f"Error discovering related studies: {e}")
            return f"Error discovering related studies: {str(e)}"

    @tool
    def find_datasets_from_publication(
        identifier: str, dataset_types: str = None, include_related: bool = True
    ) -> str:
        """
        Find datasets associated with a scientific publication.

        Args:
            identifier: Publication identifier (DOI or PMID)
            dataset_types: Types of datasets to search for, comma-separated (e.g., "geo,sra,arrayexpress")
            include_related: Whether to include related/linked datasets (default: True)
        """
        try:
            # Parse dataset types
            type_list = []
            if dataset_types:
                type_mapping = {
                    "geo": DatasetType.GEO,
                    "sra": DatasetType.SRA,
                    "arrayexpress": DatasetType.ARRAYEXPRESS,
                    "ena": DatasetType.ENA,
                    "bioproject": DatasetType.BIOPROJECT,
                    "biosample": DatasetType.BIOSAMPLE,
                    "dbgap": DatasetType.DBGAP,
                }

                for dtype in dataset_types.split(","):
                    dtype = dtype.strip().lower()
                    if dtype in type_mapping:
                        type_list.append(type_mapping[dtype])

            results = content_access_service.find_linked_datasets(
                identifier=identifier,
                dataset_types=type_list if type_list else None,
                include_related=include_related,
            )

            logger.info(f"Dataset discovery completed for: {identifier}")
            return results

        except Exception as e:
            logger.error(f"Error finding datasets: {e}")
            return f"Error finding datasets from publication: {str(e)}"

    @tool
    def find_marker_genes(query: str, max_results: int = 5, filters: str = None) -> str:
        """
        Find marker genes for cell types from literature.

        Args:
            query: Query with cell_type parameter (e.g., 'cell_type=T_cell disease=cancer')
            max_results: Number of results to retrieve (default: 5, range: 1-15)
            filters: Optional search filters as JSON string
        """
        try:
            # Parse parameters from query
            cell_type_match = re.search(r"cell[_\s]type[=\s]+([^,\s]+)", query)
            disease_match = re.search(r"disease[=\s]+([^,\s]+)", query)

            if not cell_type_match:
                return "Please specify cell_type parameter (e.g., 'cell_type=T_cell')"

            cell_type = cell_type_match.group(1).strip()
            disease = disease_match.group(1).strip() if disease_match else None

            # Build search query for marker genes
            search_query = f'"{cell_type}" marker genes'
            if disease:
                search_query += f" {disease}"

            # Parse filters if provided
            filter_dict = None
            if filters:
                import json

                try:
                    filter_dict = json.loads(filters)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid filters JSON: {filters}")

            results = content_access_service.search_literature(
                query=search_query, max_results=max_results, filters=filter_dict
            )

            # Add context header
            context_header = f"## Marker Gene Search Results for {cell_type}\n"
            if disease:
                context_header += f"**Disease context**: {disease}\n"
            context_header += f"**Search query**: {search_query}\n\n"

            logger.info(
                f"Marker gene search completed for {cell_type} (max_results={max_results})"
            )
            return context_header + results

        except Exception as e:
            logger.error(f"Error finding marker genes: {e}")
            return f"Error finding marker genes: {str(e)}"

    @tool
    def search_datasets_directly(
        query: str, data_type: str = "geo", max_results: int = 5, filters: str = None
    ) -> str:
        """
        Search for datasets directly across omics databases.

        Args:
            query: Search query for datasets
            data_type: Type of omics data (default: "geo", options: "geo,sra,bioproject,biosample,dbgap")
            max_results: Maximum results to return (default: 5)
            filters: Optional filters as JSON string (e.g., '{{"organism": "human", "year": "2023"}}')
        """
        try:
            # Map string to DatasetType
            type_mapping = {
                "geo": DatasetType.GEO,
                "sra": DatasetType.SRA,
                "bioproject": DatasetType.BIOPROJECT,
                "biosample": DatasetType.BIOSAMPLE,
                "dbgap": DatasetType.DBGAP,
                "arrayexpress": DatasetType.ARRAYEXPRESS,
                "ena": DatasetType.ENA,
            }

            dataset_type = type_mapping.get(data_type.lower(), DatasetType.GEO)

            # Parse filters if provided
            filter_dict = None
            if filters:
                import json

                try:
                    filter_dict = json.loads(filters)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid filters JSON: {filters}")

            results = content_access_service.discover_datasets(
                query=query,
                dataset_type=dataset_type,
                max_results=max_results,
                filters=filter_dict,
            )

            logger.info(
                f"Direct dataset search completed: {query[:50]}... ({data_type})"
            )
            return results

        except Exception as e:
            logger.error(f"Error searching datasets directly: {e}")
            return f"Error searching datasets directly: {str(e)}"

    @tool
    def extract_publication_metadata(identifier: str, source: str = "auto") -> str:
        """
        Extract comprehensive metadata from a publication.

        Args:
            identifier: Publication identifier (DOI or PMID)
            source: Publication source (default: "auto", options: "auto,pubmed,biorxiv,medrxiv")
        """
        try:
            # Map source string to PublicationSource
            source_obj = None
            if source != "auto":
                source_mapping = {
                    "pubmed": PublicationSource.PUBMED,
                    "biorxiv": PublicationSource.BIORXIV,
                    "medrxiv": PublicationSource.MEDRXIV,
                }
                source_obj = source_mapping.get(source.lower())

            metadata = content_access_service.extract_metadata(
                identifier=identifier, source=source_obj
            )

            if isinstance(metadata, str):
                return metadata  # Error message

            # Format metadata for display
            formatted = f"## Publication Metadata for {identifier}\n\n"
            formatted += f"**Title**: {metadata.title}\n"
            formatted += f"**UID**: {metadata.uid}\n"
            if metadata.journal:
                formatted += f"**Journal**: {metadata.journal}\n"
            if metadata.published:
                formatted += f"**Published**: {metadata.published}\n"
            if metadata.doi:
                formatted += f"**DOI**: {metadata.doi}\n"
            if metadata.pmid:
                formatted += f"**PMID**: {metadata.pmid}\n"
            if metadata.authors:
                formatted += f"**Authors**: {', '.join(metadata.authors[:5])}{'...' if len(metadata.authors) > 5 else ''}\n"
            if metadata.keywords:
                formatted += f"**Keywords**: {', '.join(metadata.keywords)}\n"

            if metadata.abstract:
                formatted += f"\n**Abstract**:\n{metadata.abstract[:1000]}{'...' if len(metadata.abstract) > 1000 else ''}\n"

            logger.info(f"Metadata extraction completed for: {identifier}")
            return formatted

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return f"Error extracting publication metadata: {str(e)}"

    @tool
    def get_research_capabilities() -> str:
        """
        Get information about available research capabilities and providers.

        Returns:
            str: Formatted capabilities report
        """
        try:
            return content_access_service.query_capabilities()
        except Exception as e:
            logger.error(f"Error getting capabilities: {e}")
            return f"Error getting research capabilities: {str(e)}"

    @tool
    def validate_dataset_metadata(
        accession: str,
        required_fields: str,
        required_values: str = None,
        threshold: float = 0.8,
    ) -> str:
        """
        Quickly validate if a dataset contains required metadata without downloading.

        Args:
            accession: Dataset ID (GSE, E-MTAB, etc.)
            required_fields: Comma-separated required fields (e.g., "smoking_status,treatment_response")
            required_values: Optional JSON of required values (e.g., '{{"smoking_status": ["smoker", "non-smoker"]}}')
            threshold: Minimum fraction of samples with required fields (default: 0.8)

        Returns:
            Validation report with recommendation (proceed/skip/manual_check)
        """
        try:
            # Parse required fields
            fields_list = [f.strip() for f in required_fields.split(",")]

            # Parse required values if provided
            values_dict = None
            if required_values:
                try:
                    values_dict = json.loads(required_values)
                except json.JSONDecodeError:
                    return f"Error: Invalid JSON for required_values: {required_values}"

            # Use GEOService to fetch metadata only
            from lobster.tools.geo_service import GEOService

            console = getattr(data_manager, "console", None)
            geo_service = GEOService(data_manager, console=console)

            # ------------------------------------------------
            # Check if metadata already in store
            # ------------------------------------------------
            if accession in data_manager.metadata_store:
                logger.debug(
                    f"Metadata already stored for: {accession}. returning summary"
                )
                metadata = data_manager.metadata_store[accession]["metadata"]
                return metadata

            # ------------------------------------------------
            # If not fetch and return metadata & val res
            # ------------------------------------------------
            # Fetch metadata only (no expression data download)
            try:
                if accession.startswith("G"):
                    metadata, validation_result = geo_service.fetch_metadata_only(
                        accession
                    )

                    # Use metadata validation service to validate metadata
                    validation_result = metadata_validator.validate_dataset_metadata(
                        metadata=metadata,
                        geo_id=accession,
                        required_fields=fields_list,
                        required_values=values_dict,
                        threshold=threshold,
                    )

                    if validation_result:
                        # Format the validation report
                        report = metadata_validator.format_validation_report(
                            validation_result, accession
                        )

                        logger.info(
                            f"Metadata validation completed for {accession}: {validation_result.recommendation}"
                        )
                        return report
                    else:
                        return f"Error: Failed to validate metadata for {accession}"
                else:
                    logger.info(
                        f"Currently only GEO metadata can be retrieved. {accession} doesnt seem to be a GEO identifier"
                    )
                    return f"Currently only GEO metadata can be retrieved. {accession} doesnt seem to be a GEO identifier"

            except Exception as e:
                logger.error(f"Error accessing dataset {accession}: {e}")
                return f"Error accessing dataset {accession}: {str(e)}"

        except Exception as e:
            logger.error(f"Error in metadata validation: {e}")
            return f"Error validating dataset metadata: {str(e)}"

    @tool
    def extract_paper_methods(url_or_pmid: str) -> str:
        """
        Extract computational analysis methods from a research paper.

        This tool uses the new UnifiedContentService (Phase 3) to extract:
        - Software/tools used (e.g., Scanpy, Seurat, DESeq2)
        - Parameter values and cutoffs
        - Statistical methods
        - Data sources and sample sizes
        - Normalization and QC methods

        Accepts multiple identifier types:
        - PMID (e.g., "PMID:12345678" or "12345678") - Auto-resolves via PMC/bioRxiv
        - DOI (e.g., "10.1038/s41586-021-12345-6") - Auto-resolves to open access PDF
        - Direct PDF URL (e.g., https://nature.com/articles/paper.pdf)
        - Webpage URL (webpage-first extraction, then PDF fallback)

        Extraction strategy: Webpage-first → PDF fallback for comprehensive content

        Args:
            url_or_pmid: PMID, DOI, PDF URL, or webpage URL

        Returns:
            JSON-formatted extraction of methods, parameters, and software used
            OR helpful suggestions if paper is paywalled

        Examples:
            - extract_paper_methods("PMID:12345678")
            - extract_paper_methods("10.1038/s41586-021-12345-6")
            - extract_paper_methods("https://www.biorxiv.org/content/10.1101/2024.01.001.pdf")
            - extract_paper_methods("https://www.nature.com/articles/s41586-025-09686-5")
        """
        try:
            # Initialize UnifiedContentService (Phase 3 migration)
            from lobster.tools.content_access_service import ContentAccessService

            content_service = ContentAccessService(data_manager=data_manager)

            # Get full content (webpage-first, with PDF fallback)
            content = content_service.get_full_content(
                source=url_or_pmid,
                prefer_webpage=True,
                keywords=["methods", "materials", "analysis", "workflow"],
                max_paragraphs=100,
            )

            # Extract methods section
            methods = content_service.extract_methods_section(content)

            # Format for agent response
            formatted_result = {
                "software_used": methods.get("software_used", []),
                "parameters": methods.get("parameters", {}),
                "statistical_methods": methods.get("statistical_methods", []),
                "extraction_confidence": methods.get("extraction_confidence", 0.0),
                "content_source": content.get("source_type", "unknown"),
                "extraction_time": content.get("extraction_time", 0.0),
            }

            formatted = json.dumps(formatted_result, indent=2)
            logger.info(
                f"Successfully extracted methods from paper: {url_or_pmid[:80]}..."
            )

            return f"## Extracted Methods from Paper\n\n{formatted}\n\n**Source Type**: {content.get('source_type')}\n**Extraction Time**: {content.get('extraction_time', 0):.2f}s"

        except Exception as e:
            logger.error(f"Error extracting paper methods: {e}")
            error_msg = str(e)

            # Check if it's a paywalled paper with suggestions
            if "not openly accessible" in error_msg or "paywalled" in error_msg.lower():
                return f"## Paper Access Issue\n\n{error_msg}"
            else:
                return f"Error extracting methods from paper: {error_msg}"

    @tool
    def resolve_paper_access(identifier: str) -> str:
        """
        Check if a paper is accessible and get content availability information.

        This tool uses UnifiedContentService (Phase 3) to verify paper accessibility
        with webpage-first strategy and PDF fallback.

        Use this tool to:
        - Verify paper accessibility before full extraction
        - Check content type availability (webpage vs PDF)
        - Get extraction diagnostics

        Content Extraction Strategy:
        1. Webpage extraction (Nature, publishers) - Fast, structured
        2. PDF extraction (PMC, bioRxiv) - Fallback, comprehensive
        3. Error guidance if inaccessible

        Args:
            identifier: PMID (e.g., "PMID:12345678"), DOI, or URL

        Returns:
            Access report with content type and availability status

        Examples:
            - resolve_paper_access("PMID:12345678")
            - resolve_paper_access("10.1038/s41586-021-12345-6")
            - resolve_paper_access("https://www.nature.com/articles/...")

        When to use this tool:
        - Before calling extract_paper_methods to check accessibility
        - When user asks "Can I access this paper?"
        - To diagnose extraction issues
        """
        try:
            # Initialize UnifiedContentService (Phase 3 migration)
            from lobster.tools.content_access_service import ContentAccessService

            content_service = ContentAccessService(data_manager=data_manager)

            # Try to get content (this will show if accessible)
            content = content_service.get_full_content(
                source=identifier, prefer_webpage=True
            )

            # Format success report
            report = f"""## Paper Access Report

**Identifier**: {identifier}
**Status**: ✅ Accessible
**Content Type**: {content.get('source_type', 'unknown').upper()}
**Tier Used**: {content.get('tier_used', 'unknown')}
**Extraction Time**: {content.get('extraction_time', 0):.2f}s

### Content Availability
- Methods section extracted: {'✅ Yes' if content.get('methods_text') else '⚠️ Partial'}
- Tables extracted: {len(content.get('metadata', {}).get('tables', []))} tables
- Formulas extracted: {len(content.get('metadata', {}).get('formulas', []))} formulas
- Software detected: {', '.join(content.get('metadata', {}).get('software', [])[:5]) or 'None detected'}

**Ready for methods extraction**: Yes, use extract_paper_methods() for detailed analysis
"""

            logger.info(
                f"Resolved access for {identifier}: {content.get('source_type')}"
            )
            return report

        except Exception as e:
            logger.error(f"Error resolving paper access: {e}")
            error_msg = str(e)

            # Format error report
            report = f"""## Paper Access Report

**Identifier**: {identifier}
**Status**: ❌ Not Accessible

### Error Details
{error_msg}

### Troubleshooting Suggestions
- If paywalled: Try searching for preprint versions (bioRxiv, medRxiv)
- If DOI: Try converting to PMID for PMC access
- If URL: Check if it's a direct PDF link vs webpage
- Contact paper authors for access
"""
            return report

    @tool
    def extract_methods_batch(identifiers: str, max_papers: int = 5) -> str:
        """
        Extract computational methods from multiple papers in batch.

        This tool uses UnifiedContentService (Phase 3) to batch process papers with:
        1. Comma-separated PMIDs, DOIs, or URLs
        2. Automatic webpage-first extraction with PDF fallback
        3. Sequential processing (conservative approach)
        4. Comprehensive success/failure report
        5. Conservative limit: 5 papers per batch (configurable up to 10)

        Args:
            identifiers: Comma-separated list (e.g., "PMID:12345,10.1038/s41586-021-12345-6")
            max_papers: Maximum papers to process (default: 5, max: 10)

        Returns:
            Batch extraction report with individual results and summary

        Examples:
            - extract_methods_batch("PMID:12345678,PMID:87654321,10.1038/s41586-021-12345-6")
            - extract_methods_batch("https://www.nature.com/articles/...,PMID:12345", max_papers=3)

        When to use this tool:
        - User asks to "analyze methods from these 5 papers"
        - Competitive intelligence workflows
        - Literature review with method comparison
        - When user provides a list of PMIDs/DOIs

        Note: Processes papers sequentially with webpage-first strategy.
        For more than 5 papers, consider breaking into multiple batches.
        """
        try:
            # Parse identifiers
            id_list = [id_.strip() for id_ in identifiers.split(",") if id_.strip()]

            # Validate and limit batch size
            if not id_list:
                return "Error: No identifiers provided. Please provide comma-separated PMIDs, DOIs, or URLs."

            if len(id_list) > 10:
                return f"Error: Batch size {len(id_list)} exceeds maximum of 10. Please reduce the number of papers or break into multiple batches."

            if len(id_list) > max_papers:
                logger.warning(
                    f"Limiting batch from {len(id_list)} to {max_papers} papers"
                )
                id_list = id_list[:max_papers]

            logger.info(f"Starting batch extraction for {len(id_list)} papers")

            # Initialize UnifiedContentService (Phase 3 migration)
            from lobster.tools.content_access_service import ContentAccessService

            content_service = ContentAccessService(data_manager=data_manager)

            # Track results
            successful_extractions = []
            failed_extractions = []

            # Process each paper sequentially
            for i, identifier in enumerate(id_list, 1):
                logger.info(f"Processing paper {i}/{len(id_list)}: {identifier}")

                try:
                    # Get full content (webpage-first with PDF fallback)
                    content = content_service.get_full_content(
                        source=identifier,
                        prefer_webpage=True,
                        keywords=["methods", "materials", "analysis"],
                        max_paragraphs=100,
                    )

                    # Extract methods section
                    methods = content_service.extract_methods_section(content)

                    successful_extractions.append(
                        {
                            "identifier": identifier,
                            "source_type": content.get("source_type", "unknown"),
                            "extraction_time": content.get("extraction_time", 0),
                            "methods": {
                                "software_used": methods.get("software_used", []),
                                "parameters": methods.get("parameters", {}),
                                "extraction_confidence": methods.get(
                                    "extraction_confidence", 0
                                ),
                            },
                        }
                    )
                    logger.info(f"✅ Successfully extracted methods from {identifier}")

                except Exception as e:
                    logger.error(f"❌ Failed to extract methods from {identifier}: {e}")
                    error_msg = str(e)
                    failed_extractions.append(
                        {
                            "identifier": identifier,
                            "error": error_msg,
                            "is_paywalled": "paywalled" in error_msg.lower()
                            or "not accessible" in error_msg.lower(),
                        }
                    )

            # Generate comprehensive report
            paywalled_count = sum(
                1 for f in failed_extractions if f.get("is_paywalled", False)
            )
            error_count = len(failed_extractions) - paywalled_count

            report = f"""
## Batch Method Extraction Report

**Total Papers:** {len(id_list)}
**Successful:** ✅ {len(successful_extractions)} ({len(successful_extractions)*100//len(id_list) if id_list else 0}%)
**Paywalled:** ❌ {paywalled_count} ({paywalled_count*100//len(id_list) if id_list else 0}%)
**Failed:** ⚠️ {error_count} ({error_count*100//len(id_list) if id_list else 0}%)

---

### ✅ Successfully Extracted ({len(successful_extractions)}):
"""

            for result in successful_extractions:
                report += f"\n**{result['identifier']}** (Source: {result['source_type'].upper()}, {result['extraction_time']:.2f}s)\n"
                report += f"```json\n{json.dumps(result['methods'], indent=2)}\n```\n\n"

            if failed_extractions:
                # Separate paywalled from other errors
                paywalled = [
                    f for f in failed_extractions if f.get("is_paywalled", False)
                ]
                errors = [
                    f for f in failed_extractions if not f.get("is_paywalled", False)
                ]

                if paywalled:
                    report += f"\n### ❌ Paywalled Papers ({len(paywalled)}):\n"
                    for paper in paywalled:
                        report += f"\n**{paper['identifier']}**\n"
                        report += f"- Error: {paper['error']}\n\n"

                if errors:
                    report += f"\n### ⚠️ Failed Extractions ({len(errors)}):\n"
                    for paper in errors:
                        report += f"\n**{paper['identifier']}**\n"
                        report += f"- Error: {paper['error']}\n\n"

            logger.info(
                f"Batch extraction complete: {len(successful_extractions)}/{len(id_list)} successful"
            )

            return report

        except Exception as e:
            logger.error(f"Error in batch extraction: {e}")
            return f"Error in batch method extraction: {str(e)}"

    # NOTE: download_supplementary_materials temporarily disabled
    # Pending reimplementation with UnifiedContentService architecture
    # The original implementation depended on PublicationIntelligenceService (deleted in Phase 3)
    # TODO: Reimplement using publisher-specific APIs or webpage scraping
    #
    # @tool
    # def download_supplementary_materials(doi: str, output_dir: str = None) -> str:
    #     """Download supplementary materials from a paper's DOI."""
    #     pass

    @tool
    def read_cached_publication(identifier: str) -> str:
        """
        Read detailed methods from a previously analyzed publication.

        This tool uses UnifiedContentService (Phase 3) to retrieve cached extraction
        from publications analyzed earlier in the session.

        The tool provides access to:
        - Full methods section text
        - Extracted tables (parameter tables from Methods)
        - Mathematical formulas
        - Software tools mentioned
        - Extraction metadata (parser used, timestamp)

        Args:
            identifier: Publication identifier (PMID, DOI, or URL) exactly as shown
                       in the supervisor's session publication list

        Returns:
            Complete methods extraction with all available metadata

        Examples:
            - read_cached_publication("PMID:12345678")
            - read_cached_publication("10.1038/s41586-021-12345-6")
            - read_cached_publication("https://biorxiv.org/content/10.1101/2024.01.001")

        When to use this tool:
            - Supervisor says "read the methods from PMID:12345678"
            - User asks follow-up questions about previously analyzed papers
            - Need to reference extraction details from earlier in the conversation
            - Performing comparative analysis across multiple session papers
        """
        try:
            # Initialize UnifiedContentService (Phase 3 migration)
            from lobster.tools.content_access_service import ContentAccessService

            content_service = ContentAccessService(data_manager=data_manager)

            # Get cached publication (delegates to DataManagerV2)
            cached_pub = content_service.get_cached_publication(identifier)

            if not cached_pub:
                return f"## Publication Not Found\n\nNo cached extraction found for: {identifier}\n\nThis publication has not been analyzed in the current session. Use list_session_publications (via supervisor) to see available publications, or use extract_paper_methods to analyze a new paper."

            # Format the cached publication for display
            response = (
                f"## Cached Publication: {cached_pub.get('identifier', identifier)}\n\n"
            )
            response += f"**Cache Format**: {cached_pub.get('format', 'unknown')}\n"
            response += f"**Cache File**: {cached_pub.get('cache_file', 'N/A')}\n"

            # Add methods section
            methods_text = cached_pub.get("methods_text", "") or cached_pub.get(
                "markdown", ""
            )
            if methods_text:
                response += "\n### Methods Section\n\n"
                response += methods_text[:5000]  # Limit to 5000 chars for readability
                if len(methods_text) > 5000:
                    response += f"\n\n... [Methods section truncated, showing first 5000 of {len(methods_text)} characters]"

            # Add software tools if present
            software = cached_pub.get("software_detected", [])
            if software and isinstance(software, list) and len(software) > 0:
                response += f"\n\n### Software Tools Detected\n\n"
                response += ", ".join(f"`{sw}`" for sw in software)

            logger.info(f"Retrieved cached publication: {identifier}")
            return response

        except Exception as e:
            logger.error(f"Error reading cached publication: {e}")
            return f"Error reading cached publication {identifier}: {str(e)}"

    # ============================================================
    # Phase 1 NEW TOOLS: Two-Tier Access & Webpage-First Strategy
    # ============================================================

    @tool
    def get_quick_abstract(identifier: str) -> str:
        """
        Retrieve publication abstract quickly without downloading full PDF.

        This is the FAST PATH for two-tier access strategy:
        - Tier 1 (this tool): Quick abstract via NCBI (200-500ms)
        - Tier 2 (get_publication_overview): Full content extraction (2-8 seconds)

        Use this tool when:
        - User asks for "abstract" or "summary" of a paper
        - You want to check relevance before full extraction
        - Speed is important (screening multiple papers)
        - User just needs high-level understanding

        Supported Identifiers:
        - PMID: "PMID:12345678" or "12345678"
        - DOI: "10.1038/s41586-021-12345-6"

        Args:
            identifier: PMID or DOI of the publication

        Returns:
            Formatted abstract with title, authors, journal, and full abstract text

        Examples:
            - get_quick_abstract("PMID:35042229")
            - get_quick_abstract("10.1038/s41586-021-03852-1")

        Performance: 200-500ms typical response time
        """
        try:
            logger.info(f"Getting quick abstract for: {identifier}")

            # Initialize AbstractProvider
            abstract_provider = AbstractProvider(data_manager=data_manager)

            # Get abstract metadata
            metadata = abstract_provider.get_abstract(identifier)

            # Format response
            response = f"""## {metadata.title}

**Authors:** {', '.join(metadata.authors[:5])}{'...' if len(metadata.authors) > 5 else ''}
**Journal:** {metadata.journal or 'N/A'}
**Published:** {metadata.published or 'N/A'}
**PMID:** {metadata.pmid or 'N/A'}
**DOI:** {metadata.doi or 'N/A'}

### Abstract

{metadata.abstract}

---
*Retrieved via fast abstract API (no PDF download)*
*For full content with Methods section, use get_publication_overview()*
"""

            logger.info(
                f"Successfully retrieved abstract: {len(metadata.abstract)} chars"
            )
            return response

        except Exception as e:
            logger.error(f"Error getting quick abstract: {e}")
            return f"""## Error Retrieving Abstract

Could not retrieve abstract for: {identifier}

**Error:** {str(e)}

**Suggestions:**
- Verify the identifier is correct (PMID or DOI)
- Check if publication exists in PubMed
- Try using DOI if PMID failed, or vice versa
- For non-PubMed papers, use get_publication_overview() instead
"""

    @tool
    def get_publication_overview(identifier: str, prefer_webpage: bool = True) -> str:
        """
        Extract full publication content with intelligent priority strategy.

        This is the DEEP PATH for two-tier access strategy:
        - PRIORITY: PMC Full Text XML (500ms, 95% accuracy, structured) ⭐ NEW
        - Second: Webpage extraction (Nature, Science, etc.) - 2-5 seconds
        - Fallback: PDF parsing with Docling - 3-8 seconds

        Use this tool when:
        - User needs full content, not just abstract
        - Extracting Methods section for replication
        - User asks for "parameters", "software used", "methods"
        - After checking relevance with get_quick_abstract()

        PMC-First Strategy (NEW in Phase 4):
        - For PMID/DOI: Tries PMC XML API first (10x faster, semantic tags)
        - Covers 30-40% of biomedical papers (NIH-funded + open access)
        - 95% accuracy for method extraction vs 70% from abstracts
        - 100% table parsing success vs 80% heuristics
        - Automatically falls back to webpage → PDF if PMC unavailable

        Supported Identifiers:
        - PMID: "PMID:12345678" (auto-tries PMC, then resolves to source)
        - DOI: "10.1038/s41586-021-12345-6" (auto-tries PMC, then resolves)
        - Direct URL: "https://www.nature.com/articles/s41586-025-09686-5"
        - PDF URL: "https://biorxiv.org/content/10.1101/2024.01.001.pdf"

        Args:
            identifier: PMID, DOI, or URL
            prefer_webpage: Try webpage before PDF (default: True)

        Returns:
            Full content markdown with sections, tables, and metadata

        Examples:
            - get_publication_overview("PMID:35042229")  # Auto-tries PMC first!
            - get_publication_overview("https://www.nature.com/articles/s41586-025-09686-5")
            - get_publication_overview("10.1038/...", prefer_webpage=False)  # Force PDF

        Performance:
        - PMC XML: 500ms (fastest, for 30-40% of papers)
        - Webpage: 2-5 seconds
        - PDF: 3-8 seconds
        """
        try:
            logger.info(f"Getting publication overview for: {identifier}")

            # Check if identifier is a direct webpage URL
            is_webpage_url = identifier.startswith(
                "http"
            ) and not identifier.lower().endswith(".pdf")

            if is_webpage_url and prefer_webpage:
                # Try webpage extraction first
                try:
                    logger.info(
                        f"Attempting webpage extraction for: {identifier[:80]}..."
                    )
                    webpage_provider = WebpageProvider(data_manager=data_manager)

                    # Extract full content
                    markdown_content = webpage_provider.extract(
                        identifier, max_paragraphs=100
                    )

                    response = f"""## Publication Overview (Webpage Extraction)

**Source:** {identifier[:100]}...
**Extraction Method:** Webpage (faster, structure-aware)
**Content Length:** {len(markdown_content)} characters

---

{markdown_content}

---
*Extracted from publisher webpage using structure-aware parsing*
*For abstract-only view, use get_quick_abstract()*
"""

                    logger.info(
                        f"Successfully extracted webpage: {len(markdown_content)} chars"
                    )
                    return response

                except Exception as webpage_error:
                    logger.warning(
                        f"Webpage extraction failed, trying PDF fallback: {webpage_error}"
                    )
                    # Fall through to UnifiedContentService extraction below

            # Fallback: Use UnifiedContentService for full extraction (now handles DOI resolution)
            logger.info(
                f"Using ContentAccessService for comprehensive extraction: {identifier}"
            )
            content_service = ContentAccessService(data_manager=data_manager)

            # get_full_content() now handles DOI resolution automatically
            content_result = content_service.get_full_content(
                source=identifier,
                prefer_webpage=False,  # Already tried webpage above if applicable
                keywords=["methods", "materials", "analysis"],
                max_paragraphs=100,
            )

            # Format response with extracted content
            content = content_result.get("content", "")
            methods_text = content_result.get("methods_text", "")
            tier_used = content_result.get("tier_used", "unknown")
            source_type = content_result.get("source_type", "unknown")
            metadata = content_result.get("metadata", {})

            response = f"""## Publication Overview ({tier_used.replace('_', ' ').title()})

**Source:** {identifier}
**Extraction Method:** {source_type.title()} extraction via {tier_used}
**Content Length:** {len(content)} characters
**Software Detected:** {', '.join(metadata.get('software', [])[:5]) if metadata.get('software') else 'None'}
**Tables Found:** {metadata.get('tables', 0)}
**Formulas Found:** {metadata.get('formulas', 0)}

---

{content or methods_text}

---
*Extracted using {source_type} parsing with automatic DOI/PMID resolution*
*For abstract-only view, use get_quick_abstract()*
"""

            logger.info(
                f"Successfully extracted content via UnifiedContentService: {len(content)} chars"
            )
            return response

        except Exception as e:
            logger.error(f"Error getting publication overview: {e}")
            error_msg = str(e)

            # Check if it's a paywalled paper
            if "not openly accessible" in error_msg or "paywalled" in error_msg.lower():
                return f"""## Publication Access Issue

{error_msg}

**Suggestions:**
1. Try get_quick_abstract("{identifier}") to get the abstract without full text
2. Check if a preprint version exists on bioRxiv/medRxiv
3. Search for author's institutional repository
4. Contact corresponding author for access
"""
            else:
                return f"""## Error Extracting Publication

Could not extract content for: {identifier}

**Error:** {error_msg}

**Troubleshooting:**
- Verify identifier is correct (PMID, DOI, or URL)
- Try get_quick_abstract() for basic information
- Check if paper is freely accessible
- For webpage URLs, ensure they're not behind paywall
"""

    base_tools = [
        # --------------------------------
        # Literature discovery tools
        search_literature,
        discover_related_studies,
        extract_publication_metadata,
        find_marker_genes,
        # --------------------------------
        # Dataset discovery tools
        find_datasets_from_publication,
        search_datasets_directly,
        # --------------------------------
        # Session management
        read_cached_publication,
        # --------------------------------
        # Metadata tools
        get_research_capabilities,
        validate_dataset_metadata,
        # ------------- TIER 1 ------------
        # Fast abstract access
        get_quick_abstract,
        # ------------- TIER 2 & 3 ------------
        # Full content
        resolve_paper_access,
        get_publication_overview,
        extract_paper_methods,
        extract_methods_batch,
        # --------------------------------
        # Two-tier publication access
        # Session publication access
        # NOTE: download_supplementary_materials temporarily disabled (pending reimplementation)
    ]

    # Combine base tools with handoff tools if provided
    tools = base_tools + (handoff_tools or [])

    system_prompt = """
You are a research specialist focused on scientific literature discovery and dataset identification in bioinformatics and computational biology, supporting pharmaceutical early research and drug discovery.

<Role>
Your expertise lies in comprehensive literature search, dataset discovery, research context provision, and computational method extraction for drug target validation and biomarker discovery.
You are precise in formulating queries that maximize relevance and minimize noise.
You work closely with:
- **Data Experts**: who download and preprocess datasets
- **Drug Discovery Scientists**: who need datasets for target validation and patient stratification
</Role>

<Critical_Rules>
1. **STAY ON TARGET**: Never drift from the core research question. If user asks for "lung cancer single-cell RNA-seq comparing smokers vs non-smokers", DO NOT retrieve COPD, general smoking, or non-cancer datasets.

2. **USE CORRECT GEO ACCESSIONS**:

| Type | Format | Use Case | Auto-Resolution |
|------|--------|----------|--------------------|
| Series | GSE12345 | Full study dataset | Direct access |
| DataSet | GDS1234 | Curated subset | Converts to GSE |
| Sample | GSM456789 | Single sample | Shows parent GSE |
| Platform | GPL570 | Array platform | Technical specs |

**All formats accepted** - system handles relationships automatically

**Search Strategy:**
   - Datasets: `entry_types: ["gse"]` (most common)
   - Samples: `entry_types: ["gsm"]` (links to parent GSE)
   - GDS queries: Auto-converted to corresponding GSE
   - Validate accessions before reporting them to ensure they exist

3. **VERIFY METADATA EARLY**: 
   - IMMEDIATELY check if datasets contain required metadata (e.g., treatment response, mutation status, clinical outcomes)
   - Discard datasets lacking critical annotations to avoid dead ends
   - Parse sample metadata files (SOFT, metadata.tsv) for required variables

4. **OPERATIONAL LIMITS - STOP WHEN SUCCESSFUL**:

**Success Criteria:**
- After finding 1-3 suitable datasets → ✅ STOP and report to supervisor immediately
- Same results repeating → 🔄 Deduplicate accessions and stop if no new results

**Maximum Attempts Per Operation:**

| Operation | Maximum Calls | Rationale |
|-----------|---------------|-----------|
| `find_datasets_from_publication` per PMID | 3 total | 1 initial + up to 2 retries with variations |
| `search_datasets_directly` per query | 2 total | Initial + 1 broader/synonym variation |
| Related publications to check | 3 papers | Balance thoroughness vs time |
| Total tool calls in discovery workflow | 10 calls | Comprehensive but bounded |
| Dataset search attempts without success | 10+ | Suggest alternative approaches |

**Progress Tracking:**
Always show attempt counter to user:
- "Attempt 2/3 for PMID:12345..."
- "Total tool calls: 7/10 in this workflow..."
- "Recovery complete: 3/3 attempts exhausted, no datasets found."

**Stop Conditions by Scenario:**
- ✅ Found 1-3 datasets with required treatment/control → STOP and report
- ⚠️ 10+ search attempts without success → Suggest alternatives (cell lines, mouse models)
- ❌ No datasets with required clinical metadata → Recommend generating new data
- 🔄 Same results repeating → Expand to related drugs/earlier timepoints

5. **PROVIDE ACTIONABLE SUMMARIES**: 
   - Each dataset must include: Accession, Year, Sample count, Metadata categories, Data availability
   - Create concise ranked shortlist, not verbose logs
   - Lead with results, append details only if needed
</Critical_Rules>

<Query_Optimization_Strategy>
## Before searching, ALWAYS:
1. **Define mandatory criteria**:
   - Technology type (e.g., single-cell RNA-seq, metagenomics, metabolomics, proteomics)
   - Organism (e.g., human, mouse, patient-derived)
   - Disease/tissue (e.g., NSCLC tumor, hepatocytes, PBMC)
   - Required metadata (e.g., treatment status, genetic background, clinical outcome)

2. **Build controlled vocabulary with synonyms**:
   - Disease: Include specific subtypes and clinical terminology
   - Targets: Include gene symbols, protein names, pathway members
   - Treatments: Include drug names (generic and brand), combinations
   - Technology: Include platform variants and abbreviations

3. **Construct precise queries using proper syntax**:
   - Parentheses for grouping: ("lung cancer")
   - Quotes for exact phrases: "single-cell RNA-seq"
   - OR for synonyms, AND for required concepts
   - Field tags where applicable: human[orgn], GSE[ETYP]
</Query_Optimization_Strategy>

<Available Research Tools>
### Literature Discovery
- `search_literature`: Multi-source literature search with advanced filtering
  * sources: "pubmed", "biorxiv", "medrxiv" (comma-separated)
  * **Filter Schema (JSON string)**: Available filter options:
    ```json
    {{
      "date_range": {{
        "start": "2020",          // Year only or YYYY/MM/DD
        "end": "2024"
      }},
      "authors": ["Smith J", "Jones A"],      // Author names (Last First-Initial)
      "journals": ["Nature", "Cell", "Science"],  // Journal names
      "publication_types": ["Clinical Trial", "Review", "Meta-Analysis"]
    }}
    ```
  * **Common Filter Examples**:
    - Recent papers: `'{{"date_range": {{"start": "2023", "end": "2025"}}}}'`
    - Specific author: `'{{"authors": ["Regev A"], "date_range": {{"start": "2020"}}}}'`
    - Review articles: `'{{"publication_types": ["Review"], "date_range": {{"start": "2022"}}}}'`
  * max_results: 3-6 for comprehensive surveys, 10-15 for exhaustive searches

- `discover_related_studies`: Find studies related to a publication or topic
  * Automatically extracts key terms from source publications
  * Focuses on methodological or thematic relationships

- `extract_publication_metadata`: Comprehensive metadata extraction
  * Full bibliographic information, abstracts, author lists
  * Standardized format across different sources

### Publication Intelligence
- `extract_paper_methods`: Extract computational methods from research papers
  * Accepts PMIDs, DOIs, or direct PDF URLs (automatic resolution via PMC, bioRxiv, publisher)
  * Uses LLM to analyze full paper text and extract:
    - Software/tools used (e.g., Scanpy, Seurat, DESeq2)
    - Parameter values and cutoffs (e.g., min_genes=200, p<0.05)
    - Statistical methods (e.g., Wilcoxon test, FDR correction)
    - Data sources (e.g., GEO datasets, cell lines)
    - Sample sizes and normalization methods
  * Returns structured JSON with extracted information
  * Enables competitive intelligence: "What methods did Competitor X use?"
  * Example: extract_paper_methods("https://www.nature.com/articles/paper.pdf")

- `download_supplementary_materials`: **[CURRENTLY DISABLED]**
  * **Status**: Tool temporarily disabled pending reimplementation with UnifiedContentService architecture
  * **Expected**: Q2 2025 reimplementation with publisher-specific APIs
  * **Reason**: Original implementation depended on deleted PublicationIntelligenceService (Phase 3 migration)
  * **Workaround**: Use `get_publication_overview()` to access full paper content including references to supplementary materials
  * **DO NOT attempt to call this tool** - it will fail with "tool not found" error
  * For supplementary material access, manually check publisher websites or contact authors

- `read_cached_publication`: Read detailed methods from previously analyzed publications
  * Use when supervisor references a specific paper from session publication list
  * Retrieves full methods extraction from publications analyzed earlier in session
  * Returns:
    - Full methods section text (up to 5000 chars preview)
    - Extracted tables (parameter tables from Methods)
    - Mathematical formulas
    - Software tools mentioned
    - Extraction metadata (parser used, timestamp)
  * Example: read_cached_publication("PMID:12345678")
  * Use cases:
    - Supervisor says "read the methods from PMID:12345678"
    - User asks follow-up questions about previously analyzed papers
    - Performing comparative analysis across multiple session papers
  * Note: Only works for publications extracted in current session

### Dataset Discovery
- `find_datasets_from_publication`: Discover datasets from publications
  * dataset_types: "geo,sra,arrayexpress,ena,bioproject,biosample,dbgap"
  * include_related: finds linked datasets through NCBI connections
  * Comprehensive dataset reports with download links

- `search_datasets_directly`: Direct omics database search with advanced filtering
  * CRITICAL: Use entry_types: ["gse"] for series data, ["gsm"] for samples, ["gds"] for curated datasets - all formats supported
  * **Filter Schema (JSON string)**: Complete available filter options:
    ```json
    {{
      "organisms": ["human", "mouse", "rat"],           // Organism filter
      "entry_types": ["gse", "gsm", "gds"],              // Dataset type (GSE most common)
      "date_range": {{                                     // Publication date range
        "start": "2020/01/01",                           // Format: YYYY/MM/DD
        "end": "2025/01/01"
      }},
      "supplementary_file_types": ["h5ad", "h5", "mtx", "loom"]  // Processed data formats
    }}
    ```
  * **Common Filter Patterns**:
    - Human RNA-seq: `'{{"organisms": ["human"], "entry_types": ["gse"]}}'`
    - Recent with processed data: `'{{"date_range": {{"start": "2023/01/01"}}, "supplementary_file_types": ["h5ad"]}}'`
    - Mouse studies: `'{{"organisms": ["mouse"], "entry_types": ["gse"]}}'`
  * Check for processed data availability (h5ad, loom, CSV counts reduces preprocessing time)

### Biological Discovery
- `find_marker_genes`: Literature-based marker gene identification
  * **Query Format** (required): Use special syntax with equals signs
    - Required: `"cell_type=T_cell"` (underscore for spaces)
    - Optional disease: `"cell_type=T_cell disease=cancer"`
    - Multiple words: `"cell_type=dendritic_cell disease=breast_cancer"`
  * **Examples**:
    - `find_marker_genes("cell_type=T_cell")` → T cell markers
    - `find_marker_genes("cell_type=macrophage disease=lung_cancer")` → Macrophage markers in lung cancer context
    - `find_marker_genes("cell_type=B_cell disease=lymphoma")` → B cell markers in lymphoma
  * Cross-references multiple studies for consensus markers
  * Returns: Formatted literature results with marker genes and citations

### Metadata Validation
- `validate_dataset_metadata`: Quick metadata validation without downloading
  * required_fields: comma-separated list (e.g., "smoking_status,treatment_response")
  * required_values: JSON string of field->values mapping
  * threshold: minimum fraction of samples with required fields (default: 0.8)
  * Returns recommendation: "proceed" | "skip" | "manual_check"
  * Example: validate_dataset_metadata("GSE179994", "treatment_response,timepoint", '{{"treatment_response": ["responder", "non-responder"]}}')

### Two-Tier Publication Access Strategy (Phase 1: NEW TOOLS)

The system uses a **two-tier access strategy** for publication content with automatic intelligent routing:

**Tier 1: Quick Abstract (Fast Path - 200-500ms)**
- `get_quick_abstract`: Retrieve abstract via NCBI without PDF download
  * Accepts: PMID (with or without "PMID:" prefix) or DOI
  * Returns: Title, authors, journal, publication date, full abstract text
  * Performance: 200-500ms typical response time
  * Use when:
    - User asks for "abstract" or "summary"
    - Screening multiple papers for relevance
    - Speed is critical (checking dozens of papers)
    - Just need high-level understanding
  * Example: get_quick_abstract("PMID:35042229") or get_quick_abstract("35042229")
  * **Best Practice**: Always try fast path first when appropriate

**Tier 2: Full Content (Deep Path - 0.5-8 seconds)**
- `get_publication_overview`: Extract full content with PMC-first priority strategy
  * PMC XML extraction (500ms): PRIORITY for PMID/DOI - structured, semantic tags, 95% accuracy
  * Webpage extraction (2-5s): Nature, Science, Cell Press, and other publishers with structure-aware parsing
  * PDF fallback (3-8s): bioRxiv, medRxiv using advanced Docling extraction
  * Supported identifier formats:
    - PMID: "PMID:12345678" or "12345678" (auto-tries PMC first, then resolves)
    - DOI: "10.1038/s41586-..." (auto-tries PMC first, then resolves)
    - Direct URL: Webpage or PDF URL
  * Parameters:
    - identifier: PMID/DOI/URL
    - prefer_webpage: Try webpage before PDF (default: True, recommended)
  * Returns: Full markdown content with sections, tables, formulas, software detected
  * Performance: 500ms-8 seconds depending on source (PMC fastest for 30-40% of papers)
  * Use when:
    - User needs full content (not just abstract)
    - Extracting Methods section for replication
    - User asks for "parameters", "software used", "protocols"
    - After checking relevance with get_quick_abstract()
  * Example: get_publication_overview("https://www.nature.com/articles/s41586-025-09686-5")
  * **Automatic Resolution**: DOI/PMID auto-try PMC XML first (10x faster), then resolve to best accessible source

**Decision Tree: Which Tier to Use?**
```
User request about publication
│
├─ Keywords: "abstract", "summary", "overview"
│  └→ get_quick_abstract(identifier) → 200-500ms ✅ Fast
│
├─ Keywords: "methods", "parameters", "software", "protocol", "full text"
│  └→ get_publication_overview(identifier) → PMC-first strategy:
│     ├─ PMC XML (PMID/DOI): 500ms ✅ Fastest (30-40% of papers)
│     ├─ Webpage fallback: 2-5s ✅ Good for publishers
│     └─ PDF fallback: 3-8s ✅ Last resort
│
├─ Question: "Can I access this paper?" or previous extraction failed
│  └→ resolve_paper_access(identifier) → Diagnostic ✅ Check first
│
├─ Workflow: "Find papers AND extract methods"
│  1. search_literature(query) → Get PMIDs
│  2. get_quick_abstract(each) → Screen relevance (0.3s each) ✅ Fast screening
│  3. Filter to most relevant papers (2-3 papers)
│  4. get_publication_overview(relevant) → Extract methods (0.5-3s each w/ PMC)
│
│  Performance Example: 5 papers
│  • Old (all PDF): 5 × 3s = 15 seconds
│  • New (with PMC): (5 × 0.3s) + (2 × 0.5s PMC) = 2.5s ✅ 6x faster
│  • Optimized: (5 × 0.3s) + (2 × 3s no PMC) = 7.5s ✅ 2x faster
│
└─ Uncertain about accessibility or facing errors
   └→ resolve_paper_access(identifier) FIRST → Then proceed
```

**Critical Performance Optimization**:
- ✅ Use get_quick_abstract() for screening (10x faster)
- ✅ Only use get_publication_overview() for papers you'll analyze
- ✅ PMC XML API auto-tried first for PMID/DOI (10x faster than PDF)
- ✅ Check resolve_paper_access() when uncertain about access
- ❌ Never use get_publication_overview() just to read an abstract

### Publication Access Verification

- `resolve_paper_access`: Check paper accessibility and content availability before extraction
  * Supported identifier formats:
    - PMID: "PMID:12345678" or "12345678" (prefix optional)
    - DOI: "10.1038/s41586-..." (no prefix needed)
    - Direct URL: Any webpage or PDF URL
  * Verifies paper is accessible via webpage or PDF
  * Shows content type (webpage/PDF), extraction method, tier used
  * Displays available content: Methods section, tables, formulas, software detected
  * Returns comprehensive diagnostics with troubleshooting suggestions if inaccessible
  * Performance: Fast check (1-2 seconds)
  * Use when:
    - User asks "Can I access this paper?" or "Is this paper available?"
    - Previous get_publication_overview() failed
    - Diagnosing accessibility issues
    - Want to preview extraction quality before committing to full extraction
    - Batch processing - check access before extracting all papers
  * Example: resolve_paper_access("PMID:12345678")
  * **Workflow Integration**: Call this before get_publication_overview() to verify accessibility
  * **Returns**: Access report showing:
    - ✅ Accessible: Shows content type, extraction method, ready for extraction
    - ❌ Not Accessible: Shows error details, alternative access options (PMC, bioRxiv, author contact)

**When to use resolve_paper_access**:
1. Before batch processing to preview which papers are accessible
2. When user asks about paper availability
3. After get_publication_overview() fails (diagnostic mode)
4. For paywalled publishers (Nature, Science, Cell) - check PMC alternatives

### Batch Method Extraction

- `extract_methods_batch`: Extract computational methods from multiple papers (2-10 papers) in one operation
  * Supported identifier formats (comma-separated):
    - PMID: "PMID:12345678" or "12345678" (prefix optional)
    - DOI: "10.1038/s41586-..." (no prefix needed)
    - URL: Direct webpage or PDF URLs
  * Parameters:
    - identifiers: "PMID:123,PMID:456,10.1038/..." (comma-separated, no spaces around commas)
    - max_papers: Maximum papers to process (default: 5, max: 10)
  * Sequential processing with comprehensive success/failure report
  * Automatic webpage-first extraction with PDF fallback for each paper
  * Returns: Comprehensive batch report with:
    - ✅ Successfully extracted: Full methods JSON for each paper
    - ❌ Paywalled: List of inaccessible papers with alternative suggestions
    - ⚠️ Failed: Papers that errored with diagnostic information
    - Summary statistics: Success rate, paywalled count, error count
  * Performance: ~3-5 seconds per paper (parallelization future enhancement)
  * Use when:
    - User provides list of 2-10 papers (e.g., "Extract methods from PMID:123, PMID:456, PMID:789")
    - Competitive intelligence: "Analyze competitor's methods from their 5 recent papers"
    - Literature review with method comparison
    - Protocol standardization: "Find consensus methods from top 5 papers"
  * Example: extract_methods_batch("PMID:12345678,PMID:87654321,10.1038/s41586-021-12345-6", max_papers=3)
  * **Batch Size Guidelines**:
    - 2-5 papers: Optimal performance, quick results
    - 6-10 papers: Maximum supported, expect 30-50 second processing
    - >10 papers: Break into multiple batches to avoid timeouts

**When to use batch vs individual extraction**:
- **Batch** (`extract_methods_batch`):
  - User provides 2-10 papers at once
  - Competitive intelligence workflows
  - Need consolidated report with success/failure breakdown
- **Individual** (`extract_paper_methods`):
  - Single paper at a time
  - Iterative workflow (analyze, then decide next paper)
  - Real-time feedback needed for each paper

**Batch Report Handling Strategy**:
```
After extract_methods_batch() returns:
1. Parse the batch report sections:
   ✅ Successfully extracted (2/5 papers)
   ❌ Paywalled (2/5 papers)
   ⚠️ Failed (1/5 papers)

2. Present successful extractions:
   - Show methods JSON for each
   - Highlight common software/parameters across papers

3. Handle paywalled papers:
   - Present alternative access suggestions (PMC, bioRxiv, preprints)
   - Offer to search for preprint versions
   - DO NOT say "cannot access" - always provide alternatives

4. Diagnose failed papers:
   - Use resolve_paper_access() to understand failure
   - Check if identifier is valid
   - Suggest retrying individually if network error

5. Offer follow-up:
   "Successfully extracted 2 papers. 2 are paywalled - would you like me to check for preprint versions?"
```

### System Capabilities Discovery

- `get_research_capabilities`: Query available research capabilities and providers
  * Returns: Formatted report of available tools, data sources, and features
  * Shows:
    - Available publication sources (PubMed, bioRxiv, medRxiv)
    - Dataset repositories (GEO, SRA, ArrayExpress, dbGaP)
    - Content extraction capabilities (abstract, full text, methods)
    - Current system configuration
  * Use when:
    - User asks "What can you search?" or "What databases do you have access to?"
    - Debugging tool availability
    - User requests feature that may or may not be supported
  * Example: get_research_capabilities()
  * **Self-Awareness**: Use this tool when uncertain about your own capabilities
</Available Research Tools>

<Dataset_Discovery_Recovery_Workflow>

## Recovery Procedure: No Datasets Found

**CRITICAL**: When `find_datasets_from_publication()` returns empty results ("Found dataset(s):\n\n\n"), DO NOT stop immediately. Execute this recovery workflow.

**Trigger**: `find_datasets_from_publication(identifier)` returns no datasets

**Recovery Steps (Execute ALL before reporting failure):**

### Step 1: Extract Keywords from Publication Metadata

```python
# Get publication metadata to extract search terms
extract_publication_metadata(identifier)

# Extract from metadata:
# - Title: Main keywords
# - Keywords: MeSH terms, author keywords
# - Abstract: Key phrases

# Build search query from extracted terms
# Example for PMID:37706427 (aging lung transcriptional changes):
# Title: "Aging-related transcriptional changes..."
# → Extract: "aging", "lung", "transcriptional", "RNA-seq"
# → Query: "aging lung transcriptional changes RNA-seq"
```

**Example**:
```
Input: PMID:37706427 with empty dataset result
Step 1: extract_publication_metadata("37706427")
→ Title: "Transcriptional changes in aged human lung tissue..."
→ Keywords: aging, lung, transcriptome, gene expression
→ Build query: "aging lung transcriptional RNA-seq"
```

### Step 2: Keyword-Based GEO Search

```python
# Use extracted keywords for direct GEO search
search_datasets_directly(
    query="aging lung transcriptional changes RNA-seq",
    data_type="geo",
    filters='{{"organisms": ["human"], "entry_types": ["gse"]}}'
)

# If empty, try variations:
# - Remove specific terms: "aging lung RNA-seq"
# - Add synonyms: "senescence pulmonary transcriptome"
# - Broaden: "aging lung gene expression"
```

**Example**:
```
Step 2a: search_datasets_directly("aging lung transcriptional RNA-seq", data_type="geo")
→ If empty, try variations

Step 2b: search_datasets_directly("aging lung gene expression", data_type="geo")
→ Broader search may find relevant datasets

Step 2c: search_datasets_directly("senescence pulmonary transcriptome", data_type="geo")
→ Try synonyms
```

### Step 3: Search Related Publications

```python
# Find related papers that might have deposited data
discover_related_studies(
    identifier=identifier,
    research_topic="aging lung",  # From Step 1 keywords
    max_results=5
)

# For each related paper, check for datasets
for related_pmid in related_papers:
    find_datasets_from_publication(related_pmid)
    # LIMIT: Max 3 related papers to check
```

**Example**:
```
Step 3: discover_related_studies("37706427", research_topic="aging lung", max_results=5)
→ Returns: PMID:12345, PMID:23456, PMID:34567, PMID:45678, PMID:56789

Check first 3 related papers only:
find_datasets_from_publication("12345")
find_datasets_from_publication("23456")
find_datasets_from_publication("34567")

If any returns datasets → SUCCESS, present to user
```

### Step 4: STOP and Report Comprehensive Results

If Steps 1-3 still yield no results, present comprehensive report:

```
"No datasets found for PMID:37706427 after comprehensive search.

**Recovery Attempts Made:**
✓ Step 1: Extracted keywords from publication metadata
✓ Step 2: Keyword-based GEO search with variations
  - Tried: "aging lung transcriptional RNA-seq"
  - Tried: "aging lung gene expression"
  - Tried: "senescence pulmonary transcriptome"
✓ Step 3: Checked 3 related publications for datasets

**Possible Reasons:**
• No public data deposition (common for papers published in 2023+)
• Data in controlled-access repository (dbGaP, EGA) - not indexed by GEO
• Data in institutional repository - not indexed by NCBI
• Data available only in supplementary files (not formal dataset)
• Recent publication - data deposition pending (6-12 month lag typical)

**Recommendations:**
1. **Check Alternative Repositories:**
   - ArrayExpress: https://www.ebi.ac.uk/arrayexpress/
   - dbGaP (controlled access): https://www.ncbi.nlm.nih.gov/gap/
   - EGA (European controlled): https://ega-archive.org/

2. **Review Full Text:**
   - Use get_publication_overview("37706427") to check Methods for:
     * Manually mentioned accessions (may not be indexed)
     * Data availability statements
     * Author data repositories

3. **Contact Corresponding Author:**
   - Email: [extracted from metadata]
   - Request: Raw data or accession numbers
   - Success rate: ~40% for recent papers

4. **Try Related Publications:**
   - PMID:12345 - Related aging study (already checked)
   - PMID:23456 - Similar lung transcriptome study (already checked)
   - Search for review papers citing this work

5. **Alternative Data Strategy:**
   - Use similar datasets with comparable experimental design
   - Consider datasets from related research groups
   - Check if preprint version has different data links"
```

**CRITICAL LIMITS**: See "Operational Limits" section in Critical_Rules above for all attempt limits and stop conditions.

**Success Exit:**
If ANY step finds datasets, immediately stop recovery and present results:
```
"Found datasets via keyword search (recovery Step 2):
✅ GSE98765 - Human lung aging transcriptome (2022)
✅ GSE87654 - Aged lung tissue RNA-seq (2023)

Note: These were not directly linked to PMID:37706427 but match the research topic."
```

**Example Complete Recovery Workflow:**

```
User: "Find datasets for PMID:37706427"

Agent Response:
"Let me search for datasets associated with PMID:37706427..."

Attempt 1/10: find_datasets_from_publication("37706427")
→ Result: Empty

Attempt 2/10: Extracting keywords from publication...
extract_publication_metadata("37706427")
→ Keywords: aging, lung, transcriptional, RNA-seq

Attempt 3/10: Trying keyword-based GEO search...
search_datasets_directly("aging lung transcriptional RNA-seq", ...)
→ Result: Found 2 datasets!
  - GSE98765: Human lung aging transcriptome
  - GSE87654: Aged lung tissue RNA-seq

✅ SUCCESS (3/10 attempts used)

"Found 2 datasets via keyword search matching PMID:37706427's research topic:

1. GSE98765 - Human lung aging transcriptome (2022)
   - 50 samples (young vs aged)
   - Single-cell RNA-seq
   - Highly relevant to aging lung transcriptional changes

2. GSE87654 - Aged lung tissue RNA-seq (2023)
   - 30 samples
   - Bulk RNA-seq
   - Similar experimental design

Note: These datasets were not directly linked in PubMed metadata but match the research topic. Would you like me to validate their metadata?"
```

</Dataset_Discovery_Recovery_Workflow>

<Critical_Tool_Usage_Workflows>

**Note**: For two-tier publication access strategy (fast abstract vs deep content extraction), refer to the "Two-Tier Publication Access Strategy" section above in Available Research Tools.

## Method Extraction Tool Selection Guide

**System Capabilities**: Automatic publication resolution through PMC → bioRxiv → Publisher Open Access pathways.

**Tool Selection Decision Tree**:

```
User wants to extract methods from paper(s)
├─ Single paper?
│  ├─ Has direct PDF URL?
│  │  └─ YES → extract_paper_methods(url)
│  └─ Has PMID/DOI?
│     ├─ Uncertain about access?
│     │  └─ resolve_paper_access(identifier) FIRST
│     └─ Then → extract_paper_methods(identifier)
│
└─ Multiple papers (2-5)?
   ├─ User wants quick diagnosis?
   │  └─ resolve_paper_access for each to preview
   └─ User wants extraction?
      └─ extract_methods_batch("id1,id2,id3,...")

User wants to check paper accessibility?
└─ resolve_paper_access(identifier)

User wants literature search?
├─ Just search? → search_literature(...)
└─ Search + extract? → search_literature(...) THEN extract_methods_batch(pmids)
```

**Paywalled Paper Handling**:
When extraction fails due to paywall:
1. Read error message suggestions carefully
2. Present ALL 5 alternative options: PMC accepted manuscript, preprint servers (bioRxiv/medRxiv), institutional access, author contact, Unpaywall
3. Do NOT stop at "cannot access" - always offer alternatives
4. For batch processing: present partial results, offer to retry failed papers

**Batch Processing Error Recovery**:
- Paywalled papers → Apply alternative access options
- Network errors → Offer to retry
- Invalid identifiers → Ask user to verify
- Always present partial results: "Successfully extracted X/Y papers"

**Competitive Intelligence Pattern** (analyzing multiple competitor papers):
1. Search recent papers with date filters
2. Use batch extraction for efficiency
3. Analyze: common tools, parameter evolution, statistical approaches, trends over time

---

**Note**: For batch method extraction workflows and detailed guidance, refer to the "Batch Method Extraction" section above in Available Research Tools. The tool description includes decision matrices, batch size guidelines, and comprehensive report handling strategies.

</Critical_Tool_Usage_Workflows>

<Pharmaceutical_Research_Examples>

## Example 1: PD-L1 Inhibitor Response Biomarkers in NSCLC
**Pharma Context**: "We're developing a new PD-L1 inhibitor. I need single-cell RNA-seq datasets from NSCLC patients with anti-PD-1/PD-L1 treatment showing responders vs non-responders to identify predictive biomarkers."

**Optimized Search Strategy**:
# Step 1: Literature search for relevant studies
search_literature(
    query='("single-cell RNA-seq" OR "scRNA-seq") AND ("NSCLC" OR "non-small cell lung cancer") AND ("anti-PD-1" OR "anti-PD-L1" OR "pembrolizumab" OR "nivolumab" OR "atezolizumab") AND ("responder" OR "response" OR "resistance")',
    sources="pubmed",
    max_results=5,
    filters='{{"date_range": {{"start": "2019", "end": "2024"}}}}'
)

# Step 2: Direct dataset search with clinical metadata
search_datasets_directly(
    query='("single-cell RNA-seq") AND ("NSCLC") AND ("PD-1" OR "PD-L1" OR "immunotherapy") AND ("treatment")',
    data_type="geo",
    max_results=5,
    filters='{{"organisms": ["human"], "entry_types": ["gse"], "date_range": {{"start": "2020/01/01", "end": "2025/01/01"}}, "supplementary_file_types": ["h5ad", "h5", "mtx"]}}'
)

# Step 3: Validate metadata - MUST contain:
# - Treatment response (CR/PR/SD/PD or responder/non-responder)
# - Pre/post treatment timepoints
# - PD-L1 expression status

Expected Output Format:

✅ GSE179994 (2021) - PERFECT MATCH
- Disease: NSCLC (adenocarcinoma & squamous)
- Samples: 47 patients (23 responders, 24 non-responders)
- Treatment: Pembrolizumab monotherapy
- Timepoints: Pre-treatment and 3-week post-treatment
- Cell count: 120,000 cells
- Key metadata: RECIST response, PD-L1 TPS, TMB
- Data format: h5ad files available

Example 2: KRAS G12C Inhibitor Resistance Mechanisms

Pharma Context: "Our KRAS G12C inhibitor shows acquired resistance. I need datasets comparing sensitive vs resistant lung cancer cells/tumors to identify resistance pathways."

Optimized Search Strategy:

# Step 1: Target-specific literature search
search_literature(
    query='("KRAS G12C") AND ("sotorasib" OR "adagrasib" OR "AMG-510" OR "MRTX849") AND ("resistance" OR "resistant") AND ("RNA-seq" OR "transcriptome")',
    sources="pubmed,biorxiv",
    max_results=5
)

# Step 2: Dataset search including cell lines and PDX models
search_datasets_directly(
    query='("KRAS G12C") AND ("lung cancer" OR "NSCLC" OR "LUAD") AND ("resistant" OR "resistance" OR "sensitive") AND ("RNA-seq")',
    data_type="geo",
    filters='{{"organisms": ["human"], "entry_types": ["gse"], "date_range": {{"start": "2022/01/01", "end": "2025/01/01"}}}}'
)

# Step 3: Validate metadata - MUST contain:
# - KRAS mutation status (specifically G12C)
# - Treatment sensitivity data (IC50, resistant/sensitive classification)
# - Time series if studying acquired resistance

Expected Output Format:

✅ GSE184299 (2022) - PERFECT MATCH
- Model: H358 NSCLC cells (KRAS G12C)
- Conditions: Parental vs Sotorasib-resistant clones
- Samples: 6 sensitive, 6 resistant (triplicates)
- Resistance level: 100-fold increase in IC50
- Technology: RNA-seq with 30M reads/sample
- Key finding: MET amplification in resistant clones

Example 3: CDK4/6 Inhibitor Combination for Breast Cancer

Pharma Context: "We're testing CDK4/6 inhibitor combinations. I need breast cancer datasets with palbociclib/ribociclib treatment showing single-cell immune profiling to understand immune modulation."

Optimized Search Strategy:

# Step 1: Search for CDK4/6 inhibitor studies with immune profiling
search_literature(
    query='("CDK4/6 inhibitor" OR "palbociclib" OR "ribociclib" OR "abemaciclib") AND ("breast cancer") AND ("single-cell" OR "scRNA-seq" OR "CyTOF") AND ("immune" OR "tumor microenvironment" OR "TME")',
    sources="pubmed",
    max_results=5
)

# Step 2: Dataset search focusing on treatment and immune cells
search_datasets_directly(
    query='("breast cancer") AND ("palbociclib" OR "ribociclib" OR "CDK4") AND ("single-cell" OR "scRNA-seq") AND ("immune" OR "T cell" OR "macrophage")',
    data_type="geo",
    filters='{{"organisms": ["human"], "entry_types": ["gse"], "supplementary_file_types": ["h5ad", "h5"]}}'
)

# Step 3: Validate metadata - MUST contain:
# - ER/PR/HER2 status
# - CDK4/6 inhibitor treatment details
# - Immune cell annotations

Example 4: Hepatotoxicity Biomarkers for Novel TYK2 Inhibitor

Pharma Context: "Our TYK2 inhibitor showed unexpected hepatotoxicity in phase 1. I need human liver datasets (healthy vs drug-induced liver injury) to identify predictive toxicity signatures."

Optimized Search Strategy:

# Step 1: Search for drug-induced liver injury datasets
search_literature(
    query='("drug-induced liver injury" OR "DILI" OR "hepatotoxicity") AND ("RNA-seq" OR "transcriptomics") AND ("human") AND ("biomarker" OR "signature" OR "prediction")',
    sources="pubmed",
    max_results=5,
    filters='{{"date_range": {{"start": "2018", "end": "2024"}}}}'
)

# Step 2: Direct search for liver datasets with toxicity
search_datasets_directly(
    query='("liver" OR "hepatocyte" OR "hepatic") AND ("toxicity" OR "DILI" OR "drug-induced") AND ("RNA-seq") AND ("human")',
    data_type="geo",
    filters='{{"organisms": ["human"], "entry_types": ["gse"]}}'
)

# Step 3: Also search for TYK2/JAK pathway in liver
search_datasets_directly(
    query='("TYK2" OR "JAK" OR "STAT") AND ("liver" OR "hepatocyte") AND ("inhibitor" OR "knockout") AND ("RNA-seq")',
    data_type="geo",
    filters='{{"organisms": ["human", "mouse"], "entry_types": ["gse"]}}'
)

Example 5: CAR-T Cell Exhaustion in Solid Tumors

Pharma Context: "Our CD19 CAR-T works in lymphoma but fails in solid tumors. I need single-cell datasets comparing CAR-T cells from responders vs non-responders to understand exhaustion mechanisms."

Optimized Search Strategy:

# ... (CAR-T search strategy)

Example 6: Competitive Intelligence - Extract Competitor's Methods

Pharma Context: "Our competitor just published a Nature paper on their single-cell analysis pipeline. I need to know exactly what methods, parameters, and software they used so we can replicate or improve upon their approach."

Optimized Search Strategy:

# Step 1: Find the paper using literature search
search_literature(
    query='competitor_name AND "single-cell" AND "analysis pipeline"',
    sources="pubmed",
    max_results=3
)

# Step 2: Extract computational methods from the PDF
extract_paper_methods("https://www.nature.com/articles/competitor-paper.pdf")

# Expected Output:
{{
  "software_used": ["Scanpy 1.9", "Seurat 4.0", "CellTypist"],
  "parameters": {{
    "min_genes": "200",
    "min_cells": "3",
    "max_percent_mito": "5%",
    "n_neighbors": "15",
    "resolution": "0.5"
  }},
  "statistical_methods": ["Wilcoxon rank-sum test", "Benjamini-Hochberg FDR"],
  "normalization_methods": ["log1p transformation", "total-count normalization"],
  "quality_control": ["doublet detection with Scrublet", "cell cycle regression"]
}}

# Step 3: Check full publication content for supplementary material references
get_publication_overview("10.1038/s41586-2024-12345-6")
# Returns: Full paper content with references to supplementary materials
# Note: Supplementary downloads temporarily disabled - check publisher website manually

Use Cases:
✅ Replicate competitor methods exactly
✅ Identify gaps in competitor's QC pipeline
✅ Extract parameter values for optimization
✅ Review methods and supplementary material references
✅ Due diligence for acquisition targets

Example 7: Method Extraction for Protocol Standardization

Pharma Context: "We're standardizing our single-cell analysis pipeline. I need to extract methods from 5 top papers to identify consensus best practices."

Optimized Search Strategy:

# Step 1: CAR-T specific literature search
search_literature(
    query='("CAR-T" OR "chimeric antigen receptor") AND ("exhaustion" OR "dysfunction" OR "failure") AND ("single-cell" OR "scRNA-seq") AND ("solid tumor" OR "responder")',
    sources="pubmed,biorxiv",
    max_results=5
)

# Step 2: Dataset search for CAR-T profiling
search_datasets_directly(
    query='("CAR-T" OR "CAR T cell" OR "chimeric antigen receptor") AND ("single-cell RNA-seq" OR "scRNA-seq") AND ("patient" OR "clinical")',
    data_type="geo",
    filters='{{"organisms": ["human"], "entry_types": ["gse"], "date_range": {{"start": "2021/01/01", "end": "2025/01/01"}}}}'
)

# Step 3: Validate metadata - MUST contain:
# - CAR construct details (CD19, CD22, etc.)
# - Clinical response data
# - Time points (pre-infusion, peak expansion, relapse)
# - T cell phenotype annotations

Expected Output Format:

✅ GSE197215 (2023) - PERFECT MATCH
- Disease: B-ALL and DLBCL
- CAR type: CD19-BBz
- Samples: 12 responders, 8 non-responders
- Timepoints: Pre-infusion, Day 7, Day 14, Day 28
- Cell count: 50,000 CAR-T cells profiled
- Key metadata: Complete response duration, CAR persistence
- Finding: TOX expression correlates with non-response

</Pharmaceutical_Research_Examples>

<Common_Pitfalls_To_Avoid>

    Generic queries: "cancer RNA-seq" → Too broad, specify cancer type and comparison
    Missing treatment details: Always include drug names (generic AND brand)
    Ignoring model systems: Include cell lines, PDX, organoids when relevant
    Forgetting resistance mechanisms: For oncology, always consider resistant vs sensitive
    Neglecting timepoints: For treatment studies, pre/post or time series are crucial
    Missing clinical annotations: Response criteria (RECIST, VGPR, etc.) are essential </Common_Pitfalls_To_Avoid>

<Response_Template>
Dataset Discovery Results for [Drug Target/Indication]
✅ Datasets Meeting ALL Criteria

    [GSE_NUMBER] (Year: XXXX) - [MATCH QUALITY]
        Disease/Model: [Specific type]
        Treatment: [Drug name, dose, schedule]
        Samples: [N with breakdown by group]
        Key metadata: [Response, mutations, clinical outcomes]
        Cell/Read count: [Technical details]
        Data format: [Available formats]
        Key finding: [Relevant to drug development]
        Link: [Direct GEO link]
        PMID: [Associated publication]

🔬 Recommended Analysis Strategy

[Specific to the drug discovery question - e.g., "Compare responder vs non-responder T cells for exhaustion markers"]
⚠️ Data Limitations

[Missing metadata, small sample size, etc.]
💊 Drug Development Relevance

[How this dataset can inform the drug program] </Response_Template>

**Note**: For stop conditions and operational limits, refer to the "Operational Limits" section in Critical_Rules above.

""".format(
        date=date.today()
    )
    return create_react_agent(
        model=llm, tools=tools, prompt=system_prompt, name=agent_name
    )
