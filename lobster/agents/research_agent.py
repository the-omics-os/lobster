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
from lobster.tools.providers.base_provider import DatasetType, PublicationSource
from lobster.tools.publication_intelligence_service import (
    PublicationIntelligenceService,
)
from lobster.tools.publication_service import PublicationService
# Phase 1: New providers for two-tier access
from lobster.tools.providers.abstract_provider import AbstractProvider
from lobster.tools.providers.webpage_provider import WebpageProvider
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

    # Initialize publication service with NCBI API key
    publication_service = PublicationService(data_manager=data_manager)

    # Initialize research agent assistant for metadata validation
    research_assistant = ResearchAgentAssistant()

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

            results = publication_service.search_literature(
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
            metadata = publication_service.extract_publication_metadata(identifier)

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

            results = publication_service.search_literature(
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

            results = publication_service.find_datasets_from_publication(
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

            results = publication_service.search_literature(
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

            results = publication_service.search_datasets_directly(
                query=query,
                data_type=dataset_type,
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

            metadata = publication_service.extract_publication_metadata(
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
            return publication_service.get_provider_capabilities()
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

                    # Use the assistant to validate metadata
                    validation_result = research_assistant.validate_dataset_metadata(
                        metadata=metadata,
                        geo_id=accession,
                        required_fields=fields_list,
                        required_values=values_dict,
                        threshold=threshold,
                    )

                    if validation_result:
                        # Format the validation report
                        report = research_assistant.format_validation_report(
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

        This tool downloads and analyzes research papers to extract:
        - Software/tools used (e.g., Scanpy, Seurat, DESeq2)
        - Parameter values and cutoffs
        - Statistical methods
        - Data sources and sample sizes
        - Normalization and QC methods

        Accepts multiple identifier types:
        - PMID (e.g., "PMID:12345678" or "12345678") - Auto-resolves via PMC/bioRxiv
        - DOI (e.g., "10.1038/s41586-021-12345-6") - Auto-resolves to open access PDF
        - Direct PDF URL (e.g., https://nature.com/articles/paper.pdf)
        - Webpage URL (will auto-detect PDF link)

        Resolution priority: PMC → bioRxiv/medRxiv → Publisher Open Access

        Args:
            url_or_pmid: PMID, DOI, or PDF URL

        Returns:
            JSON-formatted extraction of methods, parameters, and software used
            OR helpful suggestions if paper is paywalled

        Examples:
            - extract_paper_methods("PMID:12345678")
            - extract_paper_methods("10.1038/s41586-021-12345-6")
            - extract_paper_methods("https://www.biorxiv.org/content/10.1101/2024.01.001.pdf")
            - extract_paper_methods("https://elifesciences.org/articles/12345")
        """
        try:
            # Initialize intelligence service
            intelligence_service = PublicationIntelligenceService(
                data_manager=data_manager
            )

            # Extract methods using LLM with automatic identifier resolution
            methods = intelligence_service.extract_methods_from_paper(url_or_pmid)

            # Format for agent response
            formatted = json.dumps(methods, indent=2)
            logger.info(
                f"Successfully extracted methods from paper: {url_or_pmid[:80]}..."
            )

            return f"## Extracted Methods from Paper\n\n{formatted}"

        except Exception as e:
            logger.error(f"Error extracting paper methods: {e}")
            error_msg = str(e)

            # Check if it's a paywalled paper with suggestions
            if "not openly accessible" in error_msg:
                return f"## Paper Access Issue\n\n{error_msg}"
            else:
                return f"Error extracting methods from paper: {error_msg}"

    @tool
    def resolve_paper_access(identifier: str) -> str:
        """
        Check if a paper is accessible and get PDF URL or access suggestions.

        Use this tool before extract_paper_methods to:
        - Verify paper accessibility
        - Get direct PDF URL
        - Receive guidance if paywalled (PMC links, preprints, author contact)

        Resolution Strategy:
        1. PubMed Central (PMC) - Free full text
        2. bioRxiv/medRxiv - Preprint servers
        3. Publisher Open Access
        4. Helpful suggestions if paywalled

        Args:
            identifier: PMID (e.g., "PMID:12345678"), DOI, or paper identifier

        Returns:
            Access report with PDF URL OR alternative suggestions

        Examples:
            - resolve_paper_access("PMID:12345678")
            - resolve_paper_access("10.1038/s41586-021-12345-6")

        When to use this tool:
        - Before calling extract_paper_methods to check accessibility
        - When user asks "Can I access this paper?"
        - To diagnose why PDF extraction failed
        """
        try:
            # Use research assistant to resolve
            result = research_assistant.resolve_publication_to_pdf(identifier)

            # Format the result
            report = research_assistant.format_resolution_report(result)

            logger.info(f"Resolved access for {identifier}: {result.access_type}")

            return report

        except Exception as e:
            logger.error(f"Error resolving paper access: {e}")
            return f"Error checking paper access: {str(e)}"

    @tool
    def extract_methods_batch(identifiers: str, max_papers: int = 5) -> str:
        """
        Extract computational methods from multiple papers in batch.

        This tool:
        1. Accepts comma-separated PMIDs, DOIs, or URLs
        2. Resolves identifiers to PDFs automatically
        3. Extracts methods from each paper
        4. Returns aggregated results with success/failure report
        5. Conservative limit: 5 papers per batch (configurable up to 10)

        Args:
            identifiers: Comma-separated list (e.g., "PMID:12345,10.1038/s41586-021-12345-6")
            max_papers: Maximum papers to process (default: 5, max: 10)

        Returns:
            Batch extraction report with individual results and summary

        Examples:
            - extract_methods_batch("PMID:12345678,PMID:87654321,10.1038/s41586-021-12345-6")
            - extract_methods_batch("https://biorxiv.org/paper1.pdf,PMID:12345", max_papers=3)

        When to use this tool:
        - User asks to "analyze methods from these 5 papers"
        - Competitive intelligence workflows
        - Literature review with method comparison
        - When user provides a list of PMIDs/DOIs

        Note: This tool processes papers sequentially to be conservative.
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

            # First, resolve all identifiers to check accessibility
            resolution_results = research_assistant.batch_resolve_publications(
                id_list, max_batch=max_papers
            )

            # Track results
            successful_extractions = []
            paywalled_papers = []
            failed_extractions = []

            # Initialize intelligence service
            intelligence_service = PublicationIntelligenceService(
                data_manager=data_manager
            )

            # Process each paper
            for i, (identifier, resolution) in enumerate(
                zip(id_list, resolution_results), 1
            ):
                logger.info(f"Processing paper {i}/{len(id_list)}: {identifier}")

                if not resolution.is_accessible():
                    # Paper is paywalled or inaccessible
                    paywalled_papers.append(
                        {
                            "identifier": identifier,
                            "reason": resolution.access_type,
                            "suggestions": resolution.suggestions,
                        }
                    )
                    continue

                try:
                    # Extract methods
                    methods = intelligence_service.extract_methods_from_paper(
                        resolution.pdf_url
                    )
                    successful_extractions.append(
                        {
                            "identifier": identifier,
                            "source": resolution.source,
                            "methods": methods,
                        }
                    )
                    logger.info(f"✅ Successfully extracted methods from {identifier}")

                except Exception as e:
                    logger.error(f"❌ Failed to extract methods from {identifier}: {e}")
                    failed_extractions.append(
                        {"identifier": identifier, "error": str(e)}
                    )

            # Generate comprehensive report
            report = f"""
## Batch Method Extraction Report

**Total Papers:** {len(id_list)}
**Successful:** ✅ {len(successful_extractions)} ({len(successful_extractions)*100//len(id_list) if id_list else 0}%)
**Paywalled:** ❌ {len(paywalled_papers)} ({len(paywalled_papers)*100//len(id_list) if id_list else 0}%)
**Failed:** ⚠️ {len(failed_extractions)} ({len(failed_extractions)*100//len(id_list) if id_list else 0}%)

---

### ✅ Successfully Extracted ({len(successful_extractions)}):
"""

            for result in successful_extractions:
                report += f"\n**{result['identifier']}** (Source: {result['source']})\n"
                report += f"```json\n{json.dumps(result['methods'], indent=2)}\n```\n\n"

            if paywalled_papers:
                report += f"\n### ❌ Paywalled Papers ({len(paywalled_papers)}):\n"
                for paper in paywalled_papers:
                    report += f"\n**{paper['identifier']}**\n"
                    report += f"- Status: {paper['reason']}\n"
                    report += f"- Suggestions:\n{paper['suggestions']}\n\n"

            if failed_extractions:
                report += f"\n### ⚠️ Failed Extractions ({len(failed_extractions)}):\n"
                for paper in failed_extractions:
                    report += f"\n**{paper['identifier']}**\n"
                    report += f"- Error: {paper['error']}\n\n"

            logger.info(
                f"Batch extraction complete: {len(successful_extractions)}/{len(id_list)} successful"
            )

            return report

        except Exception as e:
            logger.error(f"Error in batch extraction: {e}")
            return f"Error in batch method extraction: {str(e)}"

    @tool
    def download_supplementary_materials(doi: str, output_dir: str = None) -> str:
        """
        Download supplementary materials from a paper's DOI.

        This tool:
        - Resolves DOI to publisher page
        - Finds supplementary material links
        - Downloads all supplementary files
        - Returns download report with file locations

        Args:
            doi: Paper DOI (e.g., "10.1038/s41586-021-12345-6")
            output_dir: Directory to save files (default: .lobster_workspace/supplements/<doi>)

        Returns:
            Download report with list of downloaded files

        Examples:
            - download_supplementary_materials("10.1038/s41586-021-12345-6")
            - download_supplementary_materials("10.1126/science.abc1234", "/path/to/output")
        """
        try:
            # Initialize intelligence service
            intelligence_service = PublicationIntelligenceService(
                data_manager=data_manager
            )

            # Download supplementary materials
            result = intelligence_service.fetch_supplementary_info_from_doi(
                doi, output_dir
            )

            logger.info(f"Supplementary download completed for DOI: {doi}")
            return f"## Supplementary Materials Download Report\n\n{result}"

        except Exception as e:
            logger.error(f"Error downloading supplementary materials: {e}")
            return f"Error downloading supplementary materials for DOI {doi}: {str(e)}"

    @tool
    def read_cached_publication(identifier: str) -> str:
        """
        Read detailed methods from a previously analyzed publication.

        This tool retrieves the full methods extraction from publications that were
        analyzed earlier in the current session. Use this when the supervisor
        references a specific paper from the session publication list.

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
            # Initialize intelligence service
            intelligence_service = PublicationIntelligenceService(
                data_manager=data_manager
            )

            # Get cached publication
            cached_pub = intelligence_service.get_cached_publication(identifier)

            if not cached_pub:
                return f"## Publication Not Found\n\nNo cached extraction found for: {identifier}\n\nThis publication has not been analyzed in the current session. Use list_session_publications (via supervisor) to see available publications, or use extract_paper_methods to analyze a new paper."

            # Format the cached publication for display
            response = f"## Cached Publication: {cached_pub['identifier']}\n\n"
            response += f"**Cache Source**: {cached_pub.get('cache_source', 'unknown')}\n"

            # Add methods section
            methods_text = cached_pub.get('methods_markdown') or cached_pub.get('methods_text', '')
            if methods_text:
                response += "\n### Methods Section\n\n"
                response += methods_text[:5000]  # Limit to 5000 chars for readability
                if len(methods_text) > 5000:
                    response += f"\n\n... [Methods section truncated, showing first 5000 of {len(methods_text)} characters]"

            # Add tables if present
            tables = cached_pub.get('tables', [])
            if tables and isinstance(tables, list) and len(tables) > 0:
                response += f"\n\n### Extracted Tables ({len(tables)})\n\n"
                for i, table in enumerate(tables[:3], 1):  # Show first 3 tables
                    response += f"**Table {i}**: [Table data available]\n"
                if len(tables) > 3:
                    response += f"\n... [Showing 3 of {len(tables)} tables]\n"

            # Add formulas if present
            formulas = cached_pub.get('formulas', [])
            if formulas and isinstance(formulas, list) and len(formulas) > 0:
                response += f"\n\n### Extracted Formulas ({len(formulas)})\n\n"
                for i, formula in enumerate(formulas[:5], 1):  # Show first 5 formulas
                    response += f"**Formula {i}**: `{formula}`\n"
                if len(formulas) > 5:
                    response += f"\n... [Showing 5 of {len(formulas)} formulas]\n"

            # Add software mentions
            software = cached_pub.get('software_mentioned', [])
            if software and isinstance(software, list) and len(software) > 0:
                response += f"\n\n### Software Tools Detected\n\n"
                response += ", ".join(f"`{sw}`" for sw in software)

            # Add provenance metadata
            provenance = cached_pub.get('provenance', {})
            if provenance:
                response += "\n\n### Extraction Metadata\n\n"
                response += f"- **Parser**: {provenance.get('parser', 'unknown')}\n"
                response += f"- **Fallback Used**: {provenance.get('fallback_used', False)}\n"
                if provenance.get('timestamp'):
                    response += f"- **Timestamp**: {provenance.get('timestamp')}\n"

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

            logger.info(f"Successfully retrieved abstract: {len(metadata.abstract)} chars")
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
        Extract full publication content with webpage-first strategy.

        This is the DEEP PATH for two-tier access strategy:
        - First attempts: Webpage extraction (Nature, Science, etc.) - 2-5 seconds
        - Fallback: PDF parsing with Docling - 3-8 seconds

        Use this tool when:
        - User needs full content, not just abstract
        - Extracting Methods section for replication
        - User asks for "parameters", "software used", "methods"
        - After checking relevance with get_quick_abstract()

        Webpage-First Strategy (NEW in Phase 1):
        - Tries publisher webpage before PDF (e.g., Nature articles)
        - Faster and more reliable for many publishers
        - Automatically falls back to PDF if webpage extraction fails

        Supported Identifiers:
        - PMID: "PMID:12345678" (auto-resolves to accessible source)
        - DOI: "10.1038/s41586-021-12345-6" (auto-resolves)
        - Direct URL: "https://www.nature.com/articles/s41586-025-09686-5"
        - PDF URL: "https://biorxiv.org/content/10.1101/2024.01.001.pdf"

        Args:
            identifier: PMID, DOI, or URL
            prefer_webpage: Try webpage before PDF (default: True)

        Returns:
            Full content markdown with sections, tables, and metadata

        Examples:
            - get_publication_overview("PMID:35042229")
            - get_publication_overview("https://www.nature.com/articles/s41586-025-09686-5")
            - get_publication_overview("10.1038/...", prefer_webpage=False)  # Force PDF

        Performance: 2-8 seconds depending on source and length
        """
        try:
            logger.info(f"Getting publication overview for: {identifier}")

            # Check if identifier is a direct webpage URL
            is_webpage_url = identifier.startswith("http") and not identifier.lower().endswith(".pdf")

            if is_webpage_url and prefer_webpage:
                # Try webpage extraction first
                try:
                    logger.info(f"Attempting webpage extraction for: {identifier[:80]}...")
                    webpage_provider = WebpageProvider(data_manager=data_manager)

                    # Extract full content
                    markdown_content = webpage_provider.extract(identifier, max_paragraphs=100)

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

                    logger.info(f"Successfully extracted webpage: {len(markdown_content)} chars")
                    return response

                except Exception as webpage_error:
                    logger.warning(f"Webpage extraction failed, trying PDF fallback: {webpage_error}")
                    # Fall through to PDF extraction below

            # Fallback to PDF extraction (original behavior)
            logger.info(f"Using PDF extraction for: {identifier}")
            intelligence_service = PublicationIntelligenceService(data_manager=data_manager)

            # Extract methods using LLM
            methods = intelligence_service.extract_methods_from_paper(identifier)

            # Format response
            formatted = json.dumps(methods, indent=2)

            response = f"""## Publication Overview (PDF Extraction)

**Source:** {identifier}
**Extraction Method:** PDF parsing with Docling

### Extracted Information

{formatted}

---
*Extracted from PDF using structure-aware parsing*
*For abstract-only view, use get_quick_abstract()*
"""

            logger.info(f"Successfully extracted PDF methods")
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

    # ============================================================
    # DEPRECATED TOOLS - Phase 1 Refactoring (2025-01-02)
    # ============================================================
    # These tools are being replaced with cleaner architecture:
    # - extract_paper_methods -> get_publication_overview
    # - resolve_paper_access -> get_publication_overview (implicit resolution)
    # - extract_methods_batch -> Use get_publication_overview in loop at agent level
    #
    # Status: Commented out, kept for reference
    # Removal: Phase 4 (after full migration)
    # ============================================================

    # @tool  # DEPRECATED - DO NOT USE
    # def extract_paper_methods(url_or_pmid: str) -> str:
    #     """
    #     DEPRECATED: Use get_publication_overview() instead.
    #
    #     This tool is being replaced in Phase 1 with cleaner two-tier architecture.
    #     """
    #     # Implementation kept for reference during migration
    #     pass

    # @tool  # DEPRECATED - DO NOT USE
    # def resolve_paper_access(identifier: str) -> str:
    #     """
    #     DEPRECATED: Use get_publication_overview() instead.
    #
    #     Resolution is now implicit in get_publication_overview() with
    #     automatic webpage-first strategy.
    #     """
    #     # Implementation kept for reference during migration
    #     pass

    # @tool  # DEPRECATED - DO NOT USE
    # def extract_methods_batch(identifiers: str, max_papers: int = 5) -> str:
    #     """
    #     DEPRECATED: Removed in Phase 1.
    #
    #     For batch processing, use get_publication_overview() in a loop
    #     at the agent level instead of tool level.
    #     """
    #     # Implementation kept for reference during migration
    #     pass

    base_tools = [
        search_literature,
        find_datasets_from_publication,
        find_marker_genes,
        discover_related_studies,
        search_datasets_directly,
        extract_publication_metadata,
        get_research_capabilities,
        validate_dataset_metadata,
        # Phase 1 NEW TOOLS: Two-tier access
        get_quick_abstract,
        get_publication_overview,
        # Other tools
        download_supplementary_materials,
        # Session publication access
        read_cached_publication,
        # DEPRECATED tools (commented out above):
        # extract_paper_methods,  # Phase 1 deprecated
        # resolve_paper_access,   # Phase 1 deprecated
        # extract_methods_batch,  # Phase 1 deprecated
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

2. **USE CORRECT ACCESSIONS**: 
   - Accept both GSE (Series) and GDS (DataSet) accessions - GDS identifiers are automatically converted to their corresponding GSE
   - Both modern RNA-seq and legacy array data are accessible through either format
   - Validate accessions before reporting them to ensure they exist

3. **VERIFY METADATA EARLY**: 
   - IMMEDIATELY check if datasets contain required metadata (e.g., treatment response, mutation status, clinical outcomes)
   - Discard datasets lacking critical annotations to avoid dead ends
   - Parse sample metadata files (SOFT, metadata.tsv) for required variables

4. **STOP WHEN SUCCESSFUL**: 
   - After finding 1-3 suitable datasets meeting ALL criteria, STOP and report to supervisor
   - Do not continue searching indefinitely
   - Maximum 10-15 search attempts before requesting guidance

5. **PROVIDE ACTIONABLE SUMMARIES**: 
   - Each dataset must include: Accession, Year, Sample count, Metadata categories, Data availability
   - Create concise ranked shortlist, not verbose logs
   - Lead with results, append details only if needed
</Critical_Rules>

<Query_Optimization_Strategy>
## Before searching, ALWAYS:
1. **Define mandatory criteria**:
   - Technology type (e.g., single-cell RNA-seq, CRISPR screen, proteomics)
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
  * filters: JSON string for date ranges, authors, journals, publication types
  * max_results: 3-6 for comprehensive surveys

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

- `download_supplementary_materials`: Download supplementary files from DOI
  * Provide paper DOI (e.g., "10.1038/s41586-021-12345-6")
  * Automatically finds and downloads all supplementary materials
  * Saves to .lobster_workspace/supplements/<doi>/
  * Returns download report with file locations
  * Useful for accessing protocols, code, raw data
  * Example: download_supplementary_materials("10.1038/s41586-021-12345-6")

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
  * Advanced GEO filters: organisms, platforms, entry types, date ranges, supplementary files
  * Filters example: '{{"organisms": ["human"], "entry_types": ["gse"], "date_range": {{"start": "2015/01/01", "end": "2025/01/01"}}}}'
  * Check for processed data availability (h5ad, loom, CSV counts)

### Biological Discovery
- `find_marker_genes`: Literature-based marker gene identification
  * cell_type parameter required
  * Optional disease context
  * Cross-references multiple studies for consensus markers

### Metadata Validation
- `validate_dataset_metadata`: Quick metadata validation without downloading
  * required_fields: comma-separated list (e.g., "smoking_status,treatment_response")
  * required_values: JSON string of field->values mapping
  * threshold: minimum fraction of samples with required fields (default: 0.8)
  * Returns recommendation: "proceed" | "skip" | "manual_check"
  * Example: validate_dataset_metadata("GSE179994", "treatment_response,timepoint", '{{"treatment_response": ["responder", "non-responder"]}}')
</Available Research Tools>

<Critical_Tool_Usage_Workflows>
## PDF Access and Method Extraction

The system supports automatic publication resolution through PMC → bioRxiv → Publisher Open Access pathways.

### Workflow 1: Extract Methods from Literature Search Results

**Scenario**: User asks "Find papers on KRAS G12C resistance and extract their methods"

**Workflow**:
```
Step 1: Search for papers
search_literature("KRAS G12C resistance mechanisms", max_results=5)
→ Returns 5 papers with PMIDs

Step 2: For EACH PMID, directly extract methods (automatic resolution occurs internally)
extract_paper_methods("PMID:12345678")
extract_paper_methods("PMID:23456789")
...

Step 3: If extraction fails with "paywalled" message:
- Present the alternative access suggestions provided
- Do NOT give up - the suggestions include PMC links, preprints, author contact
- Ask user if they want to try alternative sources
```

### Workflow 2: Batch Method Extraction

**Scenario**: User asks "Extract methods from these 5 papers: PMID:123, PMID:456, PMID:789, DOI:10.1038/..., DOI:10.1016/..."

**Workflow**:
```
Step 1: Use batch extraction tool
extract_methods_batch("PMID:123,PMID:456,PMID:789,10.1038/...,10.1016/...", max_papers=5)

Step 2: Review the batch report:
- ✅ Successfully extracted papers → Present methods
- ❌ Paywalled papers → Present alternative access suggestions
- ⚠️ Failed papers → Explain what went wrong

Step 3: For paywalled papers, offer to help:
"Papers X and Y are paywalled. Would you like me to:
1. Check alternative sources (PMC, bioRxiv)?
2. Provide author contact information?
3. Continue with the accessible papers only?"
```

**When to use batch vs individual**:
- Batch (extract_methods_batch): User provides 2-5 papers at once
- Individual (extract_paper_methods): One paper at a time, or iterative workflow

### Workflow 3: Check Accessibility First

**Scenario**: User uncertain about paper access, or previous extraction failed

**Workflow**:
```
Step 1: Check accessibility with diagnostic tool
resolve_paper_access("PMID:12345678")

Step 2: Interpret the result:
- If ✅ ACCESSIBLE: Shows PDF URL and source (PMC, bioRxiv, etc.)
  → Proceed with extract_paper_methods("PMID:12345678")

- If ❌ NOT ACCESSIBLE: Shows alternative access options
  → Present suggestions to user
  → Ask if they want to try alternatives
  → Do NOT attempt extraction (will fail)

Step 3: If user wants alternatives:
- Check PMC accepted manuscript link
- Search bioRxiv/medRxiv for preprints
- Suggest contacting corresponding author
```

**When to use resolve_paper_access**:
- User asks "Can I access this paper?"
- Previous extract_paper_methods failed
- Diagnosing accessibility issues
- Before batch processing to preview accessibility

### Workflow 4: Handle Paywalled Papers Gracefully

**Scenario**: Paper is not openly accessible

**Workflow**:
```
Step 1: When extract_paper_methods returns "Paper Access Issue":
- Read the suggestions carefully
- Present ALL alternative options to user
- Do NOT say "I cannot access this paper" and stop

Step 2: Present structured alternatives:
"This paper is paywalled at the publisher, but here are alternatives:

1. PubMed Central: Check for accepted manuscript
   - PMC search link

2. Preprint Servers: May have early version
   - bioRxiv search
   - medRxiv search

3. Institutional Access: Try through your library
   - Use VPN or library proxy
   - Request via interlibrary loan

4. Author Contact: Request PDF directly
   - Email corresponding author
   - Check ResearchGate/Academia.edu profiles

5. Unpaywall: Legal open access checker
   - Check unpaywall.org

Would you like me to try any of these alternatives?"

Step 3: If user provides alternative URL:
extract_paper_methods("[alternative URL]")
```

**Important**: Do not stop at "I cannot access this paper". Always present the 5 alternative access options.

### Workflow 5: Competitive Intelligence Analysis

**Scenario**: "Analyze competitor's methods from their 5 recent papers"

**Workflow**:
```
Step 1: Search for competitor's recent papers
search_literature(
    "competitor_name AND (2023[PDAT] OR 2024[PDAT] OR 2025[PDAT])",
    max_results=5,
    sources="pubmed",
    filters='{{"date_range": {{"start": "2023", "end": "2025"}}}}'
)

Step 2: Extract PMIDs from results

Step 3: Use batch extraction for efficiency
extract_methods_batch("PMID:123,PMID:456,PMID:789,PMID:012,PMID:345")

Step 4: Analyze the successfully extracted methods:
- Common software/tools across papers
- Parameter consistency or evolution
- Statistical approaches used
- Trends over time (2023 → 2024 → 2025)

Step 5: Present comparative analysis:
"## Competitor Method Analysis

**Papers Analyzed**: 5 (3 accessible, 2 paywalled)

**Common Tools**:
- Scanpy: Used in 3/3 papers
- Seurat: Used in 2/3 papers
- DESeq2: Used in 1/3 papers

**Parameter Patterns**:
- min_genes: 200 (consistent across papers)
- resolution: 0.5 → 0.8 (increased over time)

**Statistical Methods**:
- Wilcoxon test: Standard approach
- FDR correction: Always applied

**Trends**: Moving from Seurat to Scanpy in recent work"
```

### Tool Selection Decision Tree

**Question**: Which tool should I use for this user request?

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

### Error Recovery Strategies

#### Problem 1: "Paper not openly accessible"

**Recovery Steps**:
1. Read the suggestions in the error message
2. Present ALL 5 alternative options to user
3. Offer to check PMC accepted manuscript
4. Offer to search bioRxiv/medRxiv
5. Suggest author contact information
6. Do not stop at "cannot access"

#### Problem 2: Batch processing has failures

**Recovery Steps**:
1. Review extract_methods_batch() report
2. For each failure, identify reason:
   - Paywalled → Apply recovery for Problem 1
   - Network error → Offer to retry
   - Invalid identifier → Ask user to verify
3. Present partial results: "Successfully extracted 3/5 papers"
4. Offer to retry failed papers individually
5. Aggregate successful results and continue analysis

#### Problem 3: PMID resolution failed

**Recovery Steps**:
1. Check if recent publication (PMC lag 6-12 months)
2. Try preprint servers (bioRxiv/medRxiv)
3. Search publisher page for open access
4. Suggest checking back later
5. Offer to work with abstract/methods from PubMed

#### Problem 4: All papers in batch are paywalled

**Recovery Steps**:
1. Present all alternative access options
2. Suggest narrowing search to open-access journals
3. Offer to search bioRxiv/medRxiv directly
4. Recommend institutional access methods
5. Do not say "all papers are inaccessible, cannot proceed"

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

# Step 3: Download supplementary materials for code/protocols
download_supplementary_materials("10.1038/s41586-2024-12345-6")

# Downloads: analysis_code.zip, supplementary_tables.xlsx, protocol.pdf

Use Cases:
✅ Replicate competitor methods exactly
✅ Identify gaps in competitor's QC pipeline
✅ Extract parameter values for optimization
✅ Access supplementary code and protocols
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

<Stop_Conditions>

    ✅ Found 1-3 datasets with required treatment/control comparison → STOP and report
    ⚠️ 10+ search attempts without success → Suggest alternative approaches (cell lines, mouse models)
    ❌ No datasets with required clinical metadata → Recommend generating new data
    🔄 Same results repeating → Expand to related drugs in class or earlier timepoints </Stop_Conditions>


""".format(
        date=date.today()
    )
    return create_react_agent(
        model=llm, tools=tools, prompt=system_prompt, name=agent_name
    )
