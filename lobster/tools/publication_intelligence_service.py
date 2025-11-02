"""
Publication Intelligence Service for PDF extraction and method analysis.

Portions of this code adapted from Biomni (https://github.com/snap-stanford/Biomni)
Copyright (c) 2025 Stanford SNAP Lab
Licensed under Apache License 2.0

Enhancements for Lobster:
- Enterprise caching and error handling
- DataManagerV2 provenance integration
- Enhanced logging and monitoring
- LLM-based method extraction
- Professional naming conventions
- Structure-aware PDF parsing with Docling (v2.3.0+)
"""

import gc
import hashlib
import json
import os
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import PyPDF2
import requests
from bs4 import BeautifulSoup

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


# Custom Exception Classes
class PublicationServiceError(Exception):
    """Base exception for publication intelligence service."""

    pass


class DoclingError(PublicationServiceError):
    """Docling-specific errors."""

    pass


class PDFExtractionError(PublicationServiceError):
    """General PDF extraction errors."""

    pass


class MethodsSectionNotFoundError(PublicationServiceError):
    """Methods section could not be located."""

    pass


class PublicationIntelligenceService:
    """
    Extract methods and intelligence from scientific publications.

    **NEW in v2.3.0:** Structure-aware PDF parsing with Docling

    This service provides:
    - **Docling-powered extraction**: Structure-aware Methods section detection
    - **Table extraction**: Parameter tables from Methods sections
    - **Formula detection**: Mathematical formulas and code blocks
    - **Smart image filtering**: Exclude base64 encodings from LLM context
    - **Backward compatibility**: Automatic fallback to PyPDF2
    - **Provenance tracking**: Metadata-only logging for literature mining

    Architecture:
        Primary: Docling for structure-aware parsing
        Fallback: PyPDF2 for reliability
        Caching: Parsed documents cached as JSON

    Performance:
        - Methods extraction: 2-5 seconds per paper
        - Cache hit: <100ms
        - Memory usage: ~500MB (Docling initialization)

    Examples:
        >>> service = PublicationIntelligenceService()
        >>>
        >>> # Extract Methods section with structure
        >>> result = service.extract_methods_section(
        ...     "https://arxiv.org/pdf/2408.09869"
        ... )
        >>> print(result['methods_text'])
        >>> print(f"Found {len(result['tables'])} parameter tables")
        >>>
        >>> # Extract parameters using LLM
        >>> methods = service.extract_methods_from_paper("PMID:12345678")
        >>> print(methods['software_used'])
        ['Scanpy', 'Seurat', 'DESeq2']
    """

    def __init__(self, data_manager: Optional[DataManagerV2] = None):
        """
        Initialize with optional DataManager for provenance tracking.

        Args:
            data_manager: DataManagerV2 instance for logging tool usage
        """
        self.data_manager = data_manager
        self.cache_dir = Path(".lobster_workspace") / "literature_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectory for parsed document caching (Phase 2)
        self.parsed_docs_cache = self.cache_dir / "parsed_docs"
        self.parsed_docs_cache.mkdir(parents=True, exist_ok=True)

        # Initialize Docling converter (lazy loading for optional dependency)
        self.converter = None
        self._initialize_docling()

        logger.info(
            f"Initialized PublicationIntelligenceService with cache: {self.cache_dir}"
        )
        logger.info(f"Parsed documents cache: {self.parsed_docs_cache}")

    def _initialize_docling(self):
        """
        Initialize Docling with graceful degradation.

        If Docling is not installed, logs a warning and falls back to PyPDF2.
        """
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import ConversionStatus, DocumentConverter, PdfFormatOption

            # Store imports for later use
            self._docling_imports = {
                "DocumentConverter": DocumentConverter,
                "ConversionStatus": ConversionStatus,
                "PdfPipelineOptions": PdfPipelineOptions,
                "PdfFormatOption": PdfFormatOption,
                "InputFormat": InputFormat,
            }

            self.converter = self._create_docling_converter()
            logger.info("Initialized Docling converter for structure-aware PDF parsing")
        except ImportError:
            logger.warning(
                "Docling not installed, falling back to PyPDF2. "
                "For structure-aware extraction, install with: pip install docling docling-core"
            )
            self.converter = None
            self._docling_imports = None

    def _create_docling_converter(self):
        """
        Create optimized Docling converter for scientific PDFs.

        Configuration:
        - Table structure extraction: ENABLED (Methods parameters)
        - Code enrichment: ENABLED (code blocks detection)
        - Formula enrichment: ENABLED (equations detection)
        - OCR: DISABLED (initially - add later if needed)
        - VLM: DISABLED (too heavy for initial deployment)

        Returns:
            Configured DocumentConverter instance
        """
        DocumentConverter = self._docling_imports["DocumentConverter"]
        PdfPipelineOptions = self._docling_imports["PdfPipelineOptions"]
        PdfFormatOption = self._docling_imports["PdfFormatOption"]
        InputFormat = self._docling_imports["InputFormat"]

        pdf_options = PdfPipelineOptions()
        pdf_options.do_table_structure = True  # Extract parameter tables
        pdf_options.do_code_enrichment = True  # Detect code blocks
        pdf_options.do_formula_enrichment = True  # Detect equations
        pdf_options.do_ocr = False  # Start lightweight

        return DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
            },
        )

    def extract_pdf_content(self, url: str, use_cache: bool = True) -> str:
        """
        Extract text content from a PDF file given its URL.

        This function intelligently handles:
        - Direct PDF URLs
        - Webpage URLs containing PDF links
        - Content-type validation
        - Caching for performance

        Adapted from Biomni with enhancements for enterprise use.

        Args:
            url: URL of the PDF file or webpage containing PDF
            use_cache: Whether to use cached PDF if available (default: True)

        Returns:
            Extracted text content from the PDF

        Raises:
            ValueError: If URL does not contain valid PDF or extraction fails
            requests.exceptions.RequestException: If download fails

        Examples:
            >>> service = PublicationIntelligenceService()
            >>> text = service.extract_pdf_content("https://example.com/paper.pdf")
            >>> print(f"Extracted {len(text)} characters")
        """
        logger.info(f"Extracting PDF content from: {url[:80]}...")

        try:
            # Check cache first
            cache_key = hashlib.md5(url.encode()).hexdigest()
            cache_file = self.cache_dir / f"{cache_key}.txt"

            if use_cache and cache_file.exists():
                logger.info(f"Using cached PDF text: {cache_file.name}")
                text = cache_file.read_text(encoding="utf-8")

                # Log provenance for cached retrieval (LOBSTER ENHANCEMENT)
                if self.data_manager:
                    self.data_manager.log_tool_usage(
                        tool_name="extract_pdf_content",
                        parameters={
                            "url": url[:100],
                            "use_cache": use_cache,
                            "cache_hit": True,
                        },
                        description=f"PDF extraction from cache: {len(text)} characters",
                    )

                return text

            # Step 1: Smart URL detection
            if not url.lower().endswith(".pdf"):
                logger.info("URL doesn't end with .pdf, searching for PDF links...")
                response = requests.get(
                    url,
                    timeout=30,
                    headers={"User-Agent": "Lobster AI Research Tool/1.0"},
                )
                response.raise_for_status()

                if response.status_code == 200:
                    # Look for PDF links in HTML
                    pdf_links = re.findall(
                        r'href=[\'"]([^\'"]+\.pdf)[\'"]', response.text
                    )
                    if pdf_links:
                        # Handle relative URLs
                        if not pdf_links[0].startswith("http"):
                            base_url = "/".join(url.split("/")[:3])
                            url = (
                                base_url + pdf_links[0]
                                if pdf_links[0].startswith("/")
                                else base_url + "/" + pdf_links[0]
                            )
                        else:
                            url = pdf_links[0]
                        logger.info(f"Found PDF link: {url}")
                    else:
                        raise ValueError(
                            f"No PDF file found at {url}. Please provide a direct link to a PDF file."
                        )

            # Step 2: Download PDF
            logger.info("Downloading PDF...")
            response = requests.get(
                url, timeout=30, headers={"User-Agent": "Lobster AI Research Tool/1.0"}
            )
            response.raise_for_status()

            # Step 3: Validate PDF content type
            content_type = response.headers.get("Content-Type", "").lower()
            if (
                "application/pdf" not in content_type
                and not response.content.startswith(b"%PDF")
            ):
                raise ValueError(
                    f"The URL did not return a valid PDF file. Content type: {content_type}"
                )

            # Step 4: Extract text with PyPDF2
            logger.info("Extracting text from PDF...")
            pdf_file = BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"

            # Step 5: Clean and return
            text = re.sub(r"\s+", " ", text).strip()

            if not text:
                raise ValueError(
                    "The PDF file did not contain any extractable text. It may be an image-based PDF requiring OCR."
                )

            # Cache the result (LOBSTER ENHANCEMENT)
            cache_file.write_text(text, encoding="utf-8")
            logger.info(
                f"Extracted {len(text)} characters from PDF, cached to {cache_file.name}"
            )

            # Log provenance if DataManager available (LOBSTER ENHANCEMENT)
            if self.data_manager:
                self.data_manager.log_tool_usage(
                    tool_name="extract_pdf_content",
                    parameters={"url": url[:100], "use_cache": use_cache},
                    description=f"PDF extraction: {len(text)} characters",
                )

            return text

        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading PDF from {url}: {e}")
            raise ValueError(f"Error downloading PDF: {str(e)}")
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise ValueError(f"Error extracting text from PDF: {str(e)}")

    def extract_url_content(self, url: str) -> str:
        """
        Extract clean text content from a webpage URL.

        Removes navigation, headers, footers, scripts, and other non-content elements.
        Useful for extracting methods sections from online papers or protocol pages.

        Adapted from Biomni with enhanced error handling.

        Args:
            url: Webpage URL to extract content from

        Returns:
            Clean text content from the webpage

        Raises:
            ValueError: If content extraction fails

        Examples:
            >>> service = PublicationIntelligenceService()
            >>> text = service.extract_url_content("https://nature.com/articles/12345")
            >>> print(text)
        """
        logger.info(f"Extracting URL content from: {url}")

        try:
            response = requests.get(
                url, headers={"User-Agent": "Lobster AI Research Tool/1.0"}, timeout=30
            )
            response.raise_for_status()

            # Handle plain text or JSON (ADAPTED FROM BIOMNI)
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/plain" in content_type or "application/json" in content_type:
                return response.text.strip()

            # Parse HTML (ADAPTED FROM BIOMNI)
            soup = BeautifulSoup(response.text, "html.parser")

            # Try to find main content first, fallback to body (BIOMNI PATTERN)
            content = soup.find("main") or soup.find("article") or soup.body

            if not content:
                raise ValueError("No content found in webpage")

            # Remove unwanted elements (BIOMNI PATTERN)
            for element in content(
                ["script", "style", "nav", "header", "footer", "aside", "iframe"]
            ):
                element.decompose()

            # Extract text with better formatting (BIOMNI PATTERN)
            paragraphs = content.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
            cleaned_text = []

            for p in paragraphs:
                text = p.get_text().strip()
                if text:  # Only add non-empty paragraphs
                    cleaned_text.append(text)

            result = "\n\n".join(cleaned_text)
            logger.info(f"Extracted {len(result)} characters from URL")

            # Log provenance (LOBSTER ENHANCEMENT)
            if self.data_manager:
                self.data_manager.log_tool_usage(
                    tool_name="extract_url_content",
                    parameters={"url": url[:100]},
                    description=f"URL extraction: {len(result)} characters",
                )

            return result

        except Exception as e:
            logger.error(f"Error extracting URL content: {e}")
            raise ValueError(f"Error extracting URL content: {str(e)}")

    def fetch_supplementary_info_from_doi(
        self, doi: str, output_dir: Optional[str] = None
    ) -> str:
        """
        Fetch supplementary information for a paper given its DOI.

        Adapted from Biomni with enhanced logging and error handling.

        Args:
            doi: The paper DOI (e.g., "10.1038/s41586-021-12345-6")
            output_dir: Directory to save supplementary files (default: .lobster_workspace/supplements/<doi>)

        Returns:
            Research log string with download information

        Examples:
            >>> service = PublicationIntelligenceService()
            >>> result = service.fetch_supplementary_info_from_doi("10.1038/s41586-021-12345-6")
            >>> print(result)
        """
        research_log = []
        research_log.append(f"Starting supplementary download for DOI: {doi}")
        logger.info(f"Fetching supplementary materials for DOI: {doi}")

        try:
            # Set default output directory
            if output_dir is None:
                safe_doi = doi.replace("/", "_").replace(":", "_")
                output_dir = str(Path(".lobster_workspace") / "supplements" / safe_doi)

            # CrossRef API to resolve DOI to publisher page (ADAPTED FROM BIOMNI)
            crossref_url = f"https://doi.org/{doi}"
            headers = {"User-Agent": "Lobster AI Research Tool/1.0"}
            response = requests.get(crossref_url, headers=headers, timeout=30)

            if response.status_code != 200:
                log_message = (
                    f"Failed to resolve DOI: {doi}. Status Code: {response.status_code}"
                )
                research_log.append(log_message)
                logger.error(log_message)
                return "\n".join(research_log)

            publisher_url = response.url
            research_log.append(f"Resolved DOI to publisher page: {publisher_url}")

            # Fetch publisher page (ADAPTED FROM BIOMNI)
            response = requests.get(publisher_url, headers=headers, timeout=30)
            if response.status_code != 200:
                log_message = f"Failed to access publisher page for DOI {doi}."
                research_log.append(log_message)
                logger.error(log_message)
                return "\n".join(research_log)

            # Parse page content (BIOMNI PATTERN)
            soup = BeautifulSoup(response.content, "html.parser")
            supplementary_links = []

            # Look for supplementary materials (BIOMNI PATTERN)
            for link in soup.find_all("a", href=True):
                href = link.get("href")
                text = link.get_text().lower()
                if (
                    "supplementary" in text
                    or "supplemental" in text
                    or "appendix" in text
                ):
                    full_url = urljoin(publisher_url, href)
                    supplementary_links.append(full_url)
                    research_log.append(
                        f"Found supplementary material link: {full_url}"
                    )

            if not supplementary_links:
                log_message = f"No supplementary materials found for DOI {doi}."
                research_log.append(log_message)
                logger.warning(log_message)
                return "\n".join(research_log)

            # Create output directory (LOBSTER ENHANCEMENT)
            os.makedirs(output_dir, exist_ok=True)
            research_log.append(f"Created output directory: {output_dir}")

            # Download supplementary materials (ADAPTED FROM BIOMNI)
            downloaded_files = []
            for link in supplementary_links:
                file_name = os.path.join(output_dir, link.split("/")[-1])
                file_response = requests.get(link, headers=headers, timeout=30)
                if file_response.status_code == 200:
                    with open(file_name, "wb") as f:
                        f.write(file_response.content)
                    downloaded_files.append(file_name)
                    research_log.append(f"Downloaded file: {file_name}")
                else:
                    research_log.append(f"Failed to download file from {link}")

            if downloaded_files:
                research_log.append(
                    f"Successfully downloaded {len(downloaded_files)} file(s)."
                )
                logger.info(
                    f"Downloaded {len(downloaded_files)} supplementary files for DOI {doi}"
                )
            else:
                research_log.append(f"No files could be downloaded for DOI {doi}.")
                logger.warning(f"No files could be downloaded for DOI {doi}")

            # Log provenance (LOBSTER ENHANCEMENT)
            if self.data_manager:
                self.data_manager.log_tool_usage(
                    tool_name="fetch_supplementary_info_from_doi",
                    parameters={"doi": doi, "output_dir": output_dir},
                    description=f"Downloaded {len(downloaded_files)} supplementary files",
                )

            return "\n".join(research_log)

        except Exception as e:
            logger.error(f"Error fetching supplementary materials for DOI {doi}: {e}")
            research_log.append(f"Error: {str(e)}")
            return "\n".join(research_log)

    def resolve_and_extract_methods(
        self, identifier: str, llm=None, max_text_length: int = 10000
    ) -> Dict[str, Any]:
        """
        Resolve identifier to PDF and extract methods (combined workflow).

        This method combines resolution + extraction in one step for convenience.

        Args:
            identifier: PMID, DOI, or PDF URL
            llm: LLM instance for analysis (uses default if None)
            max_text_length: Maximum text length to send to LLM (default: 10000)

        Returns:
            Dictionary with structured method extraction or error information:
            {
                "status": "success" | "paywalled" | "error",
                "methods": Dict,           # If successful
                "suggestions": str,        # If paywalled
                "source": str,            # Resolution source
                "metadata": Dict          # Resolution metadata
            }

        Examples:
            >>> service = PublicationIntelligenceService()
            >>> result = service.resolve_and_extract_methods("PMID:12345678")
            >>> if result['status'] == 'success':
            ...     print(result['methods']['software_used'])
            >>> elif result['status'] == 'paywalled':
            ...     print(result['suggestions'])
        """
        logger.info(f"Resolving and extracting methods from: {identifier}")

        try:
            # First, try to resolve to PDF URL if not already a direct URL
            from lobster.agents.research_agent_assistant import ResearchAgentAssistant

            assistant = ResearchAgentAssistant()

            # Check if it's already a direct URL
            if identifier.startswith("http"):
                # Direct URL, no resolution needed
                url = identifier
                source = "direct_url"
            else:
                # Try to resolve
                resolution = assistant.resolve_publication_to_pdf(identifier)

                if not resolution.is_accessible():
                    # Paper is paywalled or inaccessible
                    logger.warning(
                        f"Paper {identifier} is not accessible: {resolution.access_type}"
                    )
                    return {
                        "status": "paywalled",
                        "suggestions": resolution.suggestions,
                        "source": resolution.source,
                        "metadata": resolution.metadata,
                    }

                url = resolution.pdf_url
                source = resolution.source

            # Now extract methods
            methods = self.extract_methods_from_paper(
                url, llm=llm, max_text_length=max_text_length
            )

            return {
                "status": "success",
                "methods": methods,
                "source": source,
                "metadata": {"identifier": identifier, "pdf_url": url},
            }

        except Exception as e:
            logger.error(f"Error in resolve_and_extract_methods: {e}")
            return {
                "status": "error",
                "error": str(e),
                "suggestions": f"Failed to extract methods: {str(e)}",
            }

    def extract_methods_from_paper(
        self, url_or_pmid: str, llm=None, max_text_length: int = 10000
    ) -> Dict[str, Any]:
        """
        Extract computational analysis methods from a research paper using LLM.

        NOW SUPPORTS: Direct PDF URLs, PMIDs, and DOIs with automatic resolution!
        UPGRADED: Now uses Docling for structure-aware extraction!

        Uses LLM to analyze full paper text and extract:
        - Software/tools used
        - Parameter values and cutoffs
        - Statistical methods
        - Data sources
        - Sample sizes
        - Normalization methods
        - Quality control steps

        Args:
            url_or_pmid: Paper URL, PMID (e.g., "PMID:12345678"), or DOI
            llm: LLM instance for analysis (uses default if None)
            max_text_length: DEPRECATED - now uses intelligent section extraction

        Returns:
            Dictionary with structured method extraction

        Examples:
            >>> service = PublicationIntelligenceService()
            >>> # Now works with PMIDs!
            >>> methods = service.extract_methods_from_paper("PMID:12345678")
            >>> print(methods['software_used'])
            ['Scanpy', 'Seurat', 'DESeq2']
            >>> # Also works with DOIs
            >>> methods = service.extract_methods_from_paper("10.1038/s41586-021-12345-6")
            >>> # And still works with direct URLs
            >>> methods = service.extract_methods_from_paper("https://biorxiv.org/paper.pdf")
        """
        logger.info(f"Extracting methods from: {url_or_pmid}")

        try:
            # Step 1: Resolve to PDF URL if needed (ENHANCED - Phase 1)
            url = url_or_pmid

            if not url_or_pmid.startswith("http"):
                # Not a direct URL - try to resolve
                from lobster.agents.research_agent_assistant import (
                    ResearchAgentAssistant,
                )

                assistant = ResearchAgentAssistant()
                resolution = assistant.resolve_publication_to_pdf(url_or_pmid)

                if not resolution.is_accessible():
                    # Paper is paywalled - raise informative error
                    raise ValueError(
                        f"Paper {url_or_pmid} is not openly accessible.\n\n"
                        f"{resolution.suggestions}\n\n"
                        f"Please provide a direct PDF URL or try alternative sources."
                    )

                url = resolution.pdf_url
                logger.info(
                    f"Resolved {url_or_pmid} to PDF URL via {resolution.source}"
                )

            # Step 2: Extract Methods section (Docling or fallback)
            if self.converter is not None:
                # Use Docling for intelligent extraction
                extraction_result = self.extract_methods_section(url)

                if "error" not in extraction_result:
                    methods_text = extraction_result["methods_markdown"]
                    logger.info(
                        f"Extracted Methods section using Docling: {len(methods_text)} chars, "
                        f"{len(extraction_result['tables'])} tables, "
                        f"{len(extraction_result['formulas'])} formulas"
                    )
                else:
                    # Docling failed, use fallback
                    logger.warning("Docling extraction failed, using PyPDF2 fallback")
                    full_text = self.extract_pdf_content(url)
                    methods_text = full_text[:max_text_length]
            else:
                # Docling not available, use PyPDF2
                logger.info("Using PyPDF2 extraction (Docling not installed)")
                full_text = self.extract_pdf_content(url)
                methods_text = full_text[:max_text_length]

            # Step 3: Use LLM to extract structured methods
            if llm is None:
                from lobster.config.llm_factory import create_llm
                from lobster.config.settings import get_settings

                settings = get_settings()
                model_params = settings.get_agent_llm_params("research_agent")
                llm = create_llm("research_agent", model_params)

            # Construct extraction prompt
            prompt = f"""
You are a bioinformatics expert analyzing research papers.
Extract the following computational analysis details from this paper:

1. Software/tools used (e.g., Scanpy, Seurat, DESeq2)
2. Parameter values (e.g., cutoffs, thresholds, p-values)
3. Statistical methods (e.g., Wilcoxon test, t-test, ANOVA)
4. Data sources (e.g., GEO datasets, cell lines)
5. Sample sizes (e.g., number of cells, replicates)
6. Normalization methods (e.g., log1p, TPM, TMM)
7. Quality control steps (e.g., filtering criteria)

Methods section text:
{methods_text}

Return a JSON object with these fields:
{{
  "software_used": ["list of tools"],
  "parameters": {{"parameter_name": "value"}},
  "statistical_methods": ["list of methods"],
  "data_sources": ["list of sources"],
  "sample_sizes": {{"description": "value"}},
  "normalization_methods": ["list of methods"],
  "quality_control": ["list of QC steps"]
}}
"""

            # Get LLM response
            response = llm.invoke(prompt)

            # Parse JSON from response
            try:
                # Extract JSON from markdown code blocks if present
                json_match = re.search(
                    r"```(?:json)?\n(.*?)\n```", response.content, re.DOTALL
                )
                if json_match:
                    methods = json.loads(json_match.group(1))
                else:
                    # Try to parse entire response as JSON
                    methods = json.loads(response.content)
            except json.JSONDecodeError:
                logger.warning(
                    "Could not parse LLM response as JSON, returning raw text"
                )
                methods = {"raw_extraction": response.content}

            logger.info(f"Extracted methods with fields: {list(methods.keys())}")

            # Step 4: Enrich with extracted metadata (if Docling was used)
            if self.converter and "error" not in extraction_result:
                methods["tables"] = extraction_result["tables"]
                methods["formulas"] = extraction_result["formulas"]
                methods["software_detected"] = extraction_result["software_mentioned"]
                methods["extraction_metadata"] = extraction_result["provenance"]

                # Step 4.5: Persist as markdown for session access
                try:
                    self.persist_extraction_as_markdown(url_or_pmid, extraction_result)
                    logger.info(f"Persisted extraction for session access: {url_or_pmid[:50]}")
                except Exception as e:
                    logger.warning(f"Failed to persist markdown (non-fatal): {e}")

            # Step 5: Log provenance
            if self.data_manager:
                parser_used = "docling" if self.converter else "pypdf2"
                self.data_manager.log_tool_usage(
                    tool_name="extract_methods_from_paper",
                    parameters={"source": url_or_pmid[:100], "parser": parser_used},
                    description=f"Method extraction ({parser_used}): {list(methods.keys())}",
                )

            return methods

        except Exception as e:
            logger.error(f"Error extracting methods from paper: {e}")
            raise ValueError(f"Error extracting methods from paper: {str(e)}")

    # ==================== DOCLING-BASED METHODS EXTRACTION ====================

    def _process_docling_document(
        self, doc, source: str, keywords: List[str], max_paragraphs: int, DocItemLabel
    ) -> Dict[str, Any]:
        """
        Process an already-converted DoclingDocument to extract Methods section.

        This helper method separates document processing from conversion and caching,
        enabling cleaner retry logic in extract_methods_section().

        Args:
            doc: DoclingDocument instance (already parsed)
            source: Original PDF URL/path (for provenance)
            keywords: Section keywords for Methods detection
            max_paragraphs: Maximum paragraphs to extract
            DocItemLabel: Docling label enum for section detection

        Returns:
            Dictionary with methods_text, methods_markdown, tables, formulas, etc.

        Raises:
            MethodsSectionNotFoundError: If Methods section cannot be located
        """
        # Find Methods section header
        methods_sections = self._find_sections_by_keywords(doc, keywords, DocItemLabel)

        if not methods_sections:
            logger.warning(
                "No Methods section found with keywords, returning full document"
            )
            # Fallback: return full document with structure
            return self._extract_full_document(doc, max_paragraphs, DocItemLabel)

        # Extract Methods content
        methods_text = self._extract_section_content(
            doc, methods_sections[0], max_paragraphs, DocItemLabel
        )

        # Export to Markdown with smart image filtering (Phase 2)
        try:
            # Export full document to Markdown using Docling
            full_markdown = doc.export_to_markdown()
            # Apply image filtering to remove base64 bloat
            methods_markdown = self._filter_images_from_markdown(full_markdown)
        except Exception as e:
            logger.warning(f"Markdown export failed, using plain text: {e}")
            methods_markdown = methods_text

        # Extract tables within Methods section
        tables = self._extract_tables_in_section(doc, methods_sections[0])

        # Extract formulas
        formulas = self._extract_formulas_in_section(
            doc, methods_sections[0], DocItemLabel
        )

        # Extract software names
        software = self._extract_software_names(methods_text)

        # Log provenance (metadata-only, no IR)
        if self.data_manager:
            self.data_manager.log_tool_usage(
                tool_name="extract_methods_section",
                parameters={
                    "source": source[:100],
                    "keywords": keywords,
                    "max_paragraphs": max_paragraphs,
                },
                description=f"Methods extraction: {len(methods_text)} chars, "
                f"{len(tables)} tables, {len(formulas)} formulas",
            )

        # Build result dictionary
        result = {
            "methods_text": methods_text,
            "methods_markdown": methods_markdown,
            "sections": self._build_section_hierarchy(doc, DocItemLabel),
            "tables": [self._table_to_dataframe(t) for t in tables],
            "formulas": formulas,
            "software_mentioned": software,
            "provenance": {
                "source": source,
                "parser": "docling",
                "version": "2.60.0",
                "timestamp": datetime.now().isoformat(),
                "fallback_used": False,
            },
        }

        # Persist as markdown for session access
        try:
            self.persist_extraction_as_markdown(source, result)
            logger.info(f"Persisted extraction for session access: {source[:50]}")
        except Exception as e:
            logger.warning(f"Failed to persist markdown (non-fatal): {e}")

        return result

    def extract_methods_section(
        self,
        source: str,
        keywords: Optional[List[str]] = None,
        max_paragraphs: int = 50,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Extract Methods section with structure awareness using Docling.

        This method uses Docling's structure-aware PDF parsing with:
        - Automatic Methods section detection by keywords
        - Complete section extraction (not truncated)
        - Table and formula preservation
        - Smart image filtering (Phase 2)
        - Document caching for performance (Phase 2)
        - Comprehensive retry logic and error handling (Phase 3)

        Args:
            source: URL or local path to PDF
            keywords: Section keywords to search for (default: method-related)
            max_paragraphs: Maximum paragraphs to extract (default: 50)
            max_retries: Maximum retry attempts (default: 2)

        Returns:
            Dictionary with:
                'methods_text': str - Full Methods section
                'methods_markdown': str - Markdown with tables (images filtered)
                'sections': List[Dict] - Hierarchical structure
                'tables': List[DataFrame] - Extracted tables
                'formulas': List[str] - Mathematical formulas
                'software_mentioned': List[str] - Tool names detected
                'provenance': Dict - Tracking metadata

        Examples:
            >>> service = PublicationIntelligenceService()
            >>> result = service.extract_methods_section(
            ...     "https://arxiv.org/pdf/2408.09869"
            ... )
            >>> print(f"Found {len(result['tables'])} tables")
            >>> print(result['methods_text'][:200])
        """
        if keywords is None:
            keywords = ["method", "material", "procedure", "experimental"]

        logger.info(f"Extracting Methods section from: {source[:80]}...")

        # Check if Docling is available
        if self.converter is None:
            logger.warning("Docling not available, using PyPDF2 fallback")
            return self._extract_with_pypdf2_fallback(source, keywords=keywords)

        # Import Docling types
        from docling_core.types.doc import DocItemLabel

        # Phase 3: Retry loop with comprehensive error handling
        for attempt in range(max_retries):
            try:
                logger.info(f"Extraction attempt {attempt + 1}/{max_retries}")

                # Attempt 1: Check cache first (Phase 2.3)
                cached_doc = self._get_cached_document(source)

                if cached_doc:
                    logger.info("Using cached parsed document (cache hit)")
                    return self._process_docling_document(
                        cached_doc, source, keywords, max_paragraphs, DocItemLabel
                    )

                # Attempt 2: Fresh Docling parse (cache miss)
                logger.info("Parsing PDF with Docling (cache miss)")
                result = self.converter.convert(source)
                ConversionStatus = self._docling_imports["ConversionStatus"]

                # Handle conversion status
                if result.status == ConversionStatus.SUCCESS:
                    logger.info("Docling conversion: SUCCESS")
                    doc = result.document

                    # Cache for future use
                    self._cache_document(source, doc)

                    # Cleanup conversion result
                    del result
                    gc.collect()

                    # Process document
                    return self._process_docling_document(
                        doc, source, keywords, max_paragraphs, DocItemLabel
                    )

                elif result.status == ConversionStatus.PARTIAL_SUCCESS:
                    logger.warning(
                        "Docling conversion: PARTIAL_SUCCESS (using available data)"
                    )
                    doc = result.document

                    # Still cache partial result
                    self._cache_document(source, doc)

                    # Cleanup
                    del result
                    gc.collect()

                    # Process what we have
                    return self._process_docling_document(
                        doc, source, keywords, max_paragraphs, DocItemLabel
                    )

                else:
                    # FAILURE status
                    logger.error(
                        f"Docling conversion: FAILURE (status={result.status})"
                    )
                    raise DoclingError(
                        f"Conversion failed with status: {result.status}"
                    )

            except MemoryError as e:
                logger.error(f"MemoryError on attempt {attempt + 1}/{max_retries}: {e}")
                # Aggressive cleanup
                gc.collect()

                if attempt < max_retries - 1:
                    logger.info("Retrying after memory cleanup...")
                    continue
                else:
                    logger.warning(
                        "Max retries reached after MemoryError, falling back to PyPDF2"
                    )
                    break

            except RuntimeError as e:
                error_msg = str(e)
                if "page-dimensions" in error_msg:
                    logger.error(
                        f"RuntimeError: Incompatible PDF (page-dimensions issue)"
                    )
                    # Don't retry this error - it's a permanent PDF incompatibility
                    break
                else:
                    logger.error(
                        f"RuntimeError on attempt {attempt + 1}/{max_retries}: {e}"
                    )
                    if attempt < max_retries - 1:
                        continue
                    raise

            except DoclingError as e:
                logger.error(
                    f"DoclingError on attempt {attempt + 1}/{max_retries}: {e}"
                )
                if attempt < max_retries - 1:
                    logger.info("Retrying after Docling error...")
                    continue
                else:
                    logger.warning("Max retries reached, falling back to PyPDF2")
                    break

            except Exception as e:
                logger.exception(
                    f"Unexpected error on attempt {attempt + 1}/{max_retries}: {e}"
                )
                if attempt < max_retries - 1:
                    logger.info("Retrying after unexpected error...")
                    continue
                else:
                    logger.warning("Max retries reached, falling back to PyPDF2")
                    break

        # Fallback: All retries exhausted or incompatible PDF
        logger.info("Using PyPDF2 fallback after Docling failures")
        return self._extract_with_pypdf2_fallback(source, keywords=keywords)

    def _find_sections_by_keywords(
        self, doc, keywords: List[str], DocItemLabel
    ) -> List:
        """
        Find section headers matching keywords.

        Strategy:
        - Case-insensitive matching
        - Partial word matching ("Method" matches "Methods and Materials")
        - Prioritize exact matches
        - Return all matches (user can inspect if multiple)

        Args:
            doc: DoclingDocument instance
            keywords: List of keywords to search for
            DocItemLabel: Docling label enum

        Returns:
            List of matching section header items
        """
        exact_matches = []
        partial_matches = []

        for item in doc.texts:
            if item.label == DocItemLabel.SECTION_HEADER:
                text_lower = item.text.lower()

                # Check for exact keyword match
                for keyword in keywords:
                    if keyword.lower() == text_lower:
                        exact_matches.append(item)
                        break

                # Check for partial match
                else:
                    for keyword in keywords:
                        if keyword.lower() in text_lower:
                            partial_matches.append(item)
                            break

        # Prioritize exact matches
        matches = exact_matches if exact_matches else partial_matches
        logger.info(f"Found {len(matches)} section(s) matching keywords: {keywords}")
        return matches

    def _extract_section_content(
        self, doc, section_header, max_paragraphs: int, DocItemLabel
    ) -> str:
        """
        Extract text content under a section header.

        Strategy:
        - Start at section header
        - Extract until next section header OR max_paragraphs reached
        - Include paragraphs only (exclude tables/figures from text)
        - Preserve paragraph boundaries

        Args:
            doc: DoclingDocument instance
            section_header: Section header item to start from
            max_paragraphs: Maximum paragraphs to extract
            DocItemLabel: Docling label enum

        Returns:
            Extracted section text
        """
        start_idx = doc.texts.index(section_header)

        content = []
        paragraph_count = 0

        for item in doc.texts[start_idx + 1 :]:  # Skip header itself
            # Stop at next major section
            if item.label == DocItemLabel.SECTION_HEADER:
                break

            # Extract paragraph text
            if item.label == DocItemLabel.PARAGRAPH:
                if hasattr(item, "text") and item.text.strip():
                    content.append(item.text)
                    paragraph_count += 1

                    if paragraph_count >= max_paragraphs:
                        logger.info(f"Reached max_paragraphs limit: {max_paragraphs}")
                        break

        result = "\n\n".join(content)
        logger.info(
            f"Extracted {len(result)} characters from {paragraph_count} paragraphs"
        )
        return result

    def _find_section_end(self, doc, start_idx: int, DocItemLabel) -> int:
        """
        Find the end index of a section.

        Args:
            doc: DoclingDocument instance
            start_idx: Starting index
            DocItemLabel: Docling label enum

        Returns:
            End index of the section
        """
        for idx in range(start_idx + 1, len(doc.texts)):
            if doc.texts[idx].label == DocItemLabel.SECTION_HEADER:
                return idx
        return len(doc.texts)

    def _extract_tables_in_section(self, doc, section_header) -> List:
        """
        Extract tables within a section's page range.

        Args:
            doc: DoclingDocument instance
            section_header: Section header item

        Returns:
            List of TableItem objects
        """
        section_pages = set()
        if hasattr(section_header, "prov"):
            for prov in section_header.prov:
                if hasattr(prov, "page_no"):
                    section_pages.add(prov.page_no)

        section_tables = []
        if hasattr(doc, "tables"):
            for table in doc.tables:
                if hasattr(table, "prov"):
                    for prov in table.prov:
                        if getattr(prov, "page_no", None) in section_pages:
                            section_tables.append(table)
                            break

        logger.info(f"Found {len(section_tables)} tables in section")
        return section_tables

    def _extract_formulas_in_section(
        self, doc, section_header, DocItemLabel
    ) -> List[str]:
        """
        Extract formulas within a section.

        Args:
            doc: DoclingDocument instance
            section_header: Section header item
            DocItemLabel: Docling label enum

        Returns:
            List of formula strings
        """
        start_idx = doc.texts.index(section_header)
        end_idx = self._find_section_end(doc, start_idx, DocItemLabel)

        formulas = []
        for idx in range(start_idx, end_idx):
            item = doc.texts[idx]
            if item.label == DocItemLabel.FORMULA:
                if hasattr(item, "text"):
                    formulas.append(item.text)

        logger.info(f"Found {len(formulas)} formulas in section")
        return formulas

    def _extract_software_names(self, text: str) -> List[str]:
        """
        Extract software/tool names from text.

        Args:
            text: Text to search

        Returns:
            List of detected software names
        """
        software_keywords = [
            "scanpy",
            "seurat",
            "star",
            "kallisto",
            "salmon",
            "deseq2",
            "limma",
            "edger",
            "cellranger",
            "maxquant",
            "mofa",
            "harmony",
            "combat",
            "mnn",
            "fastqc",
            "trimmomatic",
            "cutadapt",
            "bowtie",
            "hisat2",
            "tophat",
            "spectronaut",
            "maxdia",
            "fragpipe",
            "msfragger",
        ]

        text_lower = text.lower()
        found = []
        for sw in software_keywords:
            if sw in text_lower:
                found.append(sw)

        return found

    def _build_section_hierarchy(self, doc, DocItemLabel) -> List[Dict]:
        """
        Build hierarchical section structure.

        Args:
            doc: DoclingDocument instance
            DocItemLabel: Docling label enum

        Returns:
            List of section dictionaries
        """
        hierarchy = []
        current_section = None

        for item in doc.texts:
            if item.label == DocItemLabel.SECTION_HEADER:
                if current_section:
                    hierarchy.append(current_section)

                current_section = {
                    "title": item.text,
                    "level": 1,  # Could infer from font size
                    "content_preview": "",
                }
            elif current_section and item.label == DocItemLabel.PARAGRAPH:
                if len(current_section["content_preview"]) < 200:
                    current_section["content_preview"] += item.text[:200]

        if current_section:
            hierarchy.append(current_section)

        return hierarchy

    def _extract_full_document(
        self, doc, max_paragraphs: int, DocItemLabel
    ) -> Dict[str, Any]:
        """
        Extract full document when Methods section not found.

        Args:
            doc: DoclingDocument instance
            max_paragraphs: Maximum paragraphs to extract
            DocItemLabel: Docling label enum

        Returns:
            Dictionary with full document extraction
        """
        content = []
        paragraph_count = 0

        for item in doc.texts:
            if item.label == DocItemLabel.PARAGRAPH:
                if hasattr(item, "text") and item.text.strip():
                    content.append(item.text)
                    paragraph_count += 1

                    if paragraph_count >= max_paragraphs:
                        break

        full_text = "\n\n".join(content)

        return {
            "methods_text": full_text,
            "methods_markdown": full_text,
            "sections": self._build_section_hierarchy(doc, DocItemLabel),
            "tables": [],
            "formulas": [],
            "software_mentioned": self._extract_software_names(full_text),
            "provenance": {
                "source": "full_document",
                "parser": "docling",
                "timestamp": datetime.now().isoformat(),
                "fallback_used": False,
                "note": "Methods section not found, extracted full document",
            },
        }

    def _table_to_dataframe(self, table_item):
        """
        Convert Docling TableItem to pandas DataFrame.

        Args:
            table_item: Docling TableItem object

        Returns:
            pandas DataFrame or dict representation
        """
        try:
            # Try to export as DataFrame if available
            if hasattr(table_item, "export_to_dataframe"):
                return table_item.export_to_dataframe()
            else:
                # Return dict representation as fallback
                return {"error": "DataFrame export not available"}
        except Exception as e:
            logger.warning(f"Could not convert table to DataFrame: {e}")
            return {"error": str(e)}

    def _filter_images_from_markdown(self, markdown: str) -> str:
        """
        Remove base64 image encodings from Markdown to reduce LLM context bloat.

        Base64-encoded images can add megabytes of unnecessary text. This method
        replaces them with simple placeholders while preserving document structure.

        Args:
            markdown: Markdown text potentially containing base64 images

        Returns:
            Filtered markdown with image placeholders

        Examples:
            >>> # Before: ![Figure 1](data:image/png;base64,iVBORw0KG...)
            >>> # After:  [Figure: Figure 1]
        """
        # Pattern matches: ![caption](data:image/...;base64,...)
        pattern = r"!\[([^\]]*)\]\(data:image/[^;]+;base64,[^\)]+\)"

        def replace_image(match):
            caption = match.group(1).strip() or "Image"
            return f"[Figure: {caption}]"

        original_size = len(markdown)
        filtered = re.sub(pattern, replace_image, markdown)
        filtered_size = len(filtered)

        if original_size != filtered_size:
            reduction_pct = ((original_size - filtered_size) / original_size) * 100
            logger.info(
                f"Filtered images from Markdown: {original_size:,} → {filtered_size:,} chars "
                f"({reduction_pct:.1f}% reduction)"
            )

        return filtered

    def _get_cached_document(self, source: str):
        """
        Retrieve cached DoclingDocument if available.

        Caching parsed documents significantly improves performance:
        - Fresh parse: 2-5 seconds
        - Cache hit: <100ms

        Args:
            source: PDF URL or path (used as cache key)

        Returns:
            DoclingDocument if cached, None if cache miss or error

        Implementation:
            - Cache key: MD5 hash of source URL
            - Storage: JSON serialization in parsed_docs subdirectory
            - Reconstruction: Pydantic model_validate() from JSON
        """
        if not self.converter:
            return None  # Docling not available

        try:
            # Generate cache key from source URL
            cache_key = hashlib.md5(source.encode()).hexdigest()
            cache_file = self.parsed_docs_cache / f"{cache_key}.json"

            if not cache_file.exists():
                return None  # Cache miss

            # Load JSON and reconstruct DoclingDocument
            with open(cache_file, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            # Import DoclingDocument for reconstruction
            from docling_core.types.doc import DoclingDocument

            # Reconstruct document from JSON using Pydantic
            doc = DoclingDocument.model_validate(json_data)

            logger.info(f"Cache hit: Loaded parsed document from {cache_file.name}")
            return doc

        except Exception as e:
            logger.warning(f"Failed to load cached document: {e}")
            return None  # Cache read error, will re-parse

    def _cache_document(self, source: str, doc) -> None:
        """
        Cache DoclingDocument as JSON for future retrieval.

        Args:
            source: PDF URL or path (used as cache key)
            doc: DoclingDocument instance to cache

        Implementation:
            - Serialization: Pydantic model_dump() to dict
            - Storage: JSON with indent=2 for human readability
            - Error handling: Graceful failure (doesn't block extraction)
        """
        if not self.converter or not doc:
            return  # Nothing to cache

        try:
            # Generate cache key
            cache_key = hashlib.md5(source.encode()).hexdigest()
            cache_file = self.parsed_docs_cache / f"{cache_key}.json"

            # Serialize document to JSON using Pydantic
            json_data = doc.model_dump()

            # Write to cache file
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2)

            logger.info(
                f"Cached parsed document: {cache_file.name} ({len(json.dumps(json_data)):,} bytes)"
            )

        except Exception as e:
            logger.warning(f"Failed to cache document (non-fatal): {e}")
            # Don't raise - caching failure shouldn't block extraction

    def _extract_with_pypdf2_fallback(
        self, source: str, keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Fallback to PyPDF2 when Docling fails or is unavailable.

        Args:
            source: PDF URL or path
            keywords: Section keywords (for heuristic detection)

        Returns:
            Dictionary with same structure as Docling extraction
        """
        try:
            logger.info("Using PyPDF2 fallback for PDF extraction")
            full_text = self.extract_pdf_content(source)

            # Basic Methods section detection using heuristics
            methods_text = self._naive_methods_extraction(full_text, keywords)

            return {
                "methods_text": methods_text,
                "methods_markdown": methods_text,
                "sections": [],
                "tables": [],
                "formulas": [],
                "software_mentioned": self._extract_software_names(methods_text),
                "provenance": {
                    "source": source,
                    "parser": "pypdf2",
                    "fallback_used": True,
                    "timestamp": datetime.now().isoformat(),
                },
            }
        except Exception as e:
            logger.exception("PyPDF2 fallback also failed")
            return {
                "error": str(e),
                "methods_text": "",
                "methods_markdown": "",
                "sections": [],
                "tables": [],
                "formulas": [],
                "software_mentioned": [],
                "provenance": {
                    "source": source,
                    "parser": "none",
                    "fallback_used": True,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                },
            }

    def _naive_methods_extraction(
        self, full_text: str, keywords: Optional[List[str]] = None
    ) -> str:
        """
        Simple heuristic Methods extraction for PyPDF2 fallback.

        Strategy:
        - Search for "Methods", "Materials and Methods" keywords
        - Extract until "Results" or "Discussion"
        - If not found, return first 10K characters (old behavior)

        Args:
            full_text: Full PDF text
            keywords: Section keywords to search for

        Returns:
            Extracted Methods section text
        """
        if keywords is None:
            keywords = ["methods", "materials and methods", "experimental procedures"]

        end_keywords = ["results", "discussion", "conclusion"]

        text_lower = full_text.lower()

        # Find Methods section start
        start_idx = None
        for keyword in keywords:
            idx = text_lower.find(keyword)
            if idx != -1:
                start_idx = idx
                logger.info(f"Found '{keyword}' at position {idx}")
                break

        if start_idx is None:
            # No Methods section found, return first 10K
            logger.warning("No Methods section found, returning first 10000 characters")
            return full_text[:10000]

        # Find Methods section end
        end_idx = len(full_text)
        for keyword in end_keywords:
            idx = text_lower.find(keyword, start_idx + 100)
            if idx != -1:
                end_idx = min(end_idx, idx)

        extracted = full_text[start_idx:end_idx]
        logger.info(f"Extracted {len(extracted)} characters using heuristic")
        return extracted

    # ==================== SESSION PUBLICATION ACCESS ====================

    def persist_extraction_as_markdown(
        self, identifier: str, extraction_result: Dict[str, Any]
    ) -> Path:
        """
        Save extraction result as Markdown file for human readability.

        This method persists the methods extraction in a human-readable
        format alongside the JSON cache, enabling agents to access
        previously extracted publications in follow-up conversations.

        Args:
            identifier: Publication identifier (PMID, DOI, or URL)
            extraction_result: Dictionary returned from extract_methods_section()
                             or extract_methods_from_paper()

        Returns:
            Path to saved .md file

        Examples:
            >>> service = PublicationIntelligenceService()
            >>> result = service.extract_methods_section("https://paper.pdf")
            >>> md_path = service.persist_extraction_as_markdown(
            ...     "PMID:12345678", result
            ... )
            >>> print(md_path)
            .lobster_workspace/literature_cache/publications/PMID_12345678.md
        """
        # Create publications subdirectory
        publications_dir = self.cache_dir / "publications"
        publications_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize identifier for filename
        safe_identifier = identifier.replace(":", "_").replace("/", "_").replace("\\", "_")
        md_file = publications_dir / f"{safe_identifier}.md"

        # Build markdown content
        markdown_lines = [
            f"# Publication: {identifier}",
            "",
            f"**Extraction Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Parser**: {extraction_result.get('provenance', {}).get('parser', 'unknown')}",
            "",
            "---",
            "",
            "## Methods Section",
            "",
            extraction_result.get("methods_markdown", extraction_result.get("methods_text", "")),
            "",
        ]

        # Add tables if present
        tables = extraction_result.get("tables", [])
        if tables:
            markdown_lines.extend([
                "---",
                "",
                f"## Extracted Tables ({len(tables)})",
                "",
            ])
            for i, table in enumerate(tables, 1):
                markdown_lines.append(f"### Table {i}")
                markdown_lines.append("")
                if isinstance(table, dict) and "error" not in table:
                    markdown_lines.append("```")
                    markdown_lines.append(str(table))
                    markdown_lines.append("```")
                else:
                    markdown_lines.append(f"*Table extraction error: {table.get('error', 'Unknown')}*")
                markdown_lines.append("")

        # Add formulas if present
        formulas = extraction_result.get("formulas", [])
        if formulas:
            markdown_lines.extend([
                "---",
                "",
                f"## Extracted Formulas ({len(formulas)})",
                "",
            ])
            for i, formula in enumerate(formulas, 1):
                markdown_lines.append(f"**Formula {i}**: `{formula}`")
                markdown_lines.append("")

        # Add software mentions if present
        software = extraction_result.get("software_mentioned", [])
        if software:
            markdown_lines.extend([
                "---",
                "",
                "## Software Tools Detected",
                "",
                ", ".join(f"`{sw}`" for sw in software),
                "",
            ])

        # Add provenance metadata
        provenance = extraction_result.get("provenance", {})
        if provenance:
            markdown_lines.extend([
                "---",
                "",
                "## Extraction Metadata",
                "",
                f"- **Source**: {provenance.get('source', 'N/A')}",
                f"- **Parser**: {provenance.get('parser', 'N/A')}",
                f"- **Fallback Used**: {provenance.get('fallback_used', False)}",
                f"- **Timestamp**: {provenance.get('timestamp', 'N/A')}",
                "",
            ])

        # Write markdown file
        md_content = "\n".join(markdown_lines)
        md_file.write_text(md_content, encoding="utf-8")

        logger.info(f"Persisted extraction as markdown: {md_file.name} ({len(md_content)} chars)")
        return md_file

    def get_cached_publication(self, identifier: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached publication by identifier.

        This method allows agents to access previously extracted publications
        for follow-up questions without re-extraction. It checks both the
        Markdown cache and JSON cache.

        Args:
            identifier: Publication identifier (PMID, DOI, or URL)

        Returns:
            Dictionary with extraction results or None if not found:
            {
                "identifier": str,
                "methods_text": str,
                "methods_markdown": str,
                "tables": List,
                "formulas": List[str],
                "software_mentioned": List[str],
                "provenance": Dict,
                "cache_source": "markdown" | "json"
            }

        Examples:
            >>> service = PublicationIntelligenceService()
            >>> # After previous extraction...
            >>> cached = service.get_cached_publication("PMID:12345678")
            >>> if cached:
            ...     print(cached["methods_text"][:100])
        """
        # Strategy 1: Check Markdown cache (human-readable)
        safe_identifier = identifier.replace(":", "_").replace("/", "_").replace("\\", "_")
        publications_dir = self.cache_dir / "publications"
        md_file = publications_dir / f"{safe_identifier}.md"

        if md_file.exists():
            logger.info(f"Found cached publication (markdown): {md_file.name}")
            try:
                content = md_file.read_text(encoding="utf-8")
                # Parse markdown to extract structured data
                return {
                    "identifier": identifier,
                    "methods_text": content,  # Full markdown content
                    "methods_markdown": content,
                    "tables": [],  # Tables are in markdown format
                    "formulas": [],  # Formulas are in markdown format
                    "software_mentioned": [],  # Would require parsing
                    "provenance": {
                        "source": "markdown_cache",
                        "parser": "unknown",
                        "cached": True,
                    },
                    "cache_source": "markdown",
                }
            except Exception as e:
                logger.warning(f"Failed to read markdown cache: {e}")

        # Strategy 2: Check JSON cache (structured)
        # Try to find JSON cache by identifier
        # Note: JSON cache uses MD5(url) as key, so we need to try common URL patterns
        possible_urls = [
            identifier,  # Direct URL
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/{identifier}/",  # PMC
            f"https://pubmed.ncbi.nlm.nih.gov/{identifier.replace('PMID:', '')}/",  # PubMed
        ]

        for url in possible_urls:
            cache_key = hashlib.md5(url.encode()).hexdigest()
            cache_file = self.parsed_docs_cache / f"{cache_key}.json"

            if cache_file.exists():
                logger.info(f"Found cached publication (json): {cache_file.name}")
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        json_data = json.load(f)

                    # Reconstruct DoclingDocument if Docling available
                    if self.converter:
                        try:
                            from docling_core.types.doc import DoclingDocument

                            doc = DoclingDocument.model_validate(json_data)

                            # Re-extract methods section
                            from docling_core.types.doc import DocItemLabel

                            keywords = ["method", "material", "procedure", "experimental"]
                            result = self._process_docling_document(
                                doc, url, keywords, 50, DocItemLabel
                            )
                            result["identifier"] = identifier
                            result["cache_source"] = "json"
                            return result
                        except Exception as e:
                            logger.warning(f"Failed to reconstruct from JSON: {e}")

                except Exception as e:
                    logger.warning(f"Failed to read JSON cache: {e}")

        logger.info(f"No cached publication found for: {identifier}")
        return None

    def list_session_publications(
        self, data_manager: DataManagerV2
    ) -> List[Dict[str, Any]]:
        """
        List publications extracted in the current session.

        This method uses DataManagerV2's provenance tracking to identify
        publications that were extracted during the current session, then
        checks cache status for each.

        Args:
            data_manager: DataManagerV2 instance with provenance history

        Returns:
            List of publication summaries:
            [{
                "identifier": str,
                "tool_name": str,
                "timestamp": str,
                "cache_status": "markdown" | "json" | "both" | "none",
                "methods_length": int,
                "source": str
            }]

        Examples:
            >>> service = PublicationIntelligenceService(data_manager)
            >>> publications = service.list_session_publications(data_manager)
            >>> for pub in publications:
            ...     print(f"{pub['identifier']}: {pub['cache_status']}")
        """
        publications = []

        # Get tool usage history from data manager
        tool_history = getattr(data_manager, "tool_usage_history", [])

        # Find all publication extraction operations
        extraction_tools = [
            "extract_pdf_content",
            "extract_methods_from_paper",
            "extract_methods_section",
        ]

        for entry in tool_history:
            tool_name = entry.get("tool_name", "")
            if tool_name in extraction_tools:
                params = entry.get("parameters", {})
                description = entry.get("description", "")

                # Try to extract identifier from parameters
                identifier = (
                    params.get("url_or_pmid")
                    or params.get("source")
                    or params.get("url")
                    or "unknown"
                )

                # Truncate long URLs for display
                if len(identifier) > 80:
                    identifier = identifier[:77] + "..."

                # Check cache status
                cache_status = self._check_cache_status(identifier)

                # Estimate methods length from description
                methods_length = 0
                if "characters" in description:
                    try:
                        methods_length = int(description.split()[0].replace(",", ""))
                    except (ValueError, IndexError):
                        pass

                publications.append({
                    "identifier": identifier,
                    "tool_name": tool_name,
                    "timestamp": entry.get("timestamp", "unknown"),
                    "cache_status": cache_status,
                    "methods_length": methods_length,
                    "source": params.get("parser", "unknown"),
                })

        logger.info(f"Found {len(publications)} publications in current session")
        return publications

    def _check_cache_status(self, identifier: str) -> str:
        """
        Check cache status for a publication identifier.

        Args:
            identifier: Publication identifier

        Returns:
            Cache status: "markdown", "json", "both", or "none"
        """
        has_markdown = False
        has_json = False

        # Check markdown cache
        safe_identifier = identifier.replace(":", "_").replace("/", "_").replace("\\", "_")
        publications_dir = self.cache_dir / "publications"
        md_file = publications_dir / f"{safe_identifier}.md"
        has_markdown = md_file.exists()

        # Check JSON cache (try common URL patterns)
        possible_urls = [
            identifier,
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/{identifier}/",
            f"https://pubmed.ncbi.nlm.nih.gov/{identifier.replace('PMID:', '')}/",
        ]

        for url in possible_urls:
            cache_key = hashlib.md5(url.encode()).hexdigest()
            cache_file = self.parsed_docs_cache / f"{cache_key}.json"
            if cache_file.exists():
                has_json = True
                break

        if has_markdown and has_json:
            return "both"
        elif has_markdown:
            return "markdown"
        elif has_json:
            return "json"
        else:
            return "none"
