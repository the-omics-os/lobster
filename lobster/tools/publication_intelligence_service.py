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
"""

import hashlib
import os
import re
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


class PublicationIntelligenceService:
    """
    Extract methods and intelligence from scientific publications.

    This service provides:
    - PDF text extraction from URLs
    - Webpage content extraction
    - LLM-based method extraction from papers
    - Supplementary material download from DOIs
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
        logger.info(f"Initialized PublicationIntelligenceService with cache: {self.cache_dir}")

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
                text = cache_file.read_text(encoding='utf-8')

                # Log provenance for cached retrieval (LOBSTER ENHANCEMENT)
                if self.data_manager:
                    self.data_manager.log_tool_usage(
                        tool_name="extract_pdf_content",
                        parameters={"url": url[:100], "use_cache": use_cache, "cache_hit": True},
                        description=f"PDF extraction from cache: {len(text)} characters"
                    )

                return text

            # Step 1: Smart URL detection 
            if not url.lower().endswith(".pdf"):
                logger.info("URL doesn't end with .pdf, searching for PDF links...")
                response = requests.get(url, timeout=30, headers={"User-Agent": "Lobster AI Research Tool/1.0"})
                response.raise_for_status()

                if response.status_code == 200:
                    # Look for PDF links in HTML 
                    pdf_links = re.findall(r'href=[\'"]([^\'"]+\.pdf)[\'"]', response.text)
                    if pdf_links:
                        # Handle relative URLs
                        if not pdf_links[0].startswith("http"):
                            base_url = "/".join(url.split("/")[:3])
                            url = base_url + pdf_links[0] if pdf_links[0].startswith("/") else base_url + "/" + pdf_links[0]
                        else:
                            url = pdf_links[0]
                        logger.info(f"Found PDF link: {url}")
                    else:
                        raise ValueError(f"No PDF file found at {url}. Please provide a direct link to a PDF file.")

            # Step 2: Download PDF 
            logger.info("Downloading PDF...")
            response = requests.get(url, timeout=30, headers={"User-Agent": "Lobster AI Research Tool/1.0"})
            response.raise_for_status()

            # Step 3: Validate PDF content type 
            content_type = response.headers.get("Content-Type", "").lower()
            if "application/pdf" not in content_type and not response.content.startswith(b"%PDF"):
                raise ValueError(f"The URL did not return a valid PDF file. Content type: {content_type}")

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
                raise ValueError("The PDF file did not contain any extractable text. It may be an image-based PDF requiring OCR.")

            # Cache the result (LOBSTER ENHANCEMENT)
            cache_file.write_text(text, encoding='utf-8')
            logger.info(f"Extracted {len(text)} characters from PDF, cached to {cache_file.name}")

            # Log provenance if DataManager available (LOBSTER ENHANCEMENT)
            if self.data_manager:
                self.data_manager.log_tool_usage(
                    tool_name="extract_pdf_content",
                    parameters={"url": url[:100], "use_cache": use_cache},
                    description=f"PDF extraction: {len(text)} characters"
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
            response = requests.get(url, headers={"User-Agent": "Lobster AI Research Tool/1.0"}, timeout=30)
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
            for element in content(["script", "style", "nav", "header", "footer", "aside", "iframe"]):
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
                    description=f"URL extraction: {len(result)} characters"
                )

            return result

        except Exception as e:
            logger.error(f"Error extracting URL content: {e}")
            raise ValueError(f"Error extracting URL content: {str(e)}")

    def fetch_supplementary_info_from_doi(self, doi: str, output_dir: Optional[str] = None) -> str:
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
                log_message = f"Failed to resolve DOI: {doi}. Status Code: {response.status_code}"
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
                if "supplementary" in text or "supplemental" in text or "appendix" in text:
                    full_url = urljoin(publisher_url, href)
                    supplementary_links.append(full_url)
                    research_log.append(f"Found supplementary material link: {full_url}")

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
                research_log.append(f"Successfully downloaded {len(downloaded_files)} file(s).")
                logger.info(f"Downloaded {len(downloaded_files)} supplementary files for DOI {doi}")
            else:
                research_log.append(f"No files could be downloaded for DOI {doi}.")
                logger.warning(f"No files could be downloaded for DOI {doi}")

            # Log provenance (LOBSTER ENHANCEMENT)
            if self.data_manager:
                self.data_manager.log_tool_usage(
                    tool_name="fetch_supplementary_info_from_doi",
                    parameters={"doi": doi, "output_dir": output_dir},
                    description=f"Downloaded {len(downloaded_files)} supplementary files"
                )

            return "\n".join(research_log)

        except Exception as e:
            logger.error(f"Error fetching supplementary materials for DOI {doi}: {e}")
            research_log.append(f"Error: {str(e)}")
            return "\n".join(research_log)

    def extract_methods_from_paper(
        self,
        url_or_pmid: str,
        llm=None,
        max_text_length: int = 10000
    ) -> Dict[str, Any]:
        """
        Extract computational analysis methods from a research paper using LLM.

        Uses LLM to analyze full paper text and extract:
        - Software/tools used
        - Parameter values and cutoffs
        - Statistical methods
        - Data sources
        - Sample sizes
        - Normalization methods
        - Quality control steps

        Args:
            url_or_pmid: Paper URL or PubMed ID (PMID)
            llm: LLM instance for analysis (uses default if None)
            max_text_length: Maximum text length to send to LLM (default: 10000)

        Returns:
            Dictionary with structured method extraction

        Examples:
            >>> service = PublicationIntelligenceService()
            >>> methods = service.extract_methods_from_paper("PMID:12345678")
            >>> print(methods['software_used'])
            ['Scanpy', 'Seurat', 'DESeq2']
        """
        logger.info(f"Extracting methods from: {url_or_pmid}")

        try:
            # Step 1: Get full text
            if url_or_pmid.startswith("PMID:") or url_or_pmid.isdigit():
                # For PMID, we need to construct a URL
                # This is simplified - in production, use PubMed API to get actual PDF URL
                pmid = url_or_pmid.replace("PMID:", "")
                logger.warning(f"PMID resolution not fully implemented. Using PMID: {pmid}")
                # For now, raise an error suggesting to use direct URL
                raise ValueError(
                    f"Please provide a direct PDF URL instead of PMID. "
                    f"PMID resolution requires additional API integration."
                )
            else:
                url = url_or_pmid

            # Extract text from PDF
            full_text = self.extract_pdf_content(url)

            # Truncate if too long
            text_to_analyze = full_text[:max_text_length]
            if len(full_text) > max_text_length:
                logger.info(f"Truncated text from {len(full_text)} to {max_text_length} characters")

            # Step 2: Use LLM to extract structured methods
            if llm is None:
                from lobster.config.llm_factory import create_llm
                from lobster.config.settings import get_settings
                settings = get_settings()
                model_params = settings.get_agent_llm_params("method_expert_agent")
                llm = create_llm("method_expert_agent", model_params)

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

Paper text (first {max_text_length} characters):
{text_to_analyze}

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
            import json
            try:
                # Extract JSON from markdown code blocks if present
                json_match = re.search(r'```(?:json)?\n(.*?)\n```', response.content, re.DOTALL)
                if json_match:
                    methods = json.loads(json_match.group(1))
                else:
                    # Try to parse entire response as JSON
                    methods = json.loads(response.content)
            except json.JSONDecodeError:
                logger.warning("Could not parse LLM response as JSON, returning raw text")
                methods = {"raw_extraction": response.content}

            logger.info(f"Extracted methods with fields: {list(methods.keys())}")

            # Log provenance
            if self.data_manager:
                self.data_manager.log_tool_usage(
                    tool_name="extract_methods_from_paper",
                    parameters={"source": url_or_pmid[:100], "max_text_length": max_text_length},
                    description=f"Method extraction: {list(methods.keys())}"
                )

            return methods

        except Exception as e:
            logger.error(f"Error extracting methods from paper: {e}")
            raise ValueError(f"Error extracting methods from paper: {str(e)}")
