# Glossary

This comprehensive glossary defines bioinformatics, technical, and Lobster AI-specific terms used throughout the documentation and platform.

## A

**Affinity Proteomics**
Protein analysis method using antibody-based assays for targeted protein detection and quantification. Examples include Olink panels and antibody arrays. Typically has lower missing values (<30%) compared to mass spectrometry.

**Agent**
Specialized AI component in Lobster AI that handles specific analysis domains (e.g., Single-cell Expert, Proteomics Expert). Agents use natural language understanding to execute appropriate tools and workflows.

**Agent Factory Function**
Python function that creates and configures an agent instance. Specified in the agent registry for dynamic agent loading. Example: `lobster.agents.singlecell_expert.singlecell_expert`.

**Agent Handoff**
Architectural pattern in Lobster AI where agents transfer tasks to other agents based on analysis requirements. Coordinated by the supervisor agent using handoff tools. Example: supervisor hands off single-cell analysis to `singlecell_expert`.

**Agent Registry**
Centralized system in Lobster AI that manages all available agents, their configurations, and handoff capabilities. Located in `lobster/config/agent_registry.py`. Single source of truth for agent configuration.

**Analysis Step IR (Intermediate Representation)**
Structured representation of analysis operations used for provenance tracking and notebook export. Contains operation details, parameters, code templates, and execution context. Required for reproducibility.

**AnnData**
Annotated data structure used in Python bioinformatics for storing high-dimensional biological data. Contains expression matrix (X), observations (obs), variables (var), and additional metadata.

**Annotation**
Process of assigning biological meaning to data elements, such as identifying cell types in single-cell data or functional categories for genes.

---

## B

**Batch Effect**
Technical variation in data caused by non-biological factors such as processing date, equipment, or experimental conditions. Must be corrected to avoid confounding biological signals.

**Benjamini-Hochberg (BH)**
Multiple testing correction method that controls the False Discovery Rate (FDR). Used to adjust p-values when testing many hypotheses simultaneously.

**Bulk RNA-seq**
RNA sequencing of entire tissue samples or cell populations, providing average expression across all cells. Contrasts with single-cell RNA-seq which measures individual cells.

---

## C

**Capability-Based Routing**
Architecture pattern in ContentAccessService (v0.2+) where providers register their capabilities and are selected based on query requirements. Enables flexible provider selection with priority-based fallback.

**Cell Type Annotation**
Process of identifying and labeling cell populations in single-cell data based on marker gene expression patterns and biological knowledge. Can be automated (CellTypist) or manual (expert curation).

**Clustering**
Computational method to group similar observations (cells, samples, genes) based on expression patterns. Common algorithms include Leiden, Louvain, and k-means.

**Content Caching**
Two-tier caching system in Lobster AI (v0.2+) with session cache (in-memory, temporary) and workspace cache (filesystem, persistent). Provides 30-50x speedup for repeated access to publications and datasets.

**ContentAccessService**
Unified service in Lobster AI (v0.2+) for accessing scientific content through five specialized providers: AbstractProvider, PubMedProvider, GEOProvider, PMCProvider, and WebpageProvider. Implements three-tier cascade (PMC XML → Webpage → PDF) with automatic fallback.

**Coefficient of Variation (CV)**
Statistical measure of relative variability, calculated as standard deviation divided by mean. Used to assess technical reproducibility in proteomics.

**ComBat**
Batch correction method that removes batch effects while preserving biological variation. Commonly used in transcriptomics analysis.

**Count Matrix**
Two-dimensional data structure with features (genes/proteins) as rows and samples (cells/samples) as columns, containing quantified expression values.

---

## D

**DDA (Data-Dependent Acquisition)**
Mass spectrometry method where the instrument selects the most abundant ions for fragmentation and identification. Traditional approach for shotgun proteomics.

**DIA (Data-Independent Acquisition)**
Mass spectrometry method that systematically fragments all ions in predefined windows. Often provides more comprehensive and reproducible protein identification.

**DataManagerV2**
Core data management system in Lobster AI that handles multiple modalities, provenance tracking, and analysis history. Supports various data backends and formats.

**Differential Expression (DE)**
Statistical analysis to identify genes or proteins with significantly different expression levels between conditions or cell types.

**Design Matrix**
Mathematical representation of experimental design used in statistical models. Encodes relationships between samples and experimental factors.

**Docling**
Advanced PDF parsing library in Lobster AI (v0.2+) for extracting scientific content from publications. Achieves >90% Methods section detection rate, handles tables and formulas. Falls back to PyPDF2 (30% detection) when unavailable. Install: `pip install lobster[docling]`.

**Doublet**
Artifact in single-cell RNA-seq where two cells are captured and sequenced together, appearing as a single cell with unusually high gene counts.

**Download Queue**
Orchestration system in Lobster AI for managing multi-agent downloads. Research agent discovers URLs and creates queue entries (PENDING), supervisor polls queue and hands off to data_expert, which downloads and updates status (IN_PROGRESS → COMPLETED/FAILED).

---

## E

**Empirical Bayes**
Statistical method that improves parameter estimation by borrowing information across features. Used in limma and other differential expression tools.

**Enrichment Analysis**
Statistical method to identify biological pathways or processes that are over-represented in a gene or protein list.

---

## F

**False Discovery Rate (FDR)**
Expected proportion of false positives among rejected hypotheses. Commonly controlled at 5% (0.05) in genomics studies.

**Feature Selection**
Process of selecting the most informative variables (genes/proteins) for analysis, such as highly variable genes in single-cell RNA-seq.

**Fold Change**
Ratio of expression between two conditions. Often expressed as log2 fold change where 1 represents 2-fold increase and -1 represents 2-fold decrease.

---

## G

**GEO (Gene Expression Omnibus)**
NCBI database containing high-throughput genomics data. Lobster AI can directly download and analyze GEO datasets using GSE accession numbers.

**GSEA (Gene Set Enrichment Analysis)**
Method for determining whether a defined set of genes shows statistically significant differences between biological states.

---

## H

**H5AD**
HDF5-based file format for storing AnnData objects. Standard format for single-cell data that preserves metadata and analysis results.

**Handoff**
Process in Lobster AI where the supervisor agent transfers a query to a specialized agent based on the analysis type required.

**Highly Variable Genes (HVG)**
Genes showing high variation across cells or samples, often selected for downstream analysis as they contain the most biological information.

---

## I

**Imputation**
Statistical method to estimate missing values in datasets. Particularly important in proteomics where 30-70% of values may be missing.

**Integration**
Process of combining multiple datasets or omics layers to enable joint analysis. Includes batch correction and multi-omics integration.

---

## L

**Label-Free Quantification (LFQ)**
Mass spectrometry approach that quantifies proteins without chemical labeling, using peptide intensity measurements.

**Leiden Clustering**
Community detection algorithm for clustering that often performs better than Louvain clustering for biological data.

**Limma**
R package for analyzing gene expression data using linear models and empirical Bayes methods. Popular for differential expression analysis.

**Log Transformation**
Mathematical transformation that converts multiplicative relationships to additive ones. Common preprocessing step in genomics (log2 or natural log).

---

## M

**MA Plot**
Scatter plot showing log fold change (M) vs average expression (A). Used to visualize differential expression results and identify bias.

**Manual Curation Workflow**
Interactive workflow in Lobster AI for expert-guided cell type annotation, metadata harmonization, and quality assessment. Combines automated suggestions with domain expertise for improved biological accuracy.

**Marker Genes**
Genes specifically or highly expressed in particular cell types or conditions. Used for cell type identification and validation.

**MaxQuant**
Software platform for analyzing mass spectrometry proteomics data. Provides protein identification and quantification from raw MS data.

**Missing at Random (MAR)**
Missing data mechanism where missingness depends on observed data but not on the missing values themselves.

**Missing Not at Random (MNAR)**
Missing data mechanism where missingness depends on the unobserved values. Common in proteomics for low-abundance proteins.

**Modality**
In Lobster AI, a named dataset representing a specific omics measurement (e.g., "single_cell_rna", "ms_proteomics"). Managed by DataManagerV2.

**Moran's I**
Statistical measure of spatial autocorrelation used in spatial omics to identify spatially variable genes or proteins.

**MuData**
Data structure for storing and analyzing multi-modal omics data. Extends AnnData to handle multiple measurement types simultaneously.

---

## N

**Normalization**
Process of adjusting data to remove technical variation and enable comparison between samples. Methods include library size normalization, quantile normalization, and TMM.

**Notebook Export**
Feature in Lobster AI that exports complete analysis workflows as reproducible Jupyter notebooks. Uses Papermill for parameterization, includes provenance metadata, code snippets, and execution instructions. Access via `/pipeline export` command.

**NPX (Normalized Protein eXpression)**
Log2-transformed and normalized protein abundance values used in Olink affinity proteomics assays. Already log2-scaled, so should not be log-transformed again during preprocessing.

---

## O

**Olink**
Commercial platform for targeted proteomics using proximity extension assays (PEA). Provides high-quality protein measurements with low missing values.

**Ontology**
Structured vocabulary defining relationships between biological concepts. Gene Ontology (GO) is widely used for functional annotation.

---

## P

**PCA (Principal Component Analysis)**
Dimensionality reduction technique that identifies the directions of maximum variance in data. Used for visualization and quality control.

**PDB (Protein Data Bank)**
Public repository of 3D structural data for proteins and nucleic acids. Lobster AI (v0.2+) fetches structures using 4-character PDB IDs (e.g., "1AKE") and links them to gene expression data. Supports both PDB and mmCIF formats.

**PMC (PubMed Central)**
Free full-text archive of biomedical literature. Lobster AI (v0.2+) uses PMC XML API as priority source for publication content (500ms-2s response time, 30-40% coverage). Part of ContentAccessService three-tier cascade.

**Protein Structure Visualization**
Feature in Lobster AI (v0.2+) for visualizing protein 3D structures using PyMOL. Supports interactive mode (GUI) and batch mode (PNG generation), residue highlighting, and automatic structure-to-gene linking. Install: `make install-pymol`.

**Provider**
Component in ContentAccessService (v0.2+) that implements access to specific content sources. Five providers: AbstractProvider (abstracts, fast), PubMedProvider (literature search), GEOProvider (dataset discovery), PMCProvider (full-text, priority), WebpageProvider (fallback, PDF support).

**Pseudobulk**
Method to aggregate single-cell data to sample-level summaries, enabling population-level statistical analysis with established bulk RNA-seq methods. Recommended for differential expression testing across conditions.

**Publication Intelligence**
Automated system in Lobster AI (v0.2+) for extracting scientific methods and parameters from publications. Uses Docling for PDF parsing, ContentAccessService for content access, and structured schemas for metadata extraction.

**Pseudotime**
Computational measure of cell progression along a biological process, such as differentiation or cell cycle, based on expression similarity.

**pyDESeq2**
Pure Python implementation of the DESeq2 algorithm for differential expression analysis of RNA-seq data.

**PyMOL**
Professional molecular visualization system for protein structures. Integrated into Lobster AI (v0.2+) for 3D structure visualization. Supports both interactive GUI mode and headless batch mode (PNG generation). Install: `make install-pymol` or `brew install brewsci/bio/pymol`.

---

## Q

**Quality Control (QC)**
Assessment of data quality including technical metrics, batch effects, and sample integrity. Essential first step in any omics analysis.

**Quantile Normalization**
Normalization method that makes the distribution of values identical across samples by matching quantiles.

---

## R

**RNA Velocity**
Method to predict future cell states by analyzing spliced and unspliced mRNA ratios, revealing cell differentiation dynamics.

---

## S

**S3 Backend**
Cloud storage integration in Lobster AI (v0.2+) for storing analysis data and results in AWS S3. Provides scalable storage for large datasets with seamless switching between local and cloud storage. Requires AWS credentials configuration.

**scanpy**
Python package for analyzing single-cell gene expression data. Provides comprehensive tools for preprocessing, visualization, and analysis.

**Service**
In Lobster AI architecture, stateless components that perform specific analysis tasks. Services receive AnnData objects and return results with statistics.

**Session Export**
Feature in Lobster AI that exports complete analysis session including conversation history, loaded modalities, workspace content, generated plots, and provenance metadata. Access via `/export session` command. Enables session restoration and sharing.

**Supervisor Agent**
Central coordinator agent in Lobster AI that routes user queries to specialized agents based on analysis requirements. Manages agent handoffs, monitors download queue, and orchestrates multi-agent workflows. Entry point for all user interactions.

**Silhouette Score**
Measure of clustering quality that quantifies how similar objects are within clusters compared to other clusters.

**Single-cell RNA-seq (scRNA-seq)**
Sequencing technology that measures gene expression in individual cells, revealing cellular heterogeneity and rare cell types.

**Spatial Omics**
Analysis of molecular data with preserved spatial context, such as Visium spatial transcriptomics or imaging mass cytometry.

---

## T

**t-SNE**
Non-linear dimensionality reduction method that preserves local structure. Often used for visualizing high-dimensional single-cell data.

**Three-Tier Cascade**
Content access strategy in ContentAccessService (v0.2+) that attempts multiple access methods with automatic fallback: (1) PMC XML API (priority, 500ms-2s, 30-40% coverage), (2) Webpage/PDF extraction (fallback, 2-8s, 60-70% coverage), (3) Error with alternative suggestions. Maximizes content accessibility.

**TMM (Trimmed Mean of M-values)**
Normalization method commonly used in proteomics that assumes most proteins are not differentially expressed.

**Trajectory Analysis**
Computational method to order cells along developmental or temporal progressions, revealing biological processes over time.

---

## U

**UMAP (Uniform Manifold Approximation and Projection)**
Dimensionality reduction and visualization technique that preserves both local and global data structure. Popular for single-cell visualization.

**UMI (Unique Molecular Identifier)**
Short DNA sequence used to identify and count individual mRNA molecules, reducing PCR bias in single-cell sequencing.

---

## V

**Variance Stabilization**
Transformation that makes variance approximately constant across the range of data values. Important for statistical analysis assumptions.

**Visium**
10x Genomics platform for spatial gene expression analysis that measures transcriptomes with preserved spatial coordinates.

**Volcano Plot**
Scatter plot showing fold change vs statistical significance (-log10 p-value) for differential expression results.

---

## W

**WGCNA (Weighted Gene Co-expression Network Analysis)**
Method for constructing gene co-expression networks and identifying modules of highly correlated genes.

---

## Technical Terms (Lobster AI Specific)

**CLI (Command Line Interface)**
Lobster's Rich-enhanced terminal interface with orange branding, autocomplete, and real-time monitoring capabilities.

**Cloud Mode**
Operating mode where analyses are processed on cloud infrastructure. Activated by setting `LOBSTER_CLOUD_KEY` environment variable.

**Handoff Tool**
Specialized tool registered in agent registry that allows supervisor agent to transfer tasks to specialist agents based on analysis requirements. Enables seamless multi-agent coordination with context preservation.

**LangGraph**
Framework used by Lobster AI for creating multi-agent workflows with state management and tool integration.

**Local Mode**
Default operating mode where all analyses are processed on the user's local machine.

**Orange Theming**
Lobster AI's signature color scheme (#e45c47) used throughout the CLI interface and visualizations.

**Provenance Tracking**
W3C-PROV compliant system in Lobster AI that records complete analysis history for reproducibility.

**Rich CLI**
Enhanced command-line interface using the Rich Python library, providing advanced formatting, progress bars, and interactive elements.

**Service Pattern**
Lobster AI's architectural pattern where stateless services handle analysis logic, returning both processed data and statistics.

**Tool**
Function available to agents for performing specific tasks. Tools follow standardized patterns for data validation, service calls, and result storage.

**Workspace**
Directory structure used by Lobster AI to organize data, results, and analysis history. Default location is `.lobster_workspace/`. Contains three subdirectories: `literature/` (publications), `data/` (datasets), `metadata/` (custom content).

**WorkspaceContentService**
Type-safe caching system in Lobster AI (v0.2+) for persistent storage of research content. Uses Pydantic schemas (PublicationContent, DatasetContent, MetadataContent) with enum-based validation (ContentType, RetrievalLevel). Provides identifier-based access to cached publications and datasets across sessions.

---

## Statistical Terms

**Adjusted Rand Score**
Measure of clustering performance that compares clustering results to known ground truth, corrected for chance.

**Cohen's d**
Standardized measure of effect size representing the difference between groups in terms of standard deviations.

**Empirical Bayes Moderation**
Statistical technique that improves variance estimates by borrowing information across genes, leading to more stable results.

**Hypergeometric Test**
Statistical test used in enrichment analysis to determine if a pathway or gene set is over-represented in a list of interesting genes.

**Multiple Testing Correction**
Statistical adjustment applied when testing multiple hypotheses simultaneously to control the overall error rate.

**Pearson Correlation**
Measure of linear correlation between two variables, ranging from -1 (perfect negative correlation) to 1 (perfect positive correlation).

**Spearman Correlation**
Rank-based correlation measure that captures monotonic relationships, more robust to outliers than Pearson correlation.

**Wilcoxon Rank-Sum Test**
Non-parametric statistical test used to compare expression levels between groups. Default method for single-cell differential expression.

---

This glossary provides definitions for terms commonly encountered in Lobster AI documentation and bioinformatics analysis. For additional technical details, refer to the specific tutorial documents and API documentation.