# Bulk RNA-seq Analysis Tutorial

This comprehensive tutorial demonstrates how to perform bulk RNA-seq differential expression analysis using Lobster AI with pyDESeq2 integration, formula-based experimental design, and publication-ready results.

## Overview

In this tutorial, you will learn to:
- Load and process bulk RNA-seq count matrices
- Design complex experimental formulas using R-style syntax
- Perform differential expression analysis with pyDESeq2
- Handle batch effects and complex designs
- Create publication-quality visualizations
- Export results for downstream analysis

## Prerequisites

- Lobster AI installed and configured (see [Installation Guide](02-installation.md))
- API keys set up in your `.env` file
- Understanding of experimental design concepts
- Familiarity with RNA-seq data formats

## Tutorial Dataset

We'll use **GSE123456** (example dataset), a bulk RNA-seq experiment with:
- 24 samples (12 treatment, 12 control)
- 2 batches (sequencing runs)
- 2 time points (early, late response)
- ~20,000 genes quantified
- Complex design: `~condition + batch + time + condition:time`

## Step 1: Starting Your Analysis

Start Lobster AI and prepare for bulk RNA-seq analysis:

```bash
# Start Lobster AI with enhanced CLI
lobster chat
```

Welcome screen with professional interface:
```
🦞 LOBSTER by Omics-OS
Multi-Agent Bioinformatics Analysis System v0.2

Ready for bulk RNA-seq differential expression analysis!
```

## Step 2: Data Loading and Inspection

### Option A: Loading Kallisto/Salmon Quantification Files (Recommended)

**⚠️ NEW in v0.2+**: Quantification files are now loaded directly via CLI `/read` command (no longer requires agent interaction).

Load per-sample quantification files using the CLI:

```bash
/read /path/to/kallisto_output
```

**Or for Salmon**:
```bash
/read /path/to/salmon_output
```

**Expected Directory Structure**:
```
quantification_output/
├── sample1/
│   └── abundance.tsv  (or quant.sf for Salmon)
├── sample2/
│   └── abundance.tsv
└── sample3/
    └── abundance.tsv
```

**Expected Output:**
```
📁 Detected quantification directory with Kallisto files
🔍 Auto-detected: Kallisto format
📊 Merging 24 sample files...

✅ Successfully loaded quantification data!

📊 Quantification Tool: Kallisto
📁 Source Directory: /path/to/kallisto_output
📈 Data Shape: 24 samples × 19,847 genes
💾 Modality Created: kallisto_output

Sample IDs:
control_rep1, control_rep2, treatment_rep1, treatment_rep2...

✅ Ready for analysis!
Use "show me the data status" to view loaded datasets.
```

After loading, use natural language to interact with the agent:

```bash
🦞 You: "Show me the data status and run quality control"
```

**Expected Agent Response:**
```
🦞 Lobster: I'll check the quantification data and run QC...

📊 Bulk RNA-seq modality 'kallisto_output' ready for analysis:
- Shape: 24 samples × 19,847 genes
- Sample metadata: sample_id, source...
- Quality metrics available

Next Steps:
1. Quality assessment
2. Filtering and normalization
3. Differential expression analysis
```

**Key Features**:
- **Direct CLI Loading**: Use `/read` command - no need to ask the agent
- **Automatic Tool Detection**: CLI detects Kallisto vs Salmon automatically
- **Per-Sample Merging**: Merges abundance.tsv files from each sample subdirectory
- **Correct Orientation**: Automatically transposes to samples × genes format
- **Sample Names**: Extracted from subdirectory names
- **Quality Validation**: Verifies file integrity and consistency

### Option B: Loading Count Matrices (Traditional)

Load pre-computed count matrix and sample metadata:

```bash
🦞 You: "Load the bulk RNA-seq count matrix from counts.csv and sample metadata from metadata.csv. The data has treatment vs control samples with batch effects."
```

**Expected Output:**
```
🦞 Lobster: I'll load your bulk RNA-seq data and inspect the experimental design...

✓ Loaded count matrix: 24 samples × 19,847 genes
✓ Loaded sample metadata with experimental factors
✓ Detected design factors: condition, batch, time_point
✓ Count matrix format validated (integer counts)
✓ Sample-metadata correspondence verified
✓ Stored as modality: 'bulk_rnaseq_counts'

Experimental Design Summary:
- Conditions: treatment (n=12), control (n=12)
- Batches: batch1 (n=12), batch2 (n=12)
- Time points: early (n=12), late (n=12)
- Design matrix: 24 samples × 4 conditions
- Balanced design: ✓ (equal samples per group)
```

**Inspect the loaded data:**
```bash
🦞 You: "/data"
```

Shows dataset overview including sample counts, design factors, and data quality metrics.

## Step 3: Quality Control Assessment

Assess data quality and detect potential issues:

```bash
🦞 You: "Perform quality control analysis including sample correlation, batch effect detection, and count distribution assessment"
```

**Expected Output:**
```
🦞 Lobster: I'll perform comprehensive quality control for your bulk RNA-seq data...

✓ Sample correlation analysis completed
✓ Batch effect assessment performed
✓ Count distribution analysis finished
✓ Library size normalization factors calculated
✓ Gene filtering recommendations generated
✓ Generated QC visualization dashboard

Quality Control Results:
- Library sizes: 15.2M - 28.7M counts (good range)
- Sample correlations: 0.85 - 0.98 (excellent)
- Batch effect detected: MODERATE (batch explains 12% variance)
- Low-count genes: 3,247 genes (< 10 counts total)
- Recommended filtering: Remove genes with < 1 CPM in < 3 samples
- Overall data quality: EXCELLENT
```

**Generated QC visualizations:**
```bash
🦞 You: "/plots"
```

QC plots include:
- `sample_correlation_heatmap.html` - Sample-to-sample correlations
- `library_size_distribution.html` - Count distribution per sample
- `pca_batch_effects.html` - PCA showing batch and treatment effects
- `count_density_plots.html` - Count distribution analysis

## Step 4: Experimental Design and Formula Construction

Create the design matrix using R-style formulas:

```bash
🦞 You: "Create a design matrix using the formula ~condition + batch + time_point + condition:time_point to model treatment effects, batch effects, and time-dependent responses"
```

**Expected Output:**
```
🦞 Lobster: I'll construct the design matrix using your experimental formula...

✓ Formula parsed: ~condition + batch + time_point + condition:time_point
✓ Design matrix constructed: 24 samples × 6 coefficients
✓ Factor levels identified and ordered
✓ Interaction terms calculated
✓ Model rank and identifiability verified
✓ Contrast matrix prepared for hypothesis testing

Design Matrix Structure:
- Intercept: baseline (control, batch1, early)
- conditiontreatment: main treatment effect
- batchbatch2: batch correction term
- time_pointlate: main time effect
- conditiontreatment:time_pointlate: treatment×time interaction
- Full rank: ✓ (all contrasts estimable)

Available Contrasts:
1. Treatment vs Control (main effect)
2. Early vs Late response (time effect)
3. Treatment×Time interaction
4. Treatment effect at early time point
5. Treatment effect at late time point
```

## Step 5: Differential Expression Analysis

Perform differential expression with pyDESeq2:

```bash
🦞 You: "Run differential expression analysis using pyDESeq2 with the design formula. Test for the main treatment effect and the treatment×time interaction."
```

**Expected Output:**
```
🦞 Lobster: I'll run differential expression analysis using pyDESeq2...

✓ pyDESeq2 analysis initialized with design formula
✓ Size factor normalization completed
✓ Dispersion estimation finished (gene-wise and fitted)
✓ Statistical testing performed for specified contrasts
✓ Multiple testing correction applied (Benjamini-Hochberg)
✓ Results tables generated with effect sizes
✓ Stored results as: 'bulk_rnaseq_counts_deseq2_results'

Differential Expression Results:

Main Treatment Effect (Treatment vs Control):
- Total genes tested: 16,600 genes
- Significantly DE genes (FDR < 0.05): 2,847 genes
- Up-regulated: 1,423 genes (log2FC > 0)
- Down-regulated: 1,424 genes (log2FC < 0)
- Largest effect: GENE123 (log2FC = 4.2, FDR = 1.2e-45)

Treatment×Time Interaction:
- Significantly interacting genes (FDR < 0.05): 456 genes
- Strong interactions (|log2FC| > 1): 187 genes
- Early-specific response: 89 genes
- Late-specific response: 98 genes
```

## Step 6: Multiple Contrast Testing

Test specific biological hypotheses:

```bash
🦞 You: "Test additional contrasts: treatment effect at early time point, treatment effect at late time point, and time effect in control samples only"
```

**Expected Output:**
```
🦞 Lobster: I'll test additional biological contrasts using the fitted model...

✓ Custom contrast matrices constructed
✓ Statistical tests performed for each contrast
✓ Effect sizes and confidence intervals calculated
✓ Multiple testing correction applied per contrast
✓ Comparative analysis between contrasts completed

Additional Contrast Results:

Treatment Effect at Early Time Point:
- DE genes (FDR < 0.05): 1,891 genes
- Average log2FC: +0.73 (treatment higher)
- Top pathway: Immune Response (p = 2.3e-12)

Treatment Effect at Late Time Point:
- DE genes (FDR < 0.05): 3,204 genes
- Average log2FC: +1.15 (stronger response)
- Top pathway: Cell Cycle (p = 5.1e-18)

Time Effect in Control Samples:
- DE genes (FDR < 0.05): 567 genes
- Average log2FC: +0.34 (modest changes)
- Top pathway: Circadian Rhythm (p = 1.8e-8)

Key Finding: Treatment effect amplifies over time (interaction significant)
```

## Step 7: Publication-Quality Visualizations

Generate comprehensive visualization suite:

```bash
🦞 You: "Create publication-quality visualizations including volcano plots, MA plots, heatmaps of top DE genes, and PCA plots colored by experimental factors"
```

**Expected Output:**
```
🦞 Lobster: I'll create a comprehensive visualization suite...

✓ Volcano plots generated for all contrasts
✓ MA plots created showing fold-change vs expression
✓ Expression heatmaps for top DE genes created
✓ PCA analysis with factor coloring completed
✓ Sample clustering dendrogram generated
✓ Interactive dashboards created for data exploration
✓ All plots optimized for publication (300 DPI, proper fonts)

Generated Visualizations:
- volcano_treatment_main.html: Volcano plot for main treatment effect
- volcano_interaction.html: Volcano plot for treatment×time interaction
- ma_plot_treatment.html: MA plot showing fold-change distribution
- heatmap_top_genes.html: Expression heatmap (top 100 DE genes)
- pca_experimental_factors.html: PCA colored by condition and batch
- sample_clustering.html: Hierarchical clustering of samples
- gene_expression_profiles.html: Expression trajectories over time
- comprehensive_dashboard.html: Interactive multi-panel dashboard
```

## Step 8: Pathway and Functional Analysis

Perform pathway enrichment analysis:

```bash
🦞 You: "Run pathway enrichment analysis on the up-regulated and down-regulated genes using Gene Ontology and KEGG pathways"
```

**Expected Output:**
```
🦞 Lobster: I'll perform pathway enrichment analysis on your DE gene lists...

✓ Gene set enrichment analysis completed
✓ GO Biological Process enrichment performed
✓ KEGG pathway analysis finished
✓ Reactome pathway analysis completed
✓ Enrichment statistics calculated (hypergeometric test)
✓ Multiple testing correction applied (FDR)
✓ Pathway visualization networks generated

Pathway Enrichment Results:

Up-regulated Genes (Treatment > Control):
Top GO Terms:
1. Immune system process (GO:0002376) - 134 genes, FDR = 2.1e-23
2. Response to stimulus (GO:0050896) - 298 genes, FDR = 5.7e-19
3. Cell activation (GO:0001775) - 89 genes, FDR = 1.2e-16

Top KEGG Pathways:
1. Cytokine-cytokine receptor interaction (hsa04060) - 45 genes, FDR = 3.4e-12
2. T cell receptor signaling pathway (hsa04660) - 32 genes, FDR = 8.9e-9
3. NF-kappa B signaling pathway (hsa04064) - 28 genes, FDR = 2.1e-8

Down-regulated Genes (Treatment < Control):
Top GO Terms:
1. Metabolic process (GO:0008152) - 287 genes, FDR = 1.7e-15
2. Cellular biosynthetic process (GO:0044249) - 198 genes, FDR = 4.2e-12
3. Gene expression (GO:0010467) - 156 genes, FDR = 7.8e-11
```

## Step 9: Results Export and Reporting

Export comprehensive results:

```bash
🦞 You: "/export bulk-rnaseq-results"
```

**Expected Output:**
```
🦞 Lobster: Exporting comprehensive bulk RNA-seq analysis results...

✓ DE results tables exported (CSV format)
✓ Normalized counts matrix exported
✓ Sample metadata with QC metrics exported
✓ Statistical summaries generated
✓ Pathway enrichment results exported
✓ Analysis parameters logged
✓ Publication-ready figures exported (SVG + PNG)

Exported Files:
📊 Data Files:
- de_results_main_treatment.csv: Main treatment effect results
- de_results_interaction.csv: Treatment×time interaction results
- normalized_counts_matrix.csv: Normalized expression values
- sample_metadata_qc.csv: Sample info with QC metrics

📈 Statistical Summaries:
- analysis_summary.txt: Complete statistical summary
- pathway_enrichment_results.csv: All pathway results
- gene_annotations.csv: Gene symbols and descriptions

🎨 Visualizations:
- figures/publication/: High-resolution publication figures
- figures/interactive/: Interactive HTML plots
- figures/supplementary/: Additional analysis plots

📋 Documentation:
- analysis_parameters.json: All analysis settings
- session_log.txt: Complete analysis history
- methods_description.txt: Analysis methods for publication
```

## Step 10: Advanced Analysis Options

### Gene Set Analysis with Custom Gene Lists

```bash
🦞 You: "Perform gene set enrichment analysis using custom gene sets from literature - analyze if our treatment signature overlaps with known drug response signatures"
```

### Temporal Analysis

```bash
🦞 You: "Create temporal expression profiles showing how gene expression changes over time for both treatment and control conditions"
```

### Batch Effect Correction

```bash
🦞 You: "Apply ComBat batch correction and re-run the differential expression analysis to compare results with and without batch correction"
```

### Integration with External Databases

```bash
🦞 You: "Query the Connectivity Map (CMap) database to identify potential drug compounds that could reverse our treatment signature"
```

## Working with Complex Designs

### Multi-Factor Experiments

For experiments with multiple factors (e.g., treatment, time, cell type):

```bash
🦞 You: "Design a three-way interaction model: ~treatment * time * celltype with proper contrast specification"
```

### Paired Sample Analysis

For paired/matched sample designs:

```bash
🦞 You: "Analyze paired samples using the formula ~patient + condition to account for patient-specific effects"
```

### Time Series Analysis

For temporal experiments with multiple time points:

```bash
🦞 You: "Model time as a continuous variable and identify genes with linear, quadratic, or cubic temporal patterns"
```

## Troubleshooting Common Issues

### Issue 1: Low Count Genes
```bash
🦞 You: "Many genes have very low counts - should I filter them?"
```
**Solution**: Filter genes with < 1 CPM in < 3 samples (or minimum group size).

### Issue 2: Batch Effects Too Strong
```bash
🦞 You: "Batch effects are overwhelming the biological signal"
```
**Solution**: Use RUVSeq or sva for batch correction, or include batch as blocking factor.

### Issue 3: No Significant Genes
```bash
🦞 You: "My analysis found no significantly DE genes"
```
**Solution**: Check sample sizes, effect sizes, and consider less stringent FDR thresholds.

### Issue 4: Design Matrix Issues
```bash
🦞 You: "Getting 'matrix not full rank' error"
```
**Solution**: Check for confounded factors or redundant terms in the design.

## Best Practices

1. **Sample Size**: Minimum n=3 per group, preferably n≥6 for robust results
2. **Quality Control**: Always inspect data before analysis
3. **Multiple Testing**: Use FDR correction for genome-wide testing
4. **Effect Size**: Report fold changes with statistical significance
5. **Validation**: Validate key findings with qRT-PCR or independent datasets
6. **Documentation**: Keep detailed records of analysis parameters

## Integration with Other Analyses

### Convert to Single-Cell Format
```bash
🦞 You: "Simulate single-cell data from bulk RNA-seq for method comparison"
```

### Meta-Analysis
```bash
🦞 You: "Combine my results with published datasets for meta-analysis"
```

### Proteomics Integration
```bash
🦞 You: "Compare my RNA-seq results with proteomics data from the same experiment"
```

## Next Steps

After completing this tutorial, explore:

1. **[Proteomics Tutorial](25-tutorial-proteomics.md)** - Integrate with proteomics data
2. **[Advanced Analysis Cookbook](27-examples-cookbook.md)** - Specialized workflows
3. **[Custom Agent Tutorial](26-tutorial-custom-agent.md)** - Create specialized analysis agents
4. **[Single-Cell Tutorial](23-tutorial-single-cell.md)** - Compare bulk vs single-cell approaches

## Summary

You have successfully:
- ✅ Loaded and quality-controlled bulk RNA-seq data
- ✅ Designed complex experimental models with interactions
- ✅ Performed differential expression with pyDESeq2
- ✅ Tested multiple biological contrasts
- ✅ Generated publication-quality visualizations
- ✅ Performed pathway enrichment analysis
- ✅ Exported comprehensive results for publication
- ✅ Learned advanced analysis strategies

This workflow demonstrates Lobster AI's sophisticated bulk RNA-seq analysis capabilities using natural language interaction with professional-grade statistical methods and visualization tools.