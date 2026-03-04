# lobster-ml

Machine learning and deep learning for biological data analysis and framework export.

## Installation

```bash
# Basic installation
pip install lobster-ml

# With deep learning dependencies (scVI, PyTorch)
pip install lobster-ml[ml]
```

## Agents

| Agent | Description |
|-------|-------------|
| `machine_learning_expert` | ML specialist for biological data. Feature engineering, data splitting, framework export, and deep learning embeddings. |

## Services

| Service | Purpose |
|---------|---------|
| MLPreparationService | Feature selection, scaling, and train/test/validation splitting |
| MLTranscriptomicsServiceALPHA | Transcriptomics-specific ML workflows (ALPHA) |
| MLProteomicsServiceALPHA | Proteomics-specific ML workflows (ALPHA) |
| scVIEmbeddingService | Deep learning embeddings using scVI for single-cell data |

## Features

### ML Readiness Assessment
- Evaluate biological datasets for machine learning suitability
- Check sample size, class balance, and feature quality
- Identify potential data leakage and batch effects
- Recommend preprocessing steps before ML pipeline

### Feature Engineering
- Highly variable gene selection for dimensionality reduction
- PCA-based feature extraction with variance thresholds
- Marker gene features from differential expression
- Z-score normalization and scaling

### Data Splitting
- Stratified train/test/validation splits
- Configurable split ratios (default: 70/15/15)
- Class balance preservation across splits
- Batch-aware splitting to prevent data leakage

### Framework Export
- NumPy arrays for scikit-learn workflows
- CSV export for general ML frameworks
- PyTorch tensor datasets with DataLoader support
- TensorFlow NPZ format for Keras models

### Deep Learning Embeddings
- scVI integration for variational autoencoder embeddings
- Latent space visualization and clustering
- Transfer learning from pre-trained models
- GPU acceleration when available

## Requirements

- Python 3.12+
- lobster-ai >= 1.0.0
- Optional: torch, scvi-tools (for deep learning features)

## Tier Requirement

This is a **premium** agent. Access is controlled at runtime via Lobster AI's tier system.

## Documentation

Full documentation: [docs.omics-os.com/docs/agents/ml](https://docs.omics-os.com/docs/agents/ml)

## License

MIT
