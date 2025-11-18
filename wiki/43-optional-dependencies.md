# Optional Dependencies Guide

This guide covers optional software components that enhance Lobster AI with specialized capabilities. None of these are required for basic functionality, but they unlock advanced features for specific use cases.

## Table of Contents

- [Overview](#overview)
- [PyMOL - Protein Structure Visualization](#pymol---protein-structure-visualization)
- [Docling - Advanced PDF Parsing](#docling---advanced-pdf-parsing)
- [System Libraries by Platform](#system-libraries-by-platform)
- [Testing Optional Dependencies](#testing-optional-dependencies)

---

## Overview

Lobster AI works out-of-the-box for most bioinformatics workflows. Optional dependencies add capabilities for specialized analyses:

| Dependency | Purpose | When Needed | Installation Effort |
|------------|---------|-------------|-------------------|
| **PyMOL** | 3D protein structure visualization | Protein structure analysis, linking to omics data | Medium (macOS/Linux), High (Windows) |
| **Docling** | Advanced PDF parsing | Extracting methods from complex publications | Low (pip install) |
| **System Libraries** | Compilation support | Native installation on Linux | Low (apt/dnf install) |

**Installation Strategy:**
1. Start with core Lobster AI installation
2. Add optional dependencies as needed for your analyses
3. Use Docker if optional dependencies are difficult to install natively

---

## PyMOL - Protein Structure Visualization

### What is PyMOL?

PyMOL is an industry-standard molecular visualization system for displaying and analyzing 3D protein structures. Lobster AI integrates with PyMOL to:

- Fetch protein structures from PDB database
- Visualize structures with customizable styles
- Highlight specific residues or mutations
- Link protein structures to omics data (e.g., RNA-seq expression levels)
- Generate publication-quality structure images

**Version Required**: PyMOL 2.4+

### When Do You Need PyMOL?

PyMOL is optional but recommended if you:
- Analyze protein-coding genes and want to visualize their 3D structures
- Study mutations and their structural context
- Need to link transcriptomics/proteomics data to protein structure
- Create figures showing protein structure for publications

**Without PyMOL**, Lobster AI can still:
- Perform all RNA-seq and proteomics analyses
- Download sequence data
- Run differential expression and enrichment
- Fetch protein structure metadata

### Installation

#### macOS

**Option 1: Automated (Recommended)**
```bash
cd lobster
make install-pymol
```

**Option 2: Homebrew**
```bash
# Add homebrew-science tap
brew install brewsci/bio/pymol

# Verify installation
pymol -c -Q
which pymol
```

**Option 3: Open-Source Build**
```bash
# Install dependencies
brew install glew glm freetype libpng python@3.12

# Build from source (advanced)
git clone https://github.com/schrodinger/pymol-open-source.git
cd pymol-open-source
python setup.py build install
```

#### Linux (Ubuntu/Debian)

**Option 1: Package Manager (Easy)**
```bash
sudo apt-get update
sudo apt-get install pymol

# Verify
which pymol
pymol --version
```

**Option 2: Conda/Mamba**
```bash
# If you use conda/mamba environments
conda install -c conda-forge pymol-open-source

# Or with mamba (faster)
mamba install -c conda-forge pymol-open-source
```

**Option 3: Build from Source**
```bash
# Install dependencies
sudo apt-get install build-essential python3-dev \
    libglew-dev libpng-dev libfreetype6-dev \
    libxml2-dev libmsgpack-dev python3-pyqt5.qtopengl

# Clone and build
git clone https://github.com/schrodinger/pymol-open-source.git
cd pymol-open-source
python setup.py build install --prefix=$HOME/.local
```

#### Linux (Fedora/RHEL/CentOS)

```bash
# Enable EPEL repository (if needed)
sudo dnf install epel-release

# Install PyMOL
sudo dnf install pymol

# Verify
pymol --version
```

#### Windows

**⚠️ PyMOL installation on Windows is complex. We recommend using:**

**Option 1: Docker (Recommended)**
- PyMOL is pre-installed in Lobster Docker images
- No manual setup needed
- Run: `docker-compose run --rm lobster-cli`

**Option 2: Windows Subsystem for Linux (WSL)**
- Install WSL 2 with Ubuntu
- Follow Linux installation instructions above
- Requires X11 server (VcXsrv or Xming) for GUI

**Option 3: Commercial PyMOL**
- Purchase from [pymol.org](https://pymol.org/)
- Windows installer included
- Educational licenses available

**Option 4: Conda (Windows)**
```powershell
# Install Miniconda if not already installed
# Download from: https://docs.conda.io/en/latest/miniconda.html

# Create environment with PyMOL
conda create -n pymol-env python=3.12
conda activate pymol-env
conda install -c conda-forge pymol-open-source
```

### Verification

After installation, verify PyMOL works:

```bash
# Test command-line mode
pymol -c -Q -d "fetch 1AKE; quit"

# Check version
pymol --version

# Test from Lobster
lobster chat
🦞 You: /status
# Should show: "PyMOL: Available (version X.X.X)"
```

### Usage in Lobster

Once installed, PyMOL integrates seamlessly:

```bash
# Start Lobster
lobster chat

# Fetch and visualize protein structure
🦞 You: "Fetch protein structure 1AKE"
🦞 You: "Visualize 1AKE with cartoon representation"

# Link to omics data
🦞 You: "Show expression levels of ADK gene on 1AKE structure"

# Advanced styling
🦞 You: "Highlight residues 50-100 on 1AKE structure"
🦞 You: "Color 1AKE by conservation score"
```

### Troubleshooting PyMOL

**Issue**: `pymol: command not found`

**Solutions:**
```bash
# Check if installed
which pymol
dpkg -l | grep pymol  # Ubuntu/Debian
rpm -qa | grep pymol  # Fedora/RHEL

# Add to PATH (if installed but not found)
export PATH=$PATH:$HOME/.local/bin
echo 'export PATH=$PATH:$HOME/.local/bin' >> ~/.bashrc

# Reinstall
sudo apt-get install --reinstall pymol
```

**Issue**: `ImportError: No module named pymol`

**Solutions:**
- Ensure virtual environment is activated
- PyMOL must be installed in same Python environment as Lobster
- Try installing via conda in the same environment

**Issue**: Graphics/OpenGL errors

**Solutions:**
```bash
# Test command-line mode (no GUI)
pymol -c -Q

# On remote servers, use headless mode
export DISPLAY=:0
Xvfb :0 -screen 0 1024x768x24 &
```

See [Protein Structure Visualization Guide](40-protein-structure-visualization.md) for complete usage details.

---

## Docling - Advanced PDF Parsing

### What is Docling?

Docling is a professional PDF parsing library that excels at extracting structured content from scientific publications. It provides:

- **>90% accuracy** for Methods section detection (vs 30% with basic parsers)
- **Table extraction** from complex multi-column layouts
- **Formula recognition** and LaTeX conversion
- **Figure caption extraction** with context
- **Multi-language support** for international publications

**Version Required**: Docling 2.60+

### When Do You Need Docling?

Docling is optional but highly recommended if you:
- Frequently extract analysis parameters from publications
- Work with complex, multi-column scientific PDFs
- Need to extract tables or figures programmatically
- Analyze methods from large sets of papers

**Without Docling**, Lobster AI falls back to PyPDF2:
- Basic text extraction works
- Methods section detection ~30% accurate
- No table or formula extraction
- Simple single-column PDFs work fine

### Installation

Docling is a Python package, easy to install:

**Basic Installation:**
```bash
# Activate Lobster virtual environment
source .venv/bin/activate

# Install Docling
pip install docling
```

**Full Installation (All Features):**
```bash
# With all optional features
pip install "docling[all]"

# With specific features
pip install "docling[table]"  # Table extraction
pip install "docling[ocr]"    # OCR support
```

**Docker:**
Docling is pre-installed in Lobster Docker images - no additional setup needed.

### Verification

```bash
# Test import
python -c "from docling.document_converter import DocumentConverter; print('✅ Docling installed')"

# Check version
python -c "import docling; print(docling.__version__)"

# Test in Lobster
lobster chat
🦞 You: /status
# Should show: "Docling: Available (version X.X.X)"
```

### Usage in Lobster

Docling works automatically when installed:

```bash
lobster chat

# Extract methods from publication
🦞 You: "Extract methods from PMID:38448586"

# With Docling: Returns detailed Methods section, parameters, tables
# Without Docling: Returns basic text extraction

# Read full publication
🦞 You: "Read full text of PMID:35042229"

# Extract from local PDF
🦞 You: "Extract methods from paper.pdf in my workspace"
```

### Troubleshooting Docling

**Issue**: Import errors after installation

**Solutions:**
```bash
# Ensure in correct environment
source .venv/bin/activate

# Reinstall
pip uninstall docling
pip install --no-cache-dir "docling[all]"

# Check dependencies
pip list | grep docling
```

**Issue**: Memory errors with large PDFs

**Solutions:**
- Increase available RAM
- Process PDFs in smaller batches
- Use cloud mode for large-scale extraction

**Issue**: Poor extraction quality

**Solutions:**
- Ensure PDF is text-based (not scanned image)
- For scanned PDFs, install OCR support: `pip install "docling[ocr]"`
- Try different extraction modes in Docling settings

See [Publication Intelligence Guide](37-publication-intelligence-deep-dive.md) for advanced usage.

---

## System Libraries by Platform

### macOS

Most dependencies handled by Xcode Command Line Tools:

```bash
# Install Xcode tools (if not already installed)
xcode-select --install

# Optional: Homebrew for easier package management
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install HDF5 (optional, for larger datasets)
brew install hdf5
```

### Ubuntu/Debian

Required for native Python package compilation:

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    python3.12-dev \
    libhdf5-dev \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    libssl-dev \
    libblas-dev \
    liblapack-dev
```

**Why These Are Needed:**
- **build-essential**: gcc, g++, make compilers
- **python3.12-dev**: Python header files for C extensions
- **libhdf5-dev**: HDF5 file format (AnnData, MuData)
- **libblas/liblapack-dev**: Linear algebra (NumPy, SciPy)
- **libxml2/libxslt-dev**: XML parsing (web scraping, GEO)
- **libffi/libssl-dev**: Cryptography and foreign function interface

### Fedora/RHEL/CentOS

```bash
sudo dnf install -y \
    gcc gcc-c++ make \
    python3.12-devel \
    hdf5-devel \
    libxml2-devel \
    libxslt-devel \
    openssl-devel \
    libffi-devel \
    blas-devel \
    lapack-devel
```

### Windows

**Native installation** may require:
- **Visual Studio Build Tools**: For compiling C extensions
  - Download: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
  - Select: "Desktop development with C++"
  - Size: ~2-3 GB

**Recommendation**: Use Docker to avoid compilation requirements on Windows.

---

## Testing Optional Dependencies

### Check Installation Status

Lobster provides a built-in system checker:

```bash
# Run pre-installation check
python check-system.py

# Check from within Lobster
lobster chat
🦞 You: /status

# Detailed system information
🦞 You: /dashboard
```

### Test Individual Components

**PyMOL:**
```bash
# Command line test
pymol -c -Q -d "fetch 1AKE; quit"

# Test in Lobster
lobster chat
🦞 You: "Test PyMOL by fetching structure 1AKE"
```

**Docling:**
```bash
# Python test
python -c "from docling.document_converter import DocumentConverter; print('✅ OK')"

# Test in Lobster
lobster chat
🦞 You: "Test Docling by extracting methods from a sample publication"
```

**System Libraries (Linux):**
```bash
# Check installed packages
dpkg -l | grep -E 'libhdf5|libblas|liblapack|libxml2'  # Ubuntu/Debian
rpm -qa | grep -E 'hdf5|blas|lapack|libxml2'          # Fedora/RHEL

# Verify pkg-config can find them
pkg-config --modversion hdf5
pkg-config --libs libxml-2.0
```

### Verification Script

Create a test script to check all optional dependencies:

```python
#!/usr/bin/env python3
"""Test optional dependencies"""

import sys

def check_pymol():
    try:
        import pymol
        print("✅ PyMOL: Available")
        return True
    except ImportError:
        print("⚠️  PyMOL: Not installed")
        return False

def check_docling():
    try:
        from docling.document_converter import DocumentConverter
        import docling
        print(f"✅ Docling: Available (version {docling.__version__})")
        return True
    except ImportError:
        print("⚠️  Docling: Not installed")
        return False

def check_system_libs():
    try:
        import h5py
        import lxml
        print("✅ System libraries: Available")
        return True
    except ImportError as e:
        print(f"⚠️  System libraries: Missing ({e.name})")
        return False

if __name__ == "__main__":
    print("Checking optional dependencies...\n")

    pymol_ok = check_pymol()
    docling_ok = check_docling()
    libs_ok = check_system_libs()

    print("\n" + "="*50)
    if pymol_ok and docling_ok and libs_ok:
        print("✅ All optional dependencies available")
        sys.exit(0)
    else:
        print("⚠️  Some optional dependencies missing")
        print("Lobster will work with reduced functionality")
        sys.exit(0)
```

Save as `check-optional.py` and run: `python check-optional.py`

---

## Getting Help

If you encounter issues installing optional dependencies:

1. **Check platform-specific installation guide**:
   - macOS: See Installation Guide section on macOS
   - Ubuntu/Linux: Run `./install-ubuntu.sh`
   - Windows: Use Docker Desktop (recommended) or see [Installation Guide](02-installation.md)

2. **Consider Docker**:
   - All optional dependencies pre-installed
   - No compilation required
   - Run: `docker-compose run --rm lobster-cli`

3. **Community Support**:
   - GitHub Issues: https://github.com/the-omics-os/lobster-local/issues
   - Email: info@omics-os.com
   - Documentation: [Troubleshooting Guide](28-troubleshooting.md)

---

## Summary

**Quick Decision Guide:**

| Your Situation | Recommended Setup |
|---------------|-------------------|
| Standard RNA-seq/proteomics analysis | Core Lobster only (no optional deps) |
| + Literature mining with complex PDFs | + Docling (5 min install) |
| + Protein structure analysis | + PyMOL (15-30 min install) |
| Windows user | Use Docker (includes everything) |
| Can't install PyMOL | Use cloud mode or Docker |
| Production deployment | Docker (consistent environment) |

**Installation Priority:**
1. **Start**: Core Lobster AI (required)
2. **Add if needed**: Docling for better PDF parsing (easy install)
3. **Add if needed**: PyMOL for structure visualization (moderate effort)
4. **Alternative**: Use Docker and get everything pre-installed

---

**Related Documentation:**
- [Installation Guide](02-installation.md) - Main installation instructions
- [Protein Structure Visualization](40-protein-structure-visualization.md) - Using PyMOL
- [Publication Intelligence](37-publication-intelligence-deep-dive.md) - Using Docling
- [Troubleshooting](28-troubleshooting.md) - Common issues

**Last Updated**: 2025-01-16
