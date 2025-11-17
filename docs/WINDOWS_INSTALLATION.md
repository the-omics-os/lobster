# Windows Installation Guide for Lobster AI

This guide provides detailed instructions for installing Lobster AI on Windows 10 and Windows 11.

## Table of Contents

- [Quick Start (Recommended: Docker)](#quick-start-recommended-docker)
- [Native Installation (Experimental)](#native-installation-experimental)
- [Troubleshooting](#troubleshooting)
- [Comparison: Docker vs Native](#comparison-docker-vs-native)
- [FAQ](#faq)

---

## Quick Start (Recommended: Docker)

Docker Desktop provides the most reliable Lobster AI experience on Windows. It includes all system dependencies and works identically to Linux/macOS installations.

### Prerequisites

- **Windows 10/11** (64-bit, Pro/Enterprise/Education for Hyper-V, or Home with WSL 2)
- **8GB RAM minimum** (16GB recommended for large datasets)
- **20GB free disk space**

### Step 1: Install Docker Desktop

1. Download Docker Desktop for Windows:
   - Visit: https://www.docker.com/products/docker-desktop/
   - Click "Download for Windows"

2. Run the installer:
   - Double-click `Docker Desktop Installer.exe`
   - Follow installation prompts
   - **Restart your computer** when prompted

3. Start Docker Desktop:
   - Launch from Start Menu
   - Wait for "Docker Desktop is running" status
   - Accept license agreement (first-time users)

4. Verify installation:
   ```powershell
   docker --version
   docker-compose --version
   ```

   Expected output:
   ```
   Docker version 24.0.0 or higher
   Docker Compose version 2.0.0 or higher
   ```

### Step 2: Install Lobster AI

1. **Install Git for Windows** (if not already installed):
   - Download from: https://git-scm.com/download/win
   - Run installer, accept defaults
   - Verify: `git --version`

2. **Clone the repository**:
   ```powershell
   git clone https://github.com/the-omics-os/lobster-local.git
   cd lobster-local
   ```

3. **Configure API keys**:
   ```powershell
   # Copy example configuration
   copy .env.example .env

   # Edit with Notepad
   notepad .env
   ```

   Add your API key:
   ```
   ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
   ```

   Save and close.

4. **Run Lobster AI**:

   **Option A: Interactive CLI mode**
   ```powershell
   docker-compose run --rm lobster-cli
   ```

   You'll see:
   ```
   Welcome to Lobster AI - Your bioinformatics analysis assistant
   🦞 You:
   ```

   **Option B: Web service mode**
   ```powershell
   docker-compose up lobster-server
   ```

   Access at: http://localhost:8000

### Step 3: Verify Installation

Inside the Lobster CLI, test with:
```
🦞 You: /status

Expected response: System information and current workspace details
```

### Docker Usage Tips

**Persist Your Work**:
- Data is stored in Docker volumes automatically
- Your workspace persists across sessions
- To backup: `docker-compose down && docker volume ls`

**Stop Services**:
```powershell
# Stop all services
docker-compose down

# Stop and remove volumes (fresh start)
docker-compose down -v
```

**Update Lobster**:
```powershell
git pull origin main
docker-compose build
docker-compose up
```

---

## Native Installation (Experimental)

⚠️ **Warning**: Native Windows installation is experimental. We recommend Docker for most users.

Native installation may encounter issues with:
- C/C++ compiler requirements
- System library dependencies
- Path and permission issues

### Prerequisites

1. **Python 3.12 or higher**:
   - Download from: https://www.python.org/downloads/
   - **Important**: Check "Add Python to PATH" during installation
   - Verify: `python --version`

2. **Git for Windows**:
   - Download from: https://git-scm.com/download/win
   - Verify: `git --version`

3. **(Optional) Visual Studio Build Tools**:
   - Required only if you encounter compilation errors
   - Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
   - Select "Desktop development with C++" workload (2GB+ download)

### Installation Steps

1. **Clone the repository**:
   ```powershell
   git clone https://github.com/the-omics-os/lobster-local.git
   cd lobster-local
   ```

2. **Run the installer**:

   **Option A: PowerShell script** (recommended)
   ```powershell
   .\install.ps1
   ```

   If you see "execution policy" error:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   .\install.ps1
   ```

   **Option B: Batch file** (fallback)
   ```cmd
   install.bat
   ```

3. **Configure API keys**:
   ```powershell
   notepad .env
   ```

   Add your key:
   ```
   ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
   ```

4. **Activate virtual environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   Your prompt should show `(.venv)` prefix.

5. **Run Lobster AI**:
   ```powershell
   lobster chat
   ```

### Manual Installation (Advanced)

If the install scripts fail:

```powershell
# Create virtual environment
python -m venv .venv

# Activate
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install Lobster
pip install -e .
```

---

## Troubleshooting

### Docker Issues

#### "Docker Desktop is not running"

**Solution**:
1. Launch Docker Desktop from Start Menu
2. Wait for whale icon in system tray
3. Try command again

#### "WSL 2 installation incomplete"

**Solution**:
1. Open PowerShell as Administrator
2. Run: `wsl --install`
3. Restart computer
4. Open Docker Desktop, follow WSL 2 setup

#### "drive has not been shared"

**Solution**:
1. Open Docker Desktop → Settings → Resources → File Sharing
2. Add `C:\` drive
3. Click "Apply & Restart"

#### Port 8000 already in use

**Solution**:
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process (replace PID)
taskkill /PID <process_id> /F

# Or change port in docker-compose.yml
```

### Native Installation Issues

#### "Python not found"

**Solutions**:
- Reinstall Python with "Add to PATH" checked
- Or run from Python installer directory: `C:\Users\<username>\AppData\Local\Programs\Python\Python312\python.exe`
- Add Python to PATH manually:
  1. Search "Environment Variables" in Start Menu
  2. Edit "Path" variable
  3. Add: `C:\Users\<username>\AppData\Local\Programs\Python\Python312`

#### "Microsoft Visual C++ 14.0 or greater is required"

**Solution**:
1. Download Visual Studio Build Tools: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
2. Run installer
3. Select "Desktop development with C++"
4. Install (2-3GB download)
5. Restart PowerShell
6. Retry: `pip install -e .`

**Alternative**: Use Docker to avoid compilation requirements.

#### "cannot import name 'xxx'"

**Solution**:
- Your virtual environment may be corrupted
- Recreate:
  ```powershell
  Remove-Item -Recurse -Force .venv
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -e .
  ```

#### "Permission denied" errors

**Solution**:
- Close any antivirus software temporarily
- Run PowerShell as Administrator
- Or use Docker (no admin required)

#### Long path issues

**Solution**:
Enable long path support:
1. Open Registry Editor (Win+R → `regedit`)
2. Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
3. Set `LongPathsEnabled` to `1`
4. Restart computer

Or use Docker (no long path issues).

---

## Comparison: Docker vs Native

| Feature | Docker | Native |
|---------|--------|--------|
| **Setup time** | 30 min (Docker install) | 10 min (if no errors) |
| **Reliability** | ✅ Very high | ⚠️ Moderate (experimental) |
| **Dependencies** | ✅ Included | ❌ Manual (compilers, etc.) |
| **Performance** | Good (10-20% overhead) | ✅ Best (native) |
| **Updates** | ✅ Easy (`docker-compose pull`) | ⚠️ May need reinstall |
| **Disk usage** | ~2GB (Docker + images) | ~1GB (Python packages) |
| **Multi-user** | ✅ Works well | ⚠️ Requires multiple venvs |
| **Enterprise** | ✅ Standardized | ⚠️ Varies by machine |

**Recommendation**:
- **Academic researchers**: Docker (reliable, easy to update)
- **Developers**: Native (if you need IDE integration/debugging)
- **Production/Teams**: Docker (consistent across users)

---

## FAQ

### Can I use Windows Subsystem for Linux (WSL)?

Yes! WSL 2 provides a full Linux environment:

1. Install WSL 2:
   ```powershell
   wsl --install
   ```

2. Install Ubuntu from Microsoft Store

3. Open Ubuntu terminal and follow Linux installation instructions:
   ```bash
   git clone https://github.com/the-omics-os/lobster-local.git
   cd lobster-local
   ./install-ubuntu.sh
   ```

WSL 2 performance is excellent and avoids Windows-specific issues.

### Which Python version should I use?

- **Minimum**: Python 3.12
- **Recommended**: Python 3.12 or 3.13 (latest stable)
- **Not supported**: Python 3.11 or older

### Can I run Lobster AI without admin privileges?

- **Docker**: Requires admin for initial Docker Desktop install, then no admin needed
- **Native**: No admin required if Python already installed
- **WSL**: Requires admin for initial WSL install only

### How much disk space does Lobster need?

- **Docker**:
  - Docker Desktop: ~400MB
  - Lobster images: ~1.5GB
  - Working data: varies by datasets
  - **Total**: ~2GB + data

- **Native**:
  - Python packages: ~800MB
  - Working data: varies by datasets
  - **Total**: ~1GB + data

### Can I use Anaconda/Miniconda instead of standard Python?

Yes, but not recommended due to potential conflicts. If you use Conda:

```powershell
conda create -n lobster python=3.12
conda activate lobster
pip install -e .
```

We recommend standard Python or Docker.

### How do I uninstall?

**Docker**:
```powershell
docker-compose down -v  # Remove containers and volumes
# Uninstall Docker Desktop from Windows Settings → Apps
```

**Native**:
```powershell
Remove-Item -Recurse -Force .venv  # Remove virtual environment
# Uninstall Python from Windows Settings → Apps (if no longer needed)
```

### Can I contribute fixes for Windows compatibility?

Yes! We welcome contributions:
1. Test on Windows and document issues
2. Submit pull requests with fixes
3. Help other users in GitHub Issues

See: https://github.com/the-omics-os/lobster-local/issues

---

## Getting Help

If you encounter issues:

1. **Check this guide** for common solutions
2. **Run system checker**: `python check-system.py`
3. **Search GitHub Issues**: https://github.com/the-omics-os/lobster-local/issues
4. **Report new issues**: Include:
   - Windows version (`winver`)
   - Python version (`python --version`)
   - Error messages (full output)
   - Installation method (Docker or native)

**Contact**: info@omics-os.com

---

**Last Updated**: 2025-01-16

**Windows Support Status**: Experimental (Docker recommended)
