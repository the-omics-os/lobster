#!/usr/bin/env python3
"""Test PRIDE file listing and download URL extraction."""

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.pride_provider import PRIDEProvider
from pathlib import Path

# Create minimal data manager
workspace = Path("/tmp/test_pride")
workspace.mkdir(exist_ok=True)

dm = DataManagerV2(workspace_path=workspace)
provider = PRIDEProvider(dm)

# Test file listing
print("Testing PRIDE file listing for PXD053787...")
try:
    files = provider.get_project_files("PXD053787")
    print(f"✅ Found {len(files)} files")

    # Categorize by file type
    categories = {}
    for f in files:
        cat = f.get("fileCategory", {})
        if isinstance(cat, dict):
            cat_name = cat.get("value", "UNKNOWN")
        else:
            cat_name = str(cat)
        categories[cat_name] = categories.get(cat_name, 0) + 1

    print("\nFile categories:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} files")

    # Show first 5 files
    print("\nFirst 5 files:")
    for i, f in enumerate(files[:5], 1):
        name = f.get("fileName", "Unknown")
        size = f.get("fileSizeBytes", 0)
        size_mb = size / (1024*1024) if size else 0
        print(f"  {i}. {name} ({size_mb:.1f} MB)")

except Exception as e:
    print(f"❌ File listing failed: {e}")
    import traceback
    traceback.print_exc()

# Test FTP URL extraction
print("\n\nTesting FTP URL extraction...")
try:
    ftp_urls = provider.get_ftp_urls("PXD053787", file_category="RESULT")
    print(f"✅ Found {len(ftp_urls)} RESULT file FTP URLs")

    if ftp_urls:
        print("\nFirst 3 RESULT files:")
        for i, (filename, url) in enumerate(ftp_urls[:3], 1):
            print(f"  {i}. {filename}")
            print(f"     {url[:80]}...")

except Exception as e:
    print(f"❌ FTP URL extraction failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ File listing tests complete!")
