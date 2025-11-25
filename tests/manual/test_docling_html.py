"""
Comprehensive Docling HTML Testing Script

This script systematically tests Docling's HTML extraction capabilities to diagnose
why Springer article pages return 0 chars despite successful structure parsing.

Tests:
1. Simple static HTML with known content (baseline)
2. Springer HTML inspection with requests + BeautifulSoup
3. HTMLDocumentBackend direct usage
4. Enhanced HTTP headers and fetching

Run: python tests/manual/test_docling_html.py
"""

import logging
from io import BytesIO
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_1_simple_html():
    """Test 1: Baseline - Simple static HTML with known content."""
    print("\n" + "=" * 80)
    print("TEST 1: Simple Static HTML (Baseline)")
    print("=" * 80)

    from lobster.services.data_access.docling_service import DoclingService

    # Create simple HTML with clear structure
    simple_html = """
    <html>
    <head><title>Test Document</title></head>
    <body>
        <h1>Methods and Materials</h1>
        <p>This is the first paragraph of the methods section. It contains detailed information about our experimental procedures.</p>
        <p>This is the second paragraph with more methodological details. We used advanced techniques for data analysis.</p>

        <h2>Data Collection</h2>
        <p>Data was collected using standardized protocols over a 6-month period.</p>

        <h1>Results</h1>
        <p>Our results showed significant improvements in all measured parameters.</p>
    </body>
    </html>
    """

    # Save to temp file
    temp_file = Path("/tmp/test_simple.html")
    temp_file.write_text(simple_html)

    # Test extraction
    service = DoclingService()

    try:
        result = service.extract_methods_section(
            source=str(temp_file), keywords=["method", "material"], max_paragraphs=50
        )

        print(f"\n✓ Extraction completed")
        print(f"  Methods text length: {len(result['methods_text'])} chars")
        print(f"  Methods markdown length: {len(result['methods_markdown'])} chars")
        print(f"  Sections found: {len(result['sections'])}")
        print(f"  Parser: {result['provenance']['parser']}")

        if result["methods_text"]:
            print(f"\n  Methods text preview:")
            print(f"  {result['methods_text'][:300]}...")
        else:
            print(f"\n  ❌ WARNING: methods_text is EMPTY")

        # Print sections
        if result["sections"]:
            print(f"\n  Section titles:")
            for sec in result["sections"][:5]:
                print(
                    f"    - {sec['title']}: {len(sec.get('content_preview', ''))} chars"
                )

        return len(result["methods_text"]) > 0

    except Exception as e:
        print(f"\n❌ Test 1 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_2_springer_html_inspection():
    """Test 2: Inspect Springer HTML content with requests + BeautifulSoup."""
    print("\n" + "=" * 80)
    print("TEST 2: Springer HTML Inspection")
    print("=" * 80)

    import requests
    from bs4 import BeautifulSoup

    url = "https://link.springer.com/article/10.1007/s43657-024-00165-x"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        print(f"\nFetching: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        print(f"✓ HTTP {response.status_code}")
        print(f"  Content-Type: {response.headers.get('Content-Type')}")
        print(f"  Content-Length: {len(response.text):,} chars")

        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Check for common content containers
        print(f"\n  HTML Structure Analysis:")
        print(f"    Total text length: {len(soup.get_text()):,} chars")
        print(f"    <article> tags: {len(soup.find_all('article'))}")
        print(f"    <section> tags: {len(soup.find_all('section'))}")
        print(f"    <div> tags: {len(soup.find_all('div'))}")
        print(f"    <p> tags: {len(soup.find_all('p'))}")
        print(f"    <h1> tags: {len(soup.find_all('h1'))}")
        print(f"    <h2> tags: {len(soup.find_all('h2'))}")

        # Look for main content areas
        main_content = (
            soup.find("main") or soup.find("article") or soup.find(id="main-content")
        )
        if main_content:
            print(f"\n  ✓ Found main content container:")
            print(f"    Tag: <{main_content.name}>")
            print(f"    Text length: {len(main_content.get_text()):,} chars")
            print(f"    Paragraphs: {len(main_content.find_all('p'))}")
        else:
            print(f"\n  ⚠ No main content container found")

        # Check for JavaScript-rendered content indicators
        scripts = soup.find_all("script")
        print(f"\n  JavaScript Analysis:")
        print(f"    <script> tags: {len(scripts)}")

        # Look for React/Vue mounting points
        if (
            soup.find(id="app")
            or soup.find(id="root")
            or soup.find(attrs={"data-react-root": True})
        ):
            print(f"    ⚠ Found JavaScript app mount points (likely JS-rendered)")
        else:
            print(f"    ✓ No obvious JavaScript app indicators")

        # Extract some sample text
        paragraphs = soup.find_all("p", limit=5)
        if paragraphs:
            print(f"\n  Sample paragraphs:")
            for i, p in enumerate(paragraphs[:3]):
                text = p.get_text(strip=True)
                if text:
                    print(f"    P{i+1}: {text[:80]}...")
                else:
                    print(f"    P{i+1}: [EMPTY]")

        # Check for abstract (should definitely be present)
        abstract = soup.find(
            "section", class_=lambda c: c and "abstract" in c.lower()
        ) or soup.find(id=lambda i: i and "abstract" in i.lower())
        if abstract:
            abstract_text = abstract.get_text(strip=True)
            print(f"\n  ✓ Found abstract: {len(abstract_text)} chars")
            if abstract_text:
                print(f"    Preview: {abstract_text[:150]}...")
            else:
                print(f"    ❌ Abstract is EMPTY")

        return len(soup.get_text()) > 1000

    except Exception as e:
        print(f"\n❌ Test 2 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_3_html_backend_direct():
    """Test 3: Try HTMLDocumentBackend direct usage."""
    print("\n" + "=" * 80)
    print("TEST 3: HTMLDocumentBackend Direct Usage")
    print("=" * 80)

    import requests
    from docling.backend.html_backend import HTMLDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import InputDocument

    url = "https://link.springer.com/article/10.1007/s43657-024-00165-x"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    try:
        # Fetch HTML
        print(f"\nFetching: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        html_bytes = response.content

        print(f"✓ Fetched {len(html_bytes):,} bytes")

        # Create InputDocument
        in_doc = InputDocument(
            path_or_stream=BytesIO(html_bytes),
            format=InputFormat.HTML,
            backend=HTMLDocumentBackend,
            filename="springer.html",
        )

        # Use backend directly
        print(f"\n  Creating HTMLDocumentBackend...")
        backend = HTMLDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(html_bytes))

        print(f"  Converting document...")
        dl_doc = backend.convert()

        print(f"\n✓ Conversion complete")

        # Export to markdown
        markdown = dl_doc.export_to_markdown()
        print(f"  Markdown length: {len(markdown):,} chars")

        if markdown:
            print(f"\n  Markdown preview:")
            print(markdown[:500])
            print("...")
        else:
            print(f"\n  ❌ Markdown is EMPTY")

        # Check document structure
        if hasattr(dl_doc, "texts"):
            print(f"\n  Document texts: {len(dl_doc.texts)} items")
            for i, text_item in enumerate(dl_doc.texts[:5]):
                if hasattr(text_item, "text"):
                    print(
                        f"    Item {i}: {text_item.label} - {len(text_item.text)} chars"
                    )
                    if text_item.text:
                        print(f"      Preview: {text_item.text[:80]}...")

        return len(markdown) > 100

    except Exception as e:
        print(f"\n❌ Test 3 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_4_enhanced_headers():
    """Test 4: Test with enhanced HTTP headers and different methods."""
    print("\n" + "=" * 80)
    print("TEST 4: Enhanced HTTP Headers and Fetching")
    print("=" * 80)

    import requests

    from lobster.services.data_access.docling_service import DoclingService

    url = "https://link.springer.com/article/10.1007/s43657-024-00165-x"

    # Try multiple header configurations
    header_configs = [
        {
            "name": "Basic",
            "headers": {
                "User-Agent": "Lobster AI Research Tool/1.0",
            },
        },
        {
            "name": "Browser-like",
            "headers": {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            },
        },
        {
            "name": "Academic",
            "headers": {
                "User-Agent": "Mozilla/5.0 (compatible; Academic Research Bot; +https://omics-os.com)",
                "From": "support@omics-os.com",
                "Accept": "text/html",
            },
        },
    ]

    service = DoclingService()
    results = {}

    for config in header_configs:
        print(f"\n  Testing {config['name']} headers...")

        try:
            # Fetch with headers
            response = requests.get(url, headers=config["headers"], timeout=30)
            response.raise_for_status()

            content_length = len(response.content)
            print(f"    ✓ HTTP {response.status_code}, {content_length:,} bytes")

            # Save to temp file
            temp_file = Path(f"/tmp/springer_{config['name'].lower()}.html")
            temp_file.write_bytes(response.content)

            # Try extraction
            result = service.extract_methods_section(
                source=str(temp_file),
                keywords=["method", "material"],
                max_paragraphs=50,
            )

            methods_len = len(result.get("methods_text", ""))
            markdown_len = len(result.get("methods_markdown", ""))

            print(f"    Methods text: {methods_len} chars")
            print(f"    Markdown: {markdown_len} chars")

            results[config["name"]] = {
                "success": True,
                "methods_len": methods_len,
                "markdown_len": markdown_len,
            }

        except Exception as e:
            print(f"    ❌ Failed: {e}")
            results[config["name"]] = {"success": False, "error": str(e)}

    # Summary
    print(f"\n  Header Configuration Comparison:")
    for name, result in results.items():
        if result["success"]:
            print(f"    {name}: {result['methods_len']} chars extracted")
        else:
            print(f"    {name}: FAILED")

    return any(r["success"] and r["methods_len"] > 0 for r in results.values())


def main():
    """Run all tests and report results."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "DOCLING HTML DIAGNOSTIC TESTS" + " " * 29 + "║")
    print("╚" + "=" * 78 + "╝")

    results = {
        "Test 1 (Simple HTML)": test_1_simple_html(),
        "Test 2 (Springer Inspection)": test_2_springer_html_inspection(),
        "Test 3 (HTMLBackend Direct)": test_3_html_backend_direct(),
        "Test 4 (Enhanced Headers)": test_4_enhanced_headers(),
    }

    # Final Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} - {test_name}")

    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    # Interpret results
    if results["Test 1 (Simple HTML)"]:
        print("✓ Docling can extract HTML content (baseline working)")
    else:
        print("❌ Docling CANNOT extract simple HTML (critical bug)")

    if results["Test 2 (Springer Inspection)"]:
        print("✓ Springer page contains extractable text content")
    else:
        print("❌ Springer page is empty or JavaScript-rendered")

    if results["Test 3 (HTMLBackend Direct)"]:
        print("✓ HTMLDocumentBackend extracts content successfully")
    else:
        print("❌ HTMLDocumentBackend also fails (not a backend issue)")

    if results["Test 4 (Enhanced Headers)"]:
        print("✓ Enhanced HTTP headers improve extraction")
    else:
        print("❌ HTTP headers do not affect extraction")

    # Recommendation
    print("\n" + "-" * 80)
    print("RECOMMENDATION:")
    print("-" * 80)

    if not results["Test 1 (Simple HTML)"]:
        print(
            "  → Docling HTML extraction is broken. Check Docling version/installation."
        )
    elif not results["Test 2 (Springer Inspection)"]:
        print("  → Springer content requires JavaScript rendering. Consider:")
        print("     - Use Playwright/Selenium for browser automation")
        print("     - Document this as a known limitation")
        print("     - Fallback to abstract-only mode for paywalled publishers")
    elif results["Test 3 (HTMLBackend Direct)"]:
        print("  → Use HTMLDocumentBackend directly instead of DocumentConverter")
        print("     - Update DoclingService to use specific backend for URLs")
    elif results["Test 4 (Enhanced Headers)"]:
        print("  → Add browser-like headers to HTTP requests")
        print("     - Update UnifiedContentService URL fetching with proper headers")
    else:
        print("  → Springer page structure incompatible with Docling's extraction")
        print("     - Consider custom BeautifulSoup extraction for Springer")
        print("     - Or implement publisher-specific parsers")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
