#!/usr/bin/env python3
"""
Build pre-computed ontology embeddings for offline distribution.

Parses OBO ontology files (MONDO, Uberon, Cell Ontology), generates SapBERT
embeddings for each term, stores them in ChromaDB collections with full
metadata, and produces gzipped tarballs ready for S3 upload.

This eliminates the 10-15 minute cold-start embedding time for fresh
installations. The tarballs are uploaded to S3 and auto-downloaded by
end users on first use.

Usage::

    # Build a single ontology
    python scripts/build_ontology_embeddings.py --ontology mondo

    # Build all 3 ontologies
    python scripts/build_ontology_embeddings.py --all

    # Custom output directory
    python scripts/build_ontology_embeddings.py --all --output-dir ./build_output

    # Dry-run (validate dependencies and paths without downloading)
    python scripts/build_ontology_embeddings.py --dry-run --all

Output format:
    Each ontology produces a ``{ontology}_sapbert_768/`` ChromaDB persist
    directory and a ``{ontology}_sapbert_768.tar.gz`` tarball containing it.
    The tarball can be extracted by a ``chromadb.PersistentClient`` pointed
    at the extracted directory.

Requires: obonet, chromadb, sentence-transformers (torch)
Install:  pip install 'lobster-ai[vector-search]' obonet
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Batch size for ChromaDB add operations (matches chromadb_backend.py)
_BATCH_SIZE = 5000


# ---------------------------------------------------------------------------
# OBO term extraction
# ---------------------------------------------------------------------------


def _strip_obo_definition(raw_def: str) -> str:
    """Clean OBO-encoded definition string.

    OBO definitions are stored as ``'"Some definition." [REF:001, REF:002]'``.
    This strips the outer quotes and trailing ``[references]`` bracket.

    Parameters
    ----------
    raw_def : str
        Raw OBO definition field value.

    Returns
    -------
    str
        Cleaned definition text, or empty string if input was empty.
    """
    if not raw_def:
        return ""
    cleaned = raw_def.strip('"')
    # Remove trailing [reference] bracket: '"definition" [PMID:123]'
    if '" [' in raw_def:
        cleaned = cleaned.split('" [')[0].strip()
    return cleaned


def _extract_terms(graph: Any) -> list[dict[str, Any]]:
    """Extract embeddable terms with metadata from an OBO graph.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        Parsed OBO ontology graph from obonet.

    Returns
    -------
    list[dict]
        Each dict has keys: term_id, name, text, synonyms, namespace,
        is_obsolete. Only terms with a non-empty name and not obsolete
        are included.
    """
    terms: list[dict[str, Any]] = []

    for node_id, data in graph.nodes(data=True):
        name = data.get("name", "")
        if not name:
            # Skip terms without a name (DATA-02)
            continue

        is_obsolete = str(data.get("is_obsolete", "false")).lower()
        if is_obsolete == "true":
            # Skip obsolete terms
            continue

        # Clean definition
        raw_def = data.get("def", "")
        definition = _strip_obo_definition(raw_def)

        # Build embedding text: "{label}: {definition}" or just "{label}"
        if definition:
            text = f"{name}: {definition}"
        else:
            text = name

        # Synonyms for metadata
        raw_synonyms = data.get("synonym", [])
        synonym_str = "; ".join(str(s) for s in raw_synonyms)

        namespace = data.get("namespace", "")

        terms.append(
            {
                "term_id": node_id,
                "name": name,
                "text": text,
                "synonyms": synonym_str,
                "namespace": namespace,
                "is_obsolete": is_obsolete,
            }
        )

    return terms


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------


def build_ontology(
    ontology_name: str,
    obo_url: str,
    output_dir: Path,
    collection_name: str,
) -> Path:
    """Build a ChromaDB collection and tarball for one ontology.

    Parameters
    ----------
    ontology_name : str
        Ontology key (e.g., "mondo", "uberon", "cell_ontology").
    obo_url : str
        URL to the OBO file.
    output_dir : Path
        Output directory for persist dirs and tarballs.
    collection_name : str
        Versioned collection name (e.g., "mondo_v2024_01").

    Returns
    -------
    Path
        Path to the generated tarball.

    Raises
    ------
    ImportError
        If required dependencies are not installed.
    RuntimeError
        If the ontology has no embeddable terms.
    """
    # --- Step 1: Parse OBO ---
    try:
        import obonet
    except ImportError:
        raise ImportError(
            "obonet is required for OBO parsing. "
            "Install with: pip install obonet"
        )

    logger.info("Downloading and parsing OBO for '%s' from %s ...", ontology_name, obo_url)
    try:
        graph = obonet.read_obo(obo_url)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download/parse OBO file from {obo_url}: {exc}"
        ) from exc

    total_nodes = graph.number_of_nodes()
    logger.info("Parsed %s: %d total nodes", ontology_name, total_nodes)

    # --- Step 2: Extract terms with metadata ---
    terms = _extract_terms(graph)
    if not terms:
        raise RuntimeError(
            f"Ontology '{ontology_name}' has 0 embeddable terms after filtering. "
            "Check the OBO URL and ontology content."
        )
    logger.info("Extracted %d embeddable terms (skipped %d obsolete/unnamed)",
                len(terms), total_nodes - len(terms))

    # --- Step 3: Embed with SapBERT ---
    try:
        from lobster.core.vector.embeddings.sapbert import SapBERTEmbedder
    except ImportError:
        raise ImportError(
            "SapBERT embeddings require sentence-transformers and PyTorch. "
            "Install with: pip install 'lobster-ai[vector-search]'"
        )

    embedder = SapBERTEmbedder()
    texts = [t["text"] for t in terms]
    logger.info("Embedding %d terms with SapBERT (batch_size=128)...", len(texts))

    # Embed in batches to show progress
    all_embeddings: list[list[float]] = []
    batch_size = 128
    for batch_start in range(0, len(texts), batch_size):
        batch_end = min(batch_start + batch_size, len(texts))
        batch_texts = texts[batch_start:batch_end]
        batch_embeddings = embedder.embed_batch(batch_texts)
        all_embeddings.extend(batch_embeddings)
        batch_num = batch_start // batch_size + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size
        logger.info("  Embedded batch %d/%d (%d terms)", batch_num, total_batches, len(batch_texts))

    logger.info("Embedding complete: %d vectors of dimension %d",
                len(all_embeddings), len(all_embeddings[0]) if all_embeddings else 0)

    # --- Step 4: Store in ChromaDB ---
    try:
        import chromadb
    except ImportError:
        raise ImportError(
            "ChromaDB is required for vector storage. "
            "Install with: pip install chromadb"
        )

    persist_dir = output_dir / f"{ontology_name}_sapbert_768"
    if persist_dir.exists():
        logger.info("Removing existing persist directory: %s", persist_dir)
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [t["term_id"] for t in terms]
    documents = texts
    metadatas = [
        {
            "term_id": t["term_id"],
            "name": t["name"],
            "synonyms": t["synonyms"],
            "namespace": t["namespace"],
            "is_obsolete": t["is_obsolete"],
        }
        for t in terms
    ]

    logger.info("Adding %d documents to collection '%s' in chunks of %d...",
                len(ids), collection_name, _BATCH_SIZE)

    for start in range(0, len(ids), _BATCH_SIZE):
        end = min(start + _BATCH_SIZE, len(ids))
        collection.add(
            ids=ids[start:end],
            embeddings=all_embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        chunk_num = start // _BATCH_SIZE + 1
        total_chunks = (len(ids) + _BATCH_SIZE - 1) // _BATCH_SIZE
        logger.info("  Added chunk %d/%d (%d docs)", chunk_num, total_chunks, end - start)

    final_count = collection.count()
    logger.info("Collection '%s' has %d documents", collection_name, final_count)

    # Close client before tarring
    del client

    # --- Step 5: Create tarball ---
    tarball_path = output_dir / f"{ontology_name}_sapbert_768.tar.gz"
    logger.info("Creating tarball: %s", tarball_path)

    with tarfile.open(str(tarball_path), "w:gz") as tar:
        tar.add(str(persist_dir), arcname=persist_dir.name)

    tarball_size_mb = tarball_path.stat().st_size / (1024 * 1024)

    # --- Step 6: Log summary ---
    logger.info(
        "BUILD COMPLETE for '%s':\n"
        "  Collection: %s\n"
        "  Terms:      %d\n"
        "  Tarball:    %s (%.1f MB)",
        ontology_name, collection_name, final_count, tarball_path, tarball_size_mb,
    )

    return tarball_path


# ---------------------------------------------------------------------------
# Dry-run validation
# ---------------------------------------------------------------------------


def dry_run(ontologies: dict[str, str], output_dir: Path, version_tag: str) -> None:
    """Validate the full pipeline without downloading or embedding.

    Checks all imports, output directory writability, and collection name
    consistency with ONTOLOGY_COLLECTIONS.

    Parameters
    ----------
    ontologies : dict[str, str]
        Map of ontology name to OBO URL.
    output_dir : Path
        Target output directory.
    version_tag : str
        Version tag for collection naming.
    """
    print("=" * 60)
    print("DRY RUN: Validating build pipeline")
    print("=" * 60)

    all_ok = True

    # Check dependencies
    deps = {
        "obonet": "pip install obonet",
        "chromadb": "pip install chromadb",
    }
    for dep_name, install_cmd in deps.items():
        try:
            __import__(dep_name)
            print(f"  [OK] {dep_name} is installed")
        except ImportError:
            print(f"  [MISSING] {dep_name} -- install with: {install_cmd}")
            all_ok = False

    # Check SapBERTEmbedder import
    try:
        from lobster.core.vector.embeddings.sapbert import SapBERTEmbedder  # noqa: F401
        print("  [OK] SapBERTEmbedder importable")
    except ImportError as exc:
        print(f"  [MISSING] SapBERTEmbedder -- {exc}")
        all_ok = False

    # Check output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    test_file = output_dir / ".write_test"
    try:
        test_file.write_text("test")
        test_file.unlink()
        print(f"  [OK] Output directory writable: {output_dir}")
    except OSError as exc:
        print(f"  [ERROR] Output directory not writable: {output_dir} -- {exc}")
        all_ok = False

    # Check ONTOLOGY_COLLECTIONS consistency
    try:
        from lobster.core.vector.service import ONTOLOGY_COLLECTIONS
        collections_available = True
    except ImportError:
        ONTOLOGY_COLLECTIONS = {}
        collections_available = False
        print("  [WARN] Could not import ONTOLOGY_COLLECTIONS (will use generated names)")

    print()
    print("Ontology build plan:")
    print("-" * 60)

    for ont_name, obo_url in ontologies.items():
        collection_name = f"{ont_name}_{version_tag.replace('.', '_')}"
        persist_dir = output_dir / f"{ont_name}_sapbert_768"
        tarball_path = output_dir / f"{ont_name}_sapbert_768.tar.gz"

        print(f"\n  Ontology:    {ont_name}")
        print(f"  OBO URL:     {obo_url}")
        print(f"  Collection:  {collection_name}")
        print(f"  Persist dir: {persist_dir}")
        print(f"  Tarball:     {tarball_path}")

        # Validate collection name matches ONTOLOGY_COLLECTIONS
        if collections_available and ont_name in ONTOLOGY_COLLECTIONS:
            expected = ONTOLOGY_COLLECTIONS[ont_name]
            if collection_name != expected:
                print(f"  [WARN] Collection name mismatch! Expected '{expected}', got '{collection_name}'")
                print(f"         Use --version-tag to match, or update ONTOLOGY_COLLECTIONS")
            else:
                print(f"  [OK] Collection name matches ONTOLOGY_COLLECTIONS")

    print()
    print("=" * 60)
    if all_ok:
        print(f"Dry run complete. Ready to build {len(ontologies)} ontologies.")
        print("Run without --dry-run to proceed.")
    else:
        print("Dry run found issues. Fix the above errors before building.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the ontology embedding build script."""
    parser = argparse.ArgumentParser(
        description=(
            "Build pre-computed SapBERT ontology embeddings for offline distribution. "
            "Parses OBO files, generates embeddings, stores in ChromaDB, and produces "
            "gzipped tarballs for S3 upload."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --ontology mondo\n"
            "  %(prog)s --all\n"
            "  %(prog)s --all --output-dir ./build_output\n"
            "  %(prog)s --dry-run --all\n"
        ),
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ontology",
        choices=["mondo", "uberon", "cell_ontology"],
        help="Build a single ontology (one of: mondo, uberon, cell_ontology)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Build all 3 ontologies (mondo, uberon, cell_ontology)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./ontology_build_output"),
        help="Output directory for ChromaDB persist dirs and tarballs (default: ./ontology_build_output)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dependencies and paths without downloading or embedding",
    )

    parser.add_argument(
        "--version-tag",
        type=str,
        default="v2024_01",
        help=(
            "Version tag for collection naming (default: v2024_01). "
            "Collection names are derived as '{ontology}_{version_tag}'."
        ),
    )

    args = parser.parse_args()

    # Import OBO_URLS for ontology URL resolution
    try:
        from lobster.core.vector.ontology_graph import OBO_URLS
    except ImportError:
        # Fallback if lobster isn't installed (shouldn't happen in dev)
        OBO_URLS = {
            "mondo": "https://purl.obolibrary.org/obo/mondo.obo",
            "uberon": "https://purl.obolibrary.org/obo/uberon/uberon-basic.obo",
            "cell_ontology": "https://purl.obolibrary.org/obo/cl.obo",
        }
        logger.warning("Could not import OBO_URLS from lobster; using built-in fallback")

    # Resolve ontologies to process
    if args.all:
        ontologies = {name: OBO_URLS[name] for name in ["mondo", "uberon", "cell_ontology"]}
    else:
        ontologies = {args.ontology: OBO_URLS[args.ontology]}

    output_dir = args.output_dir.resolve()

    # Dry-run mode
    if args.dry_run:
        dry_run(ontologies, output_dir, args.version_tag)
        return

    # Full build mode
    output_dir.mkdir(parents=True, exist_ok=True)

    tarballs: list[tuple[str, Path, float]] = []

    for ont_name, obo_url in ontologies.items():
        collection_name = f"{ont_name}_{args.version_tag.replace('.', '_')}"

        # Validate against ONTOLOGY_COLLECTIONS if default version tag
        try:
            from lobster.core.vector.service import ONTOLOGY_COLLECTIONS
            expected = ONTOLOGY_COLLECTIONS.get(ont_name)
            if expected and collection_name != expected:
                logger.warning(
                    "Collection name '%s' does not match ONTOLOGY_COLLECTIONS['%s'] = '%s'. "
                    "Users may need to update ONTOLOGY_COLLECTIONS to match.",
                    collection_name, ont_name, expected,
                )
        except ImportError:
            pass

        tarball_path = build_ontology(ont_name, obo_url, output_dir, collection_name)
        tarball_size_mb = tarball_path.stat().st_size / (1024 * 1024)
        tarballs.append((ont_name, tarball_path, tarball_size_mb))

    # Final summary
    print()
    print("=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    for ont_name, tarball_path, size_mb in tarballs:
        print(f"  {ont_name}: {tarball_path} ({size_mb:.1f} MB)")
    print()
    print(f"Total tarballs: {len(tarballs)}")
    print(f"Output directory: {output_dir}")
    print()
    print("Next steps:")
    print("  1. Upload tarballs to S3")
    print("  2. Update download URLs in lobster/core/vector/config.py")


if __name__ == "__main__":
    main()
