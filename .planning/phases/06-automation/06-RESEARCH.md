# Phase 6: Automation - Research

**Researched:** 2026-02-19
**Domain:** Ontology embedding data pipeline, S3 hosting, auto-download, alternative embedding providers, cloud handoff spec
**Confidence:** HIGH

## Summary

Phase 6 delivers the offline data pipeline that pre-builds ontology embeddings, hosts them on S3 as ChromaDB tarballs, and auto-downloads them on first use -- eliminating the 10-15 minute cold-start embedding time for fresh installs. It also adds two alternative embedding providers (SentenceTransformers MiniLM and OpenAI) and writes a cloud-hosted ChromaDB handoff spec for vector.omics-os.com.

The build script (`scripts/build_ontology_embeddings.py`) is a standalone CLI tool that parses OBO files via obonet, embeds each term's definition + label using SapBERT, stores them in ChromaDB PersistentClient collections with full metadata (term_id, name, synonyms, namespace, is_obsolete), and produces gzipped tarballs ready for S3 upload. The existing ChromaDB backend needs a small extension: before creating a fresh collection, it checks if a pre-built tarball is available at `~/.lobster/ontology_cache/` and extracts it. If the tarball isn't cached locally, it downloads it from `s3://lobster-ontology-data/v1/` (public read-only bucket) with a Rich progress bar. The two new embedding providers follow the exact same BaseEmbedder ABC pattern as SapBERT. The cloud handoff spec is a markdown document with architecture, auth, API design, and deployment steps for vector.omics-os.com.

**Primary recommendation:** Structure this as 4 plans: (1) MiniLM + OpenAI embedding providers with tests, (2) build script with OBO parsing and tarball generation, (3) ChromaDB auto-download extension with progress bar and cache management, (4) cloud handoff spec document.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EMBED-03 | SentenceTransformers provider loads all-MiniLM-L6-v2 (384d) as general fallback | MiniLMEmbedder following SapBERT pattern but with mean pooling and 384 dimensions |
| EMBED-04 | OpenAI provider uses text-embedding-3-small (1536d) with lazy client init | OpenAIEmbedder using `openai.OpenAI().embeddings.create()` with lazy client, import-guarded |
| DATA-01 | Build script parses OBO files and generates SapBERT embeddings | Standalone script using obonet + SapBERTEmbedder, processing 3 ontologies |
| DATA-02 | Build script embeds definition + primary label per term (not synonyms separately) | Text format: "{label}: {definition}" per term; skip terms without definitions |
| DATA-03 | Build script outputs ChromaDB collections with metadata | ChromaDB PersistentClient with term_id, name, synonyms (joined string), namespace, is_obsolete |
| DATA-04 | Build script produces tarballs for S3 upload | tar.gz of ChromaDB persist directory per ontology |
| DATA-05 | ChromaDB backend auto-downloads tarballs from S3 on first use | _ensure_collection() method checks cache, downloads with Rich progress, extracts tarball |
| DATA-06 | Tarballs hosted on S3 at s3://lobster-ontology-data/v1/ | Public-read S3 bucket with versioned prefix |
| CLOD-01 | Cloud-hosted ChromaDB handoff spec for vector.omics-os.com | Architecture document covering server mode, auth, deployment |
| CLOD-02 | Handoff spec includes architecture, API design, and deployment steps | ChromaDB server mode + Cognito auth + ECS Fargate + ALB |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| obonet | >=0.3.0 | Parse OBO ontology files into NetworkX graph | Standard for OBO parsing, already used in ontology_graph.py |
| chromadb | >=0.4.0 | PersistentClient for vector storage and tarball output | Already the default backend, proven in Phase 1-5 |
| sentence-transformers | >=2.0.0 | Load SapBERT for build script and MiniLM provider | Already used by SapBERTEmbedder |
| openai | >=1.0.0 | OpenAI embedding API client | Standard Python SDK for text-embedding-3-small |
| requests | 2.32.5 | HTTP streaming download for S3 tarballs | Already a core dependency |
| rich | >=12.0.0 | Progress bar for tarball download | Already a core dependency, extensively used in cli.py |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tarfile | stdlib | Create and extract .tar.gz archives | Build script output and auto-download extraction |
| tempfile | stdlib | Temporary directory for safe extraction | Auto-download tarball extraction |
| shutil | stdlib | Move extracted directory to cache | Auto-download cache management |
| hashlib | stdlib | Verify tarball integrity (optional) | Download verification |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| requests (streaming download) | httpx | httpx also available but requests pattern is simpler for streaming; S3 public URL is standard HTTP GET |
| Rich progress bar | tqdm | Rich already a dependency, more polished; tqdm would add a new dependency |
| tarball (tar.gz) | zip | tar.gz is smaller, standard for Linux, matches SRAgent pattern |
| Public S3 bucket | CloudFront CDN | CDN adds complexity; S3 public read is sufficient for <100MB files, can add CDN later |

**Installation:**
```bash
# Build script deps (already optional extras)
pip install obonet chromadb sentence-transformers

# OpenAI provider (optional)
pip install openai
```

**Note:** All dependencies are import-guarded. The build script is a developer tool run offline. End users only need requests (core dep) for auto-download.

## Architecture Patterns

### Recommended Project Structure
```
scripts/
    build_ontology_embeddings.py    # NEW: Offline build script (DATA-01 to DATA-04)

lobster/core/vector/
    backends/
        chromadb_backend.py         # MODIFIED: Add _ensure_collection() auto-download
    embeddings/
        minilm.py                   # NEW: MiniLM provider (EMBED-03)
        openai_embedder.py          # NEW: OpenAI provider (EMBED-04)
    config.py                       # MODIFIED: Add minilm/openai branches to create_embedder()

tests/unit/core/vector/
    test_embedders.py               # MODIFIED: Add MiniLM + OpenAI tests
    test_auto_download.py           # NEW: Auto-download + cache tests

docs/
    cloud-chromadb-handoff.md       # NEW: Cloud handoff spec (CLOD-01, CLOD-02)
```

### Pattern 1: Build Script - OBO to ChromaDB Tarball
**What:** Standalone script that parses OBO files, generates SapBERT embeddings, stores in ChromaDB, and creates tarballs.
**When to use:** Run offline by developer before release; output uploaded to S3.
**Example:**
```python
# Source: SRAgent obo-embed.py pattern + existing ontology_graph.py OBO_URLS
import obonet
import chromadb
import tarfile
from pathlib import Path

def build_ontology(ontology_name: str, obo_url: str, output_dir: Path):
    """Parse OBO, embed terms, store in ChromaDB, create tarball."""
    # 1. Parse OBO
    graph = obonet.read_obo(obo_url)

    # 2. Extract terms with definitions
    terms = []
    for node_id, data in graph.nodes(data=True):
        name = data.get("name", "")
        definition = data.get("def", "")
        if not name:
            continue
        # Embed "label: definition" or just "label" if no definition
        text = f"{name}: {definition}" if definition else name
        synonyms = data.get("synonym", [])
        # Join synonyms into a single string for metadata
        synonym_str = "; ".join(str(s) for s in synonyms) if synonyms else ""
        terms.append({
            "id": node_id,
            "text": text,
            "metadata": {
                "term_id": node_id,
                "name": name,
                "synonyms": synonym_str,
                "namespace": data.get("namespace", ""),
                "is_obsolete": str(data.get("is_obsolete", "false")),
            },
        })

    # 3. Embed with SapBERT
    from lobster.core.vector.embeddings.sapbert import SapBERTEmbedder
    embedder = SapBERTEmbedder()
    texts = [t["text"] for t in terms]
    embeddings = embedder.embed_batch(texts)

    # 4. Store in ChromaDB PersistentClient
    persist_dir = output_dir / f"{ontology_name}_sapbert_768"
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        name=f"{ontology_name}_v2024_01",
        metadata={"hnsw:space": "cosine"},
    )
    # Batch add (5000 chunk limit)
    ...

    # 5. Create tarball
    tarball_path = output_dir / f"{ontology_name}_sapbert_768.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(persist_dir, arcname=persist_dir.name)
```

### Pattern 2: Auto-Download with Cache Check
**What:** ChromaDB backend checks for pre-built collection before creating empty one; downloads from S3 if needed.
**When to use:** On first use of any ontology collection (match_ontology or query with ontology collection name).
**Example:**
```python
# Source: SRAgent tissue_ontology.py download pattern + existing ChromaDBBackend
import tarfile
import tempfile
import shutil
from pathlib import Path

# S3 URLs for pre-built ontology data
ONTOLOGY_TARBALLS = {
    "mondo_v2024_01": "https://lobster-ontology-data.s3.amazonaws.com/v1/mondo_sapbert_768.tar.gz",
    "uberon_v2024_01": "https://lobster-ontology-data.s3.amazonaws.com/v1/uberon_sapbert_768.tar.gz",
    "cell_ontology_v2024_01": "https://lobster-ontology-data.s3.amazonaws.com/v1/cell_ontology_sapbert_768.tar.gz",
}

ONTOLOGY_CACHE_DIR = Path.home() / ".lobster" / "ontology_cache"

def _ensure_ontology_data(self, collection_name: str) -> bool:
    """Check cache, download tarball if needed, extract to persist path."""
    if collection_name not in ONTOLOGY_TARBALLS:
        return False  # Not an ontology collection, skip

    # Check if collection already exists in ChromaDB
    client = self._get_client()
    try:
        coll = client.get_collection(collection_name)
        if coll.count() > 0:
            return True  # Already populated
    except ValueError:
        pass  # Collection doesn't exist yet

    # Check ontology cache for extracted data
    cache_dir = ONTOLOGY_CACHE_DIR / collection_name
    if cache_dir.exists() and any(cache_dir.iterdir()):
        # Load from cache into ChromaDB persist path
        ...
        return True

    # Download tarball from S3
    url = ONTOLOGY_TARBALLS[collection_name]
    _download_and_extract(url, cache_dir)
    return True
```

### Pattern 3: MiniLM Embedding Provider (Mean Pooling)
**What:** SentenceTransformers provider for all-MiniLM-L6-v2 with 384d mean-pooled embeddings.
**When to use:** General-purpose fallback when SapBERT is not appropriate (non-biomedical text).
**Example:**
```python
# Source: HuggingFace all-MiniLM-L6-v2 model card
class MiniLMEmbedder(BaseEmbedder):
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    DIMENSIONS = 384

    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "MiniLM embeddings require sentence-transformers. "
                "Install with: pip install sentence-transformers"
            )
        # MiniLM uses MEAN pooling (default in SentenceTransformer)
        self._model = SentenceTransformer(self.MODEL_NAME)

    def embed_text(self, text: str) -> list[float]:
        self._load_model()
        return self._model.encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        return self._model.encode(texts, convert_to_numpy=True, batch_size=128).tolist()

    @property
    def dimensions(self) -> int:
        return self.DIMENSIONS
```

### Pattern 4: OpenAI Embedding Provider (Lazy Client)
**What:** OpenAI API provider with lazy client initialization and configurable model.
**When to use:** When user wants high-quality embeddings via API (requires OPENAI_API_KEY).
**Example:**
```python
# Source: OpenAI Python SDK api.md + PyPI openai 2.21.0
class OpenAIEmbedder(BaseEmbedder):
    MODEL_NAME = "text-embedding-3-small"
    DIMENSIONS = 1536

    def __init__(self, model: str | None = None):
        self._model_name = model or self.MODEL_NAME
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI embeddings require the openai package. "
                "Install with: pip install openai"
            )
        self._client = OpenAI()  # Reads OPENAI_API_KEY from env
        return self._client

    def embed_text(self, text: str) -> list[float]:
        client = self._get_client()
        response = client.embeddings.create(
            model=self._model_name,
            input=text,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        response = client.embeddings.create(
            model=self._model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    @property
    def dimensions(self) -> int:
        return self.DIMENSIONS
```

### Pattern 5: Rich Progress Bar for Downloads
**What:** Stream download with content-length-based progress bar using Rich (already a dependency).
**When to use:** Auto-downloading ontology tarballs from S3.
**Example:**
```python
# Source: Rich docs + requests streaming
import requests
from pathlib import Path
from rich.progress import (
    Progress, BarColumn, DownloadColumn,
    TransferSpeedColumn, TimeRemainingColumn,
)

def download_with_progress(url: str, dest: Path) -> None:
    """Download file with Rich progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    dest.parent.mkdir(parents=True, exist_ok=True)

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            f"Downloading {dest.name}...",
            total=total or None,
        )
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                progress.update(task, advance=len(chunk))
```

### Anti-Patterns to Avoid
- **Embedding synonyms separately:** DATA-02 explicitly says "not synonyms separately" to avoid duplication. Embed "{label}: {definition}" as one document per term; store synonyms in metadata only.
- **Using tarfile.extractall without filter:** Security risk (CVE-2007-4559). Use `tarfile.extractall(filter='data')` (Python 3.12+) or validate member paths before extraction.
- **Downloading at import time:** Auto-download ONLY triggers when a specific ontology collection is first queried, not when the module is imported.
- **Hardcoding S3 region in URLs:** Use the virtual-hosted-style URL (`https://{bucket}.s3.amazonaws.com/`) which works globally.
- **Module-level OpenAI client creation:** The OpenAI client reads OPENAI_API_KEY from env. Must be lazy to avoid crashing when key is not set.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| OBO file parsing | Custom parser for OBO format | obonet + networkx | OBO format has edge cases (escape sequences, multi-value fields, stanzas) |
| Progress bar for downloads | Custom terminal progress display | rich.progress.Progress | Already a dependency, handles terminal width, speed, ETA |
| Tarball creation/extraction | Manual file iteration + compression | tarfile stdlib | Handles compression, permissions, symlinks correctly |
| OpenAI embedding batching | Manual chunking + rate limiting | openai.embeddings.create(input=list) | SDK handles batching, retries, and rate limits internally |
| SentenceTransformer loading | Manual HuggingFace model + tokenizer | SentenceTransformer(model_name) | Handles pooling config, device detection, batching |

**Key insight:** The build script is an offline developer tool run once per ontology release (~quarterly). Optimize for correctness and simplicity, not for speed. The auto-download is the user-facing critical path -- it must be reliable with good error messages.

## Common Pitfalls

### Pitfall 1: OBO Definition Field Encoding
**What goes wrong:** The `def` field in OBO files contains quoted strings with escape characters like `"A cell that ..." [REF:xxxxx]`. Direct use includes the quotes and references.
**Why it happens:** obonet returns the raw OBO string including quotes and cross-references in brackets.
**How to avoid:** Strip outer quotes and trailing `[reference]` from definitions. Pattern: `definition.strip('"').split('" [')[0]` or regex.
**Warning signs:** Embedded terms look like `"A cell that has..." [CL:curator]` instead of clean text.

### Pitfall 2: ChromaDB PersistentClient Directory Sharing
**What goes wrong:** Two ChromaDB PersistentClients pointing at the same directory cause SQLite locking errors.
**Why it happens:** ChromaDB uses SQLite internally, which has file-level locking.
**How to avoid:** The build script creates its OWN PersistentClient at a temporary output directory. The auto-download extracts to `~/.lobster/ontology_cache/` and the main backend reads from `~/.lobster/vector_store/`. These must be separate directories. After download, copy/move the collection data into the backend's persist path.
**Warning signs:** `sqlite3.OperationalError: database is locked`.

### Pitfall 3: Tarball Path Traversal (CVE-2007-4559)
**What goes wrong:** A malicious tarball could contain `../../etc/passwd` paths that escape the target directory.
**Why it happens:** `tarfile.extractall()` without filtering follows relative paths.
**How to avoid:** Use `tarfile.extractall(filter='data')` (Python 3.12+ which is required) or validate all member names start with expected prefix.
**Warning signs:** Files appearing outside the expected extraction directory.

### Pitfall 4: Partial Download Corruption
**What goes wrong:** Network interruption during tarball download leaves a partial file that fails to extract.
**Why it happens:** Writing directly to the cache directory.
**How to avoid:** Download to a temporary file first, then atomically move to cache location after successful extraction. Clean up temp file on error.
**Warning signs:** `tarfile.ReadError: not a gzip file` or truncated archive errors.

### Pitfall 5: OpenAI Rate Limits on Batch Embedding
**What goes wrong:** Large batch embed calls to OpenAI API get rate-limited or time out.
**Why it happens:** text-embedding-3-small has rate limits per minute.
**How to avoid:** The build script uses SapBERT (local), so this only affects the OpenAI runtime provider. For the provider, single-text and small-batch calls are fine. For batch operations, the SDK handles retries internally.
**Warning signs:** `openai.RateLimitError` with retry-after header.

### Pitfall 6: Collection Name Mismatch Between Build and Query
**What goes wrong:** Build script creates collection "mondo_v2024_01" but query looks for "mondo" which resolves to a different collection name.
**Why it happens:** ONTOLOGY_COLLECTIONS mapping in service.py resolves aliases to versioned names.
**How to avoid:** Build script MUST use the exact versioned collection names from ONTOLOGY_COLLECTIONS: `mondo_v2024_01`, `uberon_v2024_01`, `cell_ontology_v2024_01`. The tarball directory name should match.
**Warning signs:** "Collection not found" errors after downloading and extracting tarball.

### Pitfall 7: MiniLM Uses Mean Pooling (Not CLS)
**What goes wrong:** Using CLS pooling for MiniLM degrades performance.
**Why it happens:** Copy-paste from SapBERT which uses CLS pooling.
**How to avoid:** MiniLM was trained with mean pooling. Use `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")` without custom pooling config (mean is the default).
**Warning signs:** Poor similarity scores for clearly related terms.

## Code Examples

Verified patterns from official sources:

### OBO Node Data Access (obonet)
```python
# Source: obonet README + existing ontology_graph.py
import obonet

graph = obonet.read_obo("https://purl.obolibrary.org/obo/mondo.obo")

for node_id, data in graph.nodes(data=True):
    name = data.get("name", "")         # e.g., "colorectal carcinoma"
    definition = data.get("def", "")     # e.g., '"A malignant..." [url:...]'
    synonyms = data.get("synonym", [])   # list of synonym strings
    namespace = data.get("namespace", "") # e.g., "disease_ontology"
    is_obsolete = data.get("is_obsolete", False)  # bool or string
```

### ChromaDB Tarball Create + Restore
```python
# Source: SRAgent obo-embed.py + tissue_ontology.py patterns
import chromadb
import tarfile
from pathlib import Path

# CREATE (build script)
persist_dir = Path("output/mondo_sapbert_768")
client = chromadb.PersistentClient(path=str(persist_dir))
collection = client.get_or_create_collection(
    name="mondo_v2024_01",
    metadata={"hnsw:space": "cosine"},
)
# ... add documents ...
del client  # Close connection before tarring

tarball = Path("output/mondo_sapbert_768.tar.gz")
with tarfile.open(tarball, "w:gz") as tar:
    tar.add(persist_dir, arcname=persist_dir.name)

# RESTORE (auto-download)
cache_dir = Path.home() / ".lobster" / "ontology_cache"
with tarfile.open(tarball) as tar:
    tar.extractall(path=cache_dir, filter="data")
# Now cache_dir / "mondo_sapbert_768" contains the ChromaDB data
restored = chromadb.PersistentClient(
    path=str(cache_dir / "mondo_sapbert_768")
)
coll = restored.get_collection("mondo_v2024_01")
print(f"Restored {coll.count()} terms")
```

### OpenAI Embeddings API
```python
# Source: openai Python SDK api.md (verified 2.21.0)
from openai import OpenAI

client = OpenAI()  # Reads OPENAI_API_KEY from env

# Single text
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="colorectal carcinoma",
)
embedding = response.data[0].embedding  # list[float], 1536 dimensions

# Batch
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["term1", "term2", "term3"],
)
embeddings = [item.embedding for item in response.data]
```

### Config Factory Extension for New Embedders
```python
# Source: Existing VectorSearchConfig.create_embedder() pattern
def create_embedder(self) -> BaseEmbedder:
    if self.embedding_provider == EmbeddingProvider.sapbert:
        from lobster.core.vector.embeddings.sapbert import SapBERTEmbedder
        return SapBERTEmbedder()

    if self.embedding_provider == EmbeddingProvider.minilm:
        from lobster.core.vector.embeddings.minilm import MiniLMEmbedder
        return MiniLMEmbedder()

    if self.embedding_provider == EmbeddingProvider.openai:
        from lobster.core.vector.embeddings.openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder()

    raise ValueError(
        f"Unsupported embedding provider: {self.embedding_provider}. "
        f"Available: sapbert, minilm, openai"
    )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Embed on first use (10-15 min) | Pre-built tarballs auto-downloaded (<60s) | This phase | UX critical: zero-config cold start |
| SapBERT only | SapBERT + MiniLM + OpenAI providers | This phase | Flexibility for non-biomedical use cases |
| Manual ChromaDB setup | Auto-download + cache at ~/.lobster/ontology_cache/ | This phase | Zero-config for end users |
| No cloud plan | Handoff spec for vector.omics-os.com | This phase | Cloud readiness for Omics-OS Cloud users |

**Deprecated/outdated:**
- OpenAI `text-embedding-ada-002`: Superseded by `text-embedding-3-small` (better performance, cheaper). Use `text-embedding-3-small` exclusively.

## Open Questions

1. **S3 Bucket Public Access Configuration**
   - What we know: The bucket `s3://lobster-ontology-data/v1/` needs public read access for HTTP GET downloads.
   - What's unclear: Whether to use a bucket policy for public read or CloudFront distribution. For v1, public bucket is simpler.
   - Recommendation: Create S3 bucket with public read policy on the `v1/` prefix. Use virtual-hosted-style URLs: `https://lobster-ontology-data.s3.amazonaws.com/v1/{tarball}`. Add CloudFront in v2 if download latency matters.

2. **Tarball Size Estimates**
   - What we know: MONDO ~30K terms, Uberon ~30K terms, Cell Ontology ~5K terms. SapBERT embeddings are 768d float32 = 3KB per vector. ChromaDB overhead (HNSW index + SQLite) roughly 2-3x raw vectors.
   - Estimates: MONDO ~30K * 3KB * 3 = ~270MB uncompressed, ~50-80MB compressed. Uberon similar. Cell Ontology ~15MB compressed. Total ~150-200MB.
   - Recommendation: Build and measure actual sizes. If >100MB per tarball, consider splitting or using more aggressive compression.

3. **ChromaDB Version Compatibility Between Build and Runtime**
   - What we know: ChromaDB persist format can change between versions.
   - What's unclear: Whether a tarball built with chromadb 0.5.x works with chromadb 0.4.x.
   - Recommendation: Pin chromadb version in the build script and document the compatible version range. Include chromadb version in tarball filename metadata (e.g., in a `.version` file inside the tarball).

4. **Auto-Download and the ChromaDB Backend Architecture**
   - What we know: Current ChromaDBBackend uses a single PersistentClient at `~/.lobster/vector_store/`. Pre-built data lives in `~/.lobster/ontology_cache/`. These need to be reconciled.
   - What's unclear: Best approach -- (a) point backend's persist_path at the cache directory, (b) copy collection from cache into backend's directory, or (c) use a separate PersistentClient per ontology.
   - Recommendation: Option (c) -- for ontology collections specifically, use the cached PersistentClient directly. Modify `_get_or_create_collection()` to check if the collection name is a known ontology and use the cached client for those. Non-ontology collections use the default persist path.

## Sources

### Primary (HIGH confidence)
- Existing codebase: `lobster/core/vector/` -- all backends, embedders, config, service (read from source)
- Existing codebase: `lobster/core/vector/ontology_graph.py` -- OBO_URLS, obonet usage (read from source)
- HuggingFace all-MiniLM-L6-v2 model card -- 384d, mean pooling, max 256 tokens (verified via WebFetch)
- OpenAI Python SDK api.md -- `client.embeddings.create()` method signature (verified via WebFetch)
- SRAgent obo-embed.py -- ChromaDB build pattern with obonet (read from source at /Users/tyo/GITHUB/omics-os/tmp_folder/SRAgent/)
- SRAgent tissue_ontology.py -- tarball download + extract + cache pattern (read from source)
- ChromaDB docs -- PersistentClient, HttpClient, client-server mode, Docker deployment, AWS deployment (verified via WebFetch)

### Secondary (MEDIUM confidence)
- obonet README -- node data access patterns (name, def, synonym, namespace, is_obsolete) (verified via WebFetch)
- PyPI openai 2.21.0 -- current version (verified via WebFetch)
- Rich 14.3.2 -- Progress bar API (verified installed in venv)

### Tertiary (LOW confidence)
- Tarball size estimates -- calculated from term counts and embedding dimensions, not measured
- ChromaDB version compatibility across persist format -- general knowledge, not verified with specific versions

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries verified via existing codebase, official docs, or PyPI
- Architecture: HIGH -- build script pattern proven in SRAgent, auto-download pattern proven in SRAgent, embedding providers follow existing SapBERT pattern exactly
- Pitfalls: HIGH -- most are based on direct codebase reading (collection name mismatch, pooling config) or well-known issues (tarball security, partial download)
- Cloud handoff spec: MEDIUM -- ChromaDB server mode docs verified, but specific AWS deployment patterns (ECS + ALB) are extrapolated from existing lobster-cloud patterns

**Research date:** 2026-02-19
**Valid until:** 2026-04-19 (stable -- OBO format, ChromaDB API, and embedding models are mature)
