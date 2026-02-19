# Cloud-Hosted ChromaDB Deployment Specification

**Document:** Internal handoff spec for vector.omics-os.com
**Status:** Phase 1 -- Architecture & planning (no code changes)
**Author:** Phase 06-04 automation plan
**Date:** 2026-02-19

---

## 1. Overview

### Purpose

Replace local ChromaDB `PersistentClient` with a cloud-hosted ChromaDB server at **vector.omics-os.com** for Omics-OS Cloud SaaS users. This eliminates the need for users to download ontology tarballs (~150-200 MB total) and maintain local vector stores, enabling instant semantic search from the browser-based cloud platform.

### Trigger

Deploy this service when Omics-OS Cloud SaaS users need vector search without local model/data setup -- specifically when the cloud backend (`lobster-cloud/`) routes annotation, metadata, or research queries through the vector search pipeline.

### Scope

Three ontology collections with SapBERT 768-dimensional embeddings:

| Collection | Versioned Name | Approx. Terms | Source |
|------------|---------------|----------------|--------|
| MONDO (diseases) | `mondo_v2024_01` | ~30,000 | purl.obolibrary.org/obo/mondo.obo |
| Uberon (tissues/anatomy) | `uberon_v2024_01` | ~30,000 | purl.obolibrary.org/obo/uberon.obo |
| Cell Ontology (cell types) | `cell_ontology_v2024_01` | ~5,000 | purl.obolibrary.org/obo/cl.obo |

These collection names match the `ONTOLOGY_COLLECTIONS` alias map in `lobster/core/vector/service.py`.

---

## 2. Architecture

### Deployment Topology

```
                                    vector.omics-os.com
                                           |
                                    +------+------+
                                    |   Route53   |
                                    |  (A alias)  |
                                    +------+------+
                                           |
                                    +------+------+
                                    |     ALB     |
                                    | (HTTPS:443) |
                                    +------+------+
                                           |
                              +------------+------------+
                              |                         |
                    +---------+---------+     +---------+---------+
                    |  ChromaDB Server  |     | Health Check Only  |
                    |  ECS Fargate Task |     | /api/v1/heartbeat  |
                    |  (ARM64/Graviton) |     +---------+---------+
                    |  4 vCPU, 16GB RAM |
                    +---------+---------+
                              |
                    +---------+---------+
                    |   EBS gp3 Volume  |
                    |   /data/chroma/   |
                    |   (10 GB)         |
                    +-------------------+

    Omics-OS Cloud (ECS)                       ChromaDB Server (ECS)
    +---------------------+                    +---------------------+
    | FastAPI Backend      |  --- HttpClient -> | chroma run          |
    | (lobster-cloud/api)  |    X-Chroma-Token  | --host 0.0.0.0      |
    | LOBSTER_VECTOR_CLOUD |                    | --port 8000         |
    | _URL=vector.omics-os |                    | PERSIST_DIR=/data   |
    | .com                 |                    | /chroma             |
    +---------------------+                    +---------------------+
```

### Data Flow

1. User submits query via Omics-OS Cloud UI (app.omics-os.com)
2. FastAPI backend validates Cognito JWT session
3. Backend calls `VectorSearchService.query()` or `match_ontology()`
4. `ChromaDBBackend` detects `LOBSTER_VECTOR_CLOUD_URL` env var
5. Backend uses `chromadb.HttpClient` (instead of `PersistentClient`) to query ChromaDB server
6. ChromaDB server returns results from prewarmed HNSW indexes
7. Results flow back through service layer with distance-to-similarity conversion
8. User receives ranked ontology matches

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **ChromaDB server mode** (not embedded) | Multi-tenant access from ECS tasks; no SQLite locking issues |
| **Single ChromaDB instance** (not per-tenant) | Ontology data is shared and read-only; no tenant isolation needed |
| **ECS Fargate (ARM64/Graviton)** | 20% cheaper than x86, sufficient for vector search workloads |
| **4 vCPU, 16 GB RAM** | ChromaDB HNSW index for ~65K terms fits comfortably in memory |
| **EBS gp3 volume** | Persistent storage survives task restarts; gp3 baseline IOPS sufficient |
| **Always-on (no scale-to-zero)** | Cold start re-downloads/re-indexes ~150 MB; not acceptable for <1s latency |
| **Same VPC as Omics-OS Cloud** | Internal ALB routing, no public internet traversal for queries |

---

## 3. Authentication & Authorization

### Token-Based Authentication

ChromaDB server supports token authentication via `CHROMA_SERVER_AUTHN_CREDENTIALS`. The flow:

```
User -> Cognito (JWT) -> Omics-OS Cloud API -> ChromaDB Server (token)
                                                    |
                                          X-Chroma-Token header
```

Users **never** call ChromaDB directly. The Omics-OS Cloud FastAPI backend:
1. Validates the user's Cognito JWT (existing auth middleware in `api/dependencies.py`)
2. Proxies the vector search request to ChromaDB using a service-to-service token
3. Returns results to the user

### Token Configuration

**ChromaDB Server Side** (ECS task definition environment):

```bash
CHROMA_SERVER_AUTHN_CREDENTIALS=<service-token>
CHROMA_SERVER_AUTHN_PROVIDER=chromadb.auth.token.TokenAuthenticationServerProvider
```

**Omics-OS Cloud Backend Side** (ECS task definition environment):

```bash
LOBSTER_VECTOR_CLOUD_URL=https://vector.omics-os.com
LOBSTER_VECTOR_CLOUD_TOKEN=<service-token>
```

**Token Storage:**
- Service token stored in AWS Secrets Manager: `lobster-vector-service-token`
- Referenced in both ECS task definitions via Secrets Manager ARN
- Rotatable without redeployment (ECS pulls latest on task start)

### Client-Side Configuration

When `LOBSTER_VECTOR_CLOUD_URL` is set, `ChromaDBBackend` initializes with:

```python
import chromadb
from chromadb.config import Settings

client = chromadb.HttpClient(
    host="vector.omics-os.com",
    port=443,
    ssl=True,
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
        chroma_client_auth_credentials=os.environ["LOBSTER_VECTOR_CLOUD_TOKEN"],
    ),
)
```

### Authorization Model

| Actor | Access Level | Mechanism |
|-------|-------------|-----------|
| Omics-OS Cloud backend | Full read access to all 3 collections | Service token via X-Chroma-Token |
| End users | No direct access | All queries proxied through authenticated Cloud API |
| Admin (data pipeline) | Write access for collection updates | Separate admin token (future) |

---

## 4. API Design

### SDK Change: ChromaDBBackend Cloud Mode

The existing `ChromaDBBackend` in `lobster/core/vector/backends/chromadb_backend.py` detects the `LOBSTER_VECTOR_CLOUD_URL` environment variable to switch between local and cloud mode:

```python
# In ChromaDBBackend._get_client()
def _get_client(self):
    if self._client is not None:
        return self._client

    import chromadb

    cloud_url = os.environ.get("LOBSTER_VECTOR_CLOUD_URL")
    if cloud_url:
        # Cloud mode: connect to hosted ChromaDB server
        from chromadb.config import Settings

        token = os.environ.get("LOBSTER_VECTOR_CLOUD_TOKEN", "")
        self._client = chromadb.HttpClient(
            host=cloud_url.replace("https://", "").replace("http://", ""),
            port=443,
            ssl=cloud_url.startswith("https"),
            settings=Settings(
                chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
                chroma_client_auth_credentials=token,
            ),
        )
        logger.info("Connected to ChromaDB cloud at %s", cloud_url)
    else:
        # Local mode: use PersistentClient (current behavior, unchanged)
        self._client = chromadb.PersistentClient(path=self._persist_path)
        logger.info("Initialized ChromaDB at %s", self._persist_path)

    return self._client
```

### What Changes

| Component | Change | Impact |
|-----------|--------|--------|
| `ChromaDBBackend._get_client()` | Add `HttpClient` branch when env var set | One method, backward compatible |
| `VectorSearchService` | **No changes** | Backend abstraction works |
| Agent tools | **No changes** | Service abstraction works |
| `VectorSearchConfig` | **No changes** | Config factory creates backend, backend handles mode |

### What Stays The Same

- **No new REST API** -- ChromaDB server exposes its own HTTP API natively
- All existing `VectorSearchService` methods (`query()`, `match_ontology()`) work unchanged
- All agent tools (`annotate_cell_types_semantic`, `standardize_tissue_term`, `standardize_disease_term`) work unchanged
- Local mode continues to work identically when `LOBSTER_VECTOR_CLOUD_URL` is not set

---

## 5. Prewarmed Indexes

### Startup Sequence

On ECS task startup, a custom entrypoint script ensures all 3 ontology collections are available:

```bash
#!/bin/bash
# entrypoint.sh for ChromaDB ECS task

set -e

PERSIST_DIR="/data/chroma"
S3_BUCKET="lobster-ontology-data"
S3_PREFIX="v1"

COLLECTIONS=("mondo_sapbert_768" "uberon_sapbert_768" "cell_ontology_sapbert_768")
TARBALL_NAMES=("mondo_sapbert_768.tar.gz" "uberon_sapbert_768.tar.gz" "cell_ontology_sapbert_768.tar.gz")

# Check if collections exist on EBS volume
if [ ! -d "$PERSIST_DIR" ] || [ -z "$(ls -A $PERSIST_DIR 2>/dev/null)" ]; then
    echo "No existing data found. Downloading pre-built collections from S3..."
    mkdir -p "$PERSIST_DIR"

    for i in "${!TARBALL_NAMES[@]}"; do
        TARBALL="${TARBALL_NAMES[$i]}"
        echo "Downloading $TARBALL..."
        aws s3 cp "s3://${S3_BUCKET}/${S3_PREFIX}/${TARBALL}" "/tmp/${TARBALL}"
        tar -xzf "/tmp/${TARBALL}" -C "$PERSIST_DIR"
        rm "/tmp/${TARBALL}"
        echo "Extracted ${COLLECTIONS[$i]}"
    done
else
    echo "Existing data found at $PERSIST_DIR. Skipping download."
fi

# Start ChromaDB server
echo "Starting ChromaDB server..."
exec chroma run \
    --host 0.0.0.0 \
    --port 8000 \
    --path "$PERSIST_DIR"
```

### Performance Expectations

| Metric | Value | Notes |
|--------|-------|-------|
| Cold start (no EBS data) | ~60-90s | Download ~150 MB from S3 + extract |
| Warm start (EBS has data) | ~5-10s | ChromaDB loads HNSW index from disk |
| First query after start | <1s | HNSW index already in memory |
| Subsequent queries | <100ms | In-memory vector search |

### Health Check

ALB target group health check configuration:

```
Path:     /api/v1/heartbeat
Port:     8000
Protocol: HTTP
Interval: 30s
Timeout:  5s
Healthy threshold:   2
Unhealthy threshold: 3
```

ChromaDB's `/api/v1/heartbeat` endpoint returns `{"nanosecond heartbeat": <timestamp>}` when the server is ready.

---

## 6. Deployment Steps

Step-by-step CDK/AWS instructions for deploying the ChromaDB server.

### Step 1: Create ECS Fargate Service

```python
# In lobster-cloud CDK stack (new stack: VectorSearchStack)
from aws_cdk import (
    Stack, Duration,
    aws_ecs as ecs,
    aws_ec2 as ec2,
    aws_elasticloadbalancingv2 as elbv2,
    aws_efs as efs,
    aws_secretsmanager as sm,
    aws_route53 as route53,
    aws_route53_targets as targets,
)

class VectorSearchStack(Stack):
    def __init__(self, scope, id, vpc, cluster, **kwargs):
        super().__init__(scope, id, **kwargs)

        # Task definition (ARM64/Graviton for cost)
        task_def = ecs.FargateTaskDefinition(
            self, "ChromaDBTask",
            cpu=4096,        # 4 vCPU
            memory_limit_mib=16384,  # 16 GB
            runtime_platform=ecs.RuntimePlatform(
                cpu_architecture=ecs.CpuArchitecture.ARM64,
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
            ),
        )
```

### Step 2: Configure EBS Volume Mount

```python
        # EBS-backed volume for persistent ChromaDB data
        # Note: ECS Fargate supports EBS volumes via configureAtLaunch
        volume = ecs.Volume(
            name="chroma-data",
            configured_at_launch=True,
        )
        task_def.add_volume(volume=volume)
```

### Step 3: Add ChromaDB Container

```python
        # Auth token from Secrets Manager
        chroma_token = sm.Secret.from_secret_name_v2(
            self, "ChromaToken", "lobster-vector-service-token"
        )

        container = task_def.add_container(
            "chromadb",
            image=ecs.ContainerImage.from_registry("chromadb/chroma:latest"),
            environment={
                "IS_PERSISTENT": "TRUE",
                "PERSIST_DIRECTORY": "/data/chroma",
                "CHROMA_SERVER_AUTHN_PROVIDER": (
                    "chromadb.auth.token.TokenAuthenticationServerProvider"
                ),
            },
            secrets={
                "CHROMA_SERVER_AUTHN_CREDENTIALS": ecs.Secret.from_secrets_manager(
                    chroma_token
                ),
            },
            logging=ecs.LogDrivers.aws_logs(stream_prefix="chromadb"),
            port_mappings=[
                ecs.PortMapping(container_port=8000, protocol=ecs.Protocol.TCP)
            ],
            health_check=ecs.HealthCheck(
                command=["CMD-SHELL", "curl -f http://localhost:8000/api/v1/heartbeat || exit 1"],
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                retries=3,
                start_period=Duration.seconds(60),
            ),
            # Custom entrypoint for prewarm
            entry_point=["/bin/bash", "/entrypoint.sh"],
        )

        container.add_mount_points(
            ecs.MountPoint(
                container_path="/data",
                source_volume="chroma-data",
                read_only=False,
            )
        )
```

### Step 4: Create ALB Target Group

```python
        # ALB for vector.omics-os.com
        alb = elbv2.ApplicationLoadBalancer(
            self, "VectorALB",
            vpc=vpc,
            internet_facing=True,
        )

        # HTTPS listener with ACM certificate
        listener = alb.add_listener(
            "HttpsListener",
            port=443,
            certificates=[certificate],  # ACM cert for vector.omics-os.com
        )

        # Fargate service
        service = ecs.FargateService(
            self, "ChromaDBService",
            cluster=cluster,
            task_definition=task_def,
            desired_count=1,
            assign_public_ip=False,
        )

        listener.add_targets(
            "ChromaDBTarget",
            port=8000,
            targets=[service],
            health_check=elbv2.HealthCheck(
                path="/api/v1/heartbeat",
                port="8000",
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                healthy_threshold_count=2,
                unhealthy_threshold_count=3,
            ),
        )
```

### Step 5: Create Route53 Record

```python
        # DNS record for vector.omics-os.com
        hosted_zone = route53.HostedZone.from_lookup(
            self, "Zone", domain_name="omics-os.com"
        )

        route53.ARecord(
            self, "VectorDNS",
            zone=hosted_zone,
            record_name="vector",
            target=route53.RecordTarget.from_alias(
                targets.LoadBalancerTarget(alb)
            ),
        )
```

### Step 6: Store Auth Token in Secrets Manager

```bash
# Create the service token (run once)
aws secretsmanager create-secret \
    --name lobster-vector-service-token \
    --secret-string "$(openssl rand -hex 32)" \
    --region us-east-1
```

### Step 7: Update Omics-OS Cloud ECS Task Definition

Add these environment variables to the existing `LobsterCloudStack` ECS task definition:

```python
# In lobster-cloud CDK LobsterCloudStack
container.add_environment("LOBSTER_VECTOR_CLOUD_URL", "https://vector.omics-os.com")

chroma_token = sm.Secret.from_secret_name_v2(
    self, "ChromaToken", "lobster-vector-service-token"
)
container.add_secret("LOBSTER_VECTOR_CLOUD_TOKEN", ecs.Secret.from_secrets_manager(chroma_token))
```

### Step 8: Upload Pre-Built Tarballs to S3

```bash
# Create public-read bucket (one time)
aws s3 mb s3://lobster-ontology-data --region us-east-1

# Set bucket policy for public read on v1/ prefix
aws s3api put-bucket-policy --bucket lobster-ontology-data --policy '{
    "Version": "2012-10-17",
    "Statement": [{
        "Sid": "PublicReadV1",
        "Effect": "Allow",
        "Principal": "*",
        "Action": "s3:GetObject",
        "Resource": "arn:aws:s3:::lobster-ontology-data/v1/*"
    }]
}'

# Upload tarballs (generated by scripts/build_ontology_embeddings.py)
aws s3 cp output/mondo_sapbert_768.tar.gz s3://lobster-ontology-data/v1/
aws s3 cp output/uberon_sapbert_768.tar.gz s3://lobster-ontology-data/v1/
aws s3 cp output/cell_ontology_sapbert_768.tar.gz s3://lobster-ontology-data/v1/

# Verify public access
curl -I https://lobster-ontology-data.s3.amazonaws.com/v1/mondo_sapbert_768.tar.gz
```

---

## 7. Cost Estimate

Monthly cost estimate for always-on ChromaDB server in us-east-1:

| Resource | Specification | Monthly Cost |
|----------|--------------|--------------|
| ECS Fargate (ARM64) | 4 vCPU, 16 GB RAM, always-on | ~$50-80 |
| EBS gp3 volume | 10 GB, 3000 IOPS baseline | ~$1 |
| Application Load Balancer | 1 ALB + LCU hours | ~$20 |
| Route53 | 1 hosted zone record | ~$0.50 |
| Secrets Manager | 1 secret, ~100 API calls/day | ~$0.50 |
| S3 (tarball hosting) | ~200 MB stored, minimal GET requests | ~$0.01 |
| **Total** | | **~$72-102/month** |

### Cost Optimization Notes

- **Scale-to-zero not recommended:** Cold start re-downloads ~150 MB and rebuilds HNSW indexes. Acceptable for dev, not production.
- **ARM64 saves ~20%** vs equivalent x86 Fargate pricing.
- **Single instance sufficient** until >1000 concurrent queries/second. ChromaDB HNSW search is single-digit milliseconds per query.
- **EBS vs EFS:** EBS is cheaper and sufficient for single-task attachment. EFS only needed if multiple tasks share storage.

---

## 8. Migration Path

### Phase 1: Documentation (This Spec)

- Document architecture, authentication, API design, deployment steps
- No code changes to Lobster AI engine
- No infrastructure changes

### Phase 2: Deploy ChromaDB Server (CLOD-V2-01)

**Scope:**
1. Create `VectorSearchStack` CDK stack with ChromaDB ECS service
2. Add `HttpClient` branch to `ChromaDBBackend._get_client()` (see Section 4)
3. Upload pre-built tarballs to S3 bucket
4. Configure Route53 for vector.omics-os.com
5. Store service token in Secrets Manager

**Validation:**
```bash
# Verify ChromaDB server is running
curl https://vector.omics-os.com/api/v1/heartbeat

# Verify collections are available (with auth)
curl -H "Authorization: Bearer <token>" \
     https://vector.omics-os.com/api/v1/collections

# Verify query works end-to-end
python -c "
import chromadb
from chromadb.config import Settings
client = chromadb.HttpClient(
    host='vector.omics-os.com', port=443, ssl=True,
    settings=Settings(
        chroma_client_auth_provider='chromadb.auth.token.TokenAuthClientProvider',
        chroma_client_auth_credentials='<token>',
    ),
)
coll = client.get_collection('mondo_v2024_01')
print(f'MONDO terms: {coll.count()}')
"
```

### Phase 3: Cognito-Authenticated Access (CLOD-V2-02)

**Scope:**
1. Add `LOBSTER_VECTOR_CLOUD_URL` and `LOBSTER_VECTOR_CLOUD_TOKEN` to Omics-OS Cloud ECS task definition
2. Verify that Omics-OS Cloud API routes vector search through the cloud ChromaDB
3. End-to-end test: user query in app.omics-os.com triggers cloud vector search
4. Monitor latency and error rates via CloudWatch

---

## 9. Open Questions for Phase 2

### Multi-Region Deployment

**Question:** Deploy in a single region (us-east-1) or replicate to eu-west-1 for European users?

**Recommendation:** Start with us-east-1 only. Omics-OS Cloud infrastructure is all in us-east-1. Cross-region latency for vector search (~50ms EU to US) is acceptable for the query pattern (not real-time streaming). Revisit when European customer count justifies the cost.

### Scaling Threshold

**Question:** When do we need a second ChromaDB instance?

**Recommendation:** ChromaDB HNSW handles ~1000+ queries/second for 65K vectors in memory. With our current user base (early stage), a single instance is sufficient for years. Monitor via CloudWatch `TargetResponseTime` metric on the ALB. Scale when p99 latency exceeds 500ms.

### Custom Ontology Collections

**Question:** Should enterprise users be able to upload custom ontology collections?

**Recommendation:** Not in the initial deployment. Custom collections would require:
- Write API access (currently read-only)
- Per-tenant collection namespacing
- Storage quota management
- Admin UI for collection management

This is a significant feature expansion. Defer to enterprise roadmap.

### ChromaDB Version Pinning

**Question:** How to handle ChromaDB version compatibility between build script and server?

**Recommendation:** Pin the ChromaDB Docker image tag (e.g., `chromadb/chroma:0.5.23`) rather than `latest`. Include a `.version` metadata file inside each tarball with the ChromaDB version used during build. The entrypoint script checks version compatibility before serving.

---

## 10. Reference Files

| File | Relevance |
|------|-----------|
| `lobster/core/vector/backends/chromadb_backend.py` | `_get_client()` will gain `HttpClient` branch for cloud mode |
| `lobster/core/vector/config.py` | `VectorSearchConfig` -- no changes needed (backend handles mode) |
| `lobster/core/vector/service.py` | `ONTOLOGY_COLLECTIONS` -- collection name source of truth |
| `lobster-cloud/infrastructure/` | CDK stacks for AWS deployment |
| `lobster-cloud/api/dependencies.py` | Cognito JWT validation middleware |
| `scripts/build_ontology_embeddings.py` | Build script that generates tarballs for S3 (Plan 06-02) |

---

*Specification for: vector.omics-os.com cloud deployment*
*Phase: 06-automation, Plan: 04*
*Completed: 2026-02-19*
