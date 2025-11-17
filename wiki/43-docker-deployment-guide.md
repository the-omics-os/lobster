# Docker Deployment Guide

Comprehensive guide to running Lobster AI in Docker containers for development, staging, and production environments.

## Table of Contents

- [Quick Start](#quick-start)
- [Container Modes](#container-modes)
- [Configuration](#configuration)
- [Deployment Patterns](#deployment-patterns)
- [Volume Management](#volume-management)
- [Troubleshooting](#troubleshooting)
- [Production Best Practices](#production-best-practices)

---

## Quick Start

### Prerequisites

- Docker 20.10+ installed
- Docker Compose 2.0+ (optional, for multi-service setups)
- `.env` file configured with API keys

### 3-Command Setup

```bash
# 1. Clone repository
git clone https://github.com/the-omics-os/lobster.git
cd lobster

# 2. Configure API keys
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY or AWS Bedrock credentials

# 3. Run CLI
make docker-build
make docker-run-cli
```

**You're now running Lobster in a container!**

---

## Container Modes

Lobster supports two Docker deployment modes:

### Mode 1: CLI Container (Default)

**Use case**: Interactive analysis, local development, automation scripts

```bash
# Interactive chat mode
docker run -it --rm \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  omicsos/lobster:latest chat

# Single query mode (automation)
docker run --rm \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  omicsos/lobster:latest query "download GSE12345"

# Show help
docker run --rm omicsos/lobster:latest --help
```

### Mode 2: FastAPI Server Container

**Use case**: Web service, API access, cloud deployments

```bash
# Run server
docker run -d \
  --name lobster-api \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  omicsos/lobster:server

# Check health
curl http://localhost:8000/health

# Stop server
docker stop lobster-api
```

---

## Configuration

### Environment Variables

Required variables (choose ONE LLM provider):

```bash
# Option 1: Claude API
ANTHROPIC_API_KEY=sk-ant-api03-your-key

# Option 2: AWS Bedrock
AWS_BEDROCK_ACCESS_KEY=AKIA...
AWS_BEDROCK_SECRET_ACCESS_KEY=your-secret-key
```

Optional variables:

```bash
# Enhanced literature search
NCBI_API_KEY=your-ncbi-key

# Cloud mode
LOBSTER_CLOUD_KEY=your-cloud-api-key

# Server settings (for FastAPI mode)
PORT=8000
HOST=0.0.0.0
DEBUG=False

# Data processing
LOBSTER_MAX_FILE_SIZE_MB=500
```

### .env File Setup

```bash
# Create from template
cp .env.example .env

# Edit with your editor
nano .env  # or vim, code, etc.
```

**Security Note**: Never commit `.env` files to version control!

---

## Deployment Patterns

### Pattern 1: Local Development

**Scenario**: Testing on your laptop

```bash
# docker-compose.yml approach
docker-compose run --rm lobster-cli

# Direct Docker approach
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v lobster-workspace:/app/.lobster_workspace \
  --env-file .env \
  omicsos/lobster:latest chat
```

**Advantages**:
- Fast iteration
- Local data access
- No network latency

### Pattern 2: Automation & CI/CD

**Scenario**: GitHub Actions, Jenkins, automated pipelines

```bash
# Example: GitHub Actions workflow
- name: Run Lobster analysis
  run: |
    docker run --rm \
      -e ANTHROPIC_API_KEY=${{ secrets.ANTHROPIC_API_KEY }} \
      -v $(pwd)/data:/app/data \
      omicsos/lobster:latest query "load data.h5ad and create UMAP"
```

**docker-compose for CI**:

```yaml
# docker-compose.ci.yml
version: '3.8'
services:
  lobster:
    image: omicsos/lobster:latest
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./data:/app/data
      - ./results:/app/results
```

```bash
# Run in CI
docker-compose -f docker-compose.ci.yml run lobster query "analyze data"
```

### Pattern 3: AWS ECS/Fargate

**Scenario**: Production cloud deployment with auto-scaling

**Task Definition** (`lobster-task-definition.json`):

```json
{
  "family": "lobster-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "4096",
  "memory": "8192",
  "containerDefinitions": [
    {
      "name": "lobster-server",
      "image": "omicsos/lobster:server",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "secrets": [
        {
          "name": "ANTHROPIC_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789:secret:lobster/api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/lobster",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

**Deploy**:

```bash
# Register task definition
aws ecs register-task-definition --cli-input-json file://lobster-task-definition.json

# Create service
aws ecs create-service \
  --cluster your-cluster \
  --service-name lobster-api \
  --task-definition lobster-api \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx]}"
```

### Pattern 4: Kubernetes (GKE, EKS, AKS)

**Scenario**: Container orchestration with health checks, auto-scaling

**Deployment** (`k8s-deployment.yaml`):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lobster-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: lobster-api
  template:
    metadata:
      labels:
        app: lobster-api
    spec:
      containers:
      - name: lobster
        image: omicsos/lobster:server
        ports:
        - containerPort: 8000
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: lobster-secrets
              key: anthropic-api-key
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 40
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 20
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: lobster-api-service
spec:
  selector:
    app: lobster-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

**Deploy**:

```bash
# Create secret
kubectl create secret generic lobster-secrets \
  --from-literal=anthropic-api-key=sk-ant-api03-your-key

# Apply deployment
kubectl apply -f k8s-deployment.yaml

# Check status
kubectl get pods -l app=lobster-api
kubectl logs -l app=lobster-api -f
```

### Pattern 5: Docker Swarm

**Scenario**: Multi-node orchestration with built-in load balancing

**Stack File** (`docker-swarm.yml`):

```yaml
version: '3.8'

services:
  lobster-api:
    image: omicsos/lobster:server
    ports:
      - "8000:8000"
    secrets:
      - anthropic_api_key
    environment:
      - ANTHROPIC_API_KEY_FILE=/run/secrets/anthropic_api_key
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

secrets:
  anthropic_api_key:
    external: true

networks:
  lobster-network:
    driver: overlay
```

**Deploy**:

```bash
# Initialize swarm (if needed)
docker swarm init

# Create secret
echo "sk-ant-api03-your-key" | docker secret create anthropic_api_key -

# Deploy stack
docker stack deploy -c docker-swarm.yml lobster

# Check services
docker service ls
docker service ps lobster_lobster-api
```

---

## Volume Management

### Understanding Volumes

Lobster uses 3 types of persistent storage:

| Volume | Purpose | Size Estimate | Backup Priority |
|--------|---------|---------------|-----------------|
| **workspace** | Analysis sessions, plots, notebooks | 1-10 GB | HIGH - Contains results |
| **cache** | Downloaded GEO datasets, publications | 5-50 GB | MEDIUM - Can redownload |
| **data** | User input files | Varies | LOW - User controls |

### Volume Configurations

#### Named Volumes (Recommended for Production)

```yaml
# docker-compose.yml
volumes:
  lobster-workspace:
    driver: local
  lobster-cache:
    driver: local
```

**Advantages**:
- Managed by Docker
- Survive container deletion
- Efficient performance

**Backup**:

```bash
# Backup workspace
docker run --rm \
  -v lobster-workspace:/source \
  -v $(pwd)/backup:/backup \
  alpine tar czf /backup/workspace-$(date +%Y%m%d).tar.gz -C /source .

# Restore workspace
docker run --rm \
  -v lobster-workspace:/target \
  -v $(pwd)/backup:/backup \
  alpine tar xzf /backup/workspace-20250116.tar.gz -C /target
```

#### Bind Mounts (Development)

```bash
# Mount local directory
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/workspace:/app/.lobster_workspace \
  omicsos/lobster:latest chat
```

**Advantages**:
- Direct file access
- Easy debugging
- IDE integration

**Disadvantages**:
- Slower on macOS/Windows
- Permission issues possible

#### NFS/Cloud Volumes (Enterprise)

**AWS EFS**:

```bash
# Create EFS filesystem
aws efs create-file-system --performance-mode generalPurpose

# Mount in ECS task definition
"mountPoints": [
  {
    "sourceVolume": "efs-storage",
    "containerPath": "/app/.lobster_workspace"
  }
],
"volumes": [
  {
    "name": "efs-storage",
    "efsVolumeConfiguration": {
      "fileSystemId": "fs-12345678"
    }
  }
]
```

---

## Troubleshooting

### Issue 1: Container exits immediately

**Symptom**:
```bash
$ docker run omicsos/lobster:latest
# Container exits with no output
```

**Cause**: Missing environment variables

**Solution**:
```bash
# Check logs
docker logs <container-id>

# Provide API key
docker run --rm \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  omicsos/lobster:latest --help
```

### Issue 2: "Permission denied" errors

**Symptom**:
```
PermissionError: [Errno 13] Permission denied: '/app/.lobster_workspace/data'
```

**Cause**: Volume ownership mismatch

**Solution**:
```bash
# Option 1: Fix volume permissions
docker run --rm \
  -v lobster-workspace:/workspace \
  alpine chown -R 1000:1000 /workspace

# Option 2: Run as root (NOT RECOMMENDED for production)
docker run --user root ...
```

### Issue 3: Out of disk space

**Symptom**:
```
Error: no space left on device
```

**Solution**:
```bash
# Clean up Docker system
docker system prune -a --volumes

# Remove unused images
docker image prune -a

# Check volume sizes
docker system df -v
```

### Issue 4: Slow performance on macOS

**Cause**: Docker Desktop file sharing overhead

**Solution**:
```bash
# Use named volumes instead of bind mounts
docker-compose up  # Uses named volumes from compose file

# OR: Use osxfs caching
docker run -v $(pwd)/data:/app/data:delegated ...
```

### Issue 5: Cannot connect to FastAPI server

**Symptom**:
```bash
curl: (7) Failed to connect to localhost port 8000: Connection refused
```

**Solution**:
```bash
# Check if container is running
docker ps | grep lobster

# Check container logs
docker logs lobster-server

# Verify port mapping
docker port lobster-server

# Test from inside container
docker exec lobster-server curl http://localhost:8000/health
```

### Issue 6: Import errors after building

**Symptom**:
```
ModuleNotFoundError: No module named 'scanpy'
```

**Cause**: Build cache issue or pyproject.toml changes

**Solution**:
```bash
# Rebuild without cache
docker build --no-cache -t omicsos/lobster:latest .

# Verify installation inside container
docker run --rm omicsos/lobster:latest python -c "import scanpy; print(scanpy.__version__)"
```

---

## Production Best Practices

### 1. Security

```bash
# ✅ Use secrets management (not environment variables)
docker secret create anthropic_key - < api_key.txt

# ✅ Run as non-root user (already configured in Dockerfile)
USER lobsteruser

# ✅ Security scanning
docker scan omicsos/lobster:latest
trivy image omicsos/lobster:latest

# ✅ Read-only filesystem where possible
docker run --read-only \
  --tmpfs /tmp:rw,noexec,nosuid \
  omicsos/lobster:latest
```

### 2. Resource Limits

```yaml
# docker-compose.yml
services:
  lobster-server:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

### 3. Health Checks

```dockerfile
# Already included in Dockerfile.server
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

### 4. Logging

```yaml
# docker-compose.yml
services:
  lobster-server:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

**Structured logging** (if using server mode):
```python
# Logs are JSON-formatted in production
{"timestamp": "2025-01-16T10:30:00Z", "level": "INFO", "message": "Analysis complete"}
```

### 5. Multi-Stage Builds

Already implemented in Dockerfile:
- **Stage 1 (base)**: System dependencies
- **Stage 2 (builder)**: Python packages
- **Stage 3 (runtime)**: Minimal production image

**Result**: ~2GB final image vs ~5GB if single-stage

### 6. CI/CD Integration

Automated via `.github/workflows/docker.yml`:
- Build on every PR
- Security scan with Trivy
- Multi-architecture builds (amd64, arm64)
- Publish to Docker Hub on tags

### 7. Monitoring & Observability

**Prometheus metrics** (future enhancement):
```yaml
# Expose metrics endpoint
ports:
  - "8000:8000"  # API
  - "9090:9090"  # Metrics
```

**Grafana dashboards** (future):
- Request latency
- Analysis completion rate
- Memory/CPU usage
- Download queue length

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Docker Host                          │
│                                                             │
│  ┌────────────────────┐      ┌────────────────────┐        │
│  │ Lobster CLI        │      │ Lobster API Server │        │
│  │ (interactive)      │      │ (FastAPI port 8000)│        │
│  └─────────┬──────────┘      └─────────┬──────────┘        │
│            │                           │                    │
│            ├───────────────┬───────────┤                    │
│            │               │           │                    │
│       ┌────▼────┐    ┌────▼────┐  ┌──▼───────┐            │
│       │Workspace│    │  Cache  │  │   Data   │            │
│       │ Volume  │    │ Volume  │  │ Bind Mt  │            │
│       └─────────┘    └─────────┘  └──────────┘            │
│                                                             │
│  ┌──────────────────────────────────────────┐              │
│  │          Docker Network (bridge)         │              │
│  └──────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
                        │
                        │ Internet
                        ▼
               ┌────────────────────┐
               │  External Services │
               │  - Anthropic API   │
               │  - AWS Bedrock     │
               │  - NCBI PubMed     │
               │  - GEO Database    │
               └────────────────────┘
```

---

## Quick Reference

### Common Commands

```bash
# Build
make docker-build                      # Build both images
docker build -t omicsos/lobster .      # Build CLI manually

# Run CLI
make docker-run-cli                    # Via Makefile
docker-compose run --rm lobster-cli    # Via compose
docker run -it --rm omicsos/lobster:latest chat  # Direct

# Run Server
make docker-run-server                 # Via Makefile
docker-compose up lobster-server       # Via compose
docker run -d -p 8000:8000 omicsos/lobster:server  # Direct

# Inspect
docker ps                              # Running containers
docker logs lobster-api                # View logs
docker exec -it lobster-api bash       # Shell access

# Cleanup
docker-compose down                    # Stop compose services
docker stop lobster-api                # Stop server
docker system prune -a                 # Clean everything
```

### Environment Variable Summary

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `ANTHROPIC_API_KEY` | Yes* | - | Claude API access |
| `AWS_BEDROCK_ACCESS_KEY` | Yes* | - | AWS Bedrock access |
| `AWS_BEDROCK_SECRET_ACCESS_KEY` | Yes* | - | AWS Bedrock secret |
| `NCBI_API_KEY` | No | - | Enhanced PubMed |
| `LOBSTER_CLOUD_KEY` | No | - | Cloud processing |
| `PORT` | No | 8000 | Server port |
| `HOST` | No | 0.0.0.0 | Server host |
| `LOBSTER_MAX_FILE_SIZE_MB` | No | 500 | Max file size |

*One of ANTHROPIC or AWS credentials required

---

## Next Steps

- [Installation Guide](02-installation.md) - Non-Docker installation
- [Configuration Guide](03-configuration.md) - Advanced settings
- [Troubleshooting](28-troubleshooting.md) - Common issues
- [Architecture Overview](18-architecture-overview.md) - How Lobster works

---

**Questions?** Report issues at https://github.com/the-omics-os/lobster/issues or email info@omics-os.com
