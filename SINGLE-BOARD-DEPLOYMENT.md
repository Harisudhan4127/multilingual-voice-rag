# Deploying Multilingual Voice RAG on a Single Board Computer

This guide walks you through running the full stack (Qdrant + FastAPI backend + React frontend) on a single board computer (SBC) like Raspberry Pi 4/5, Orange Pi 5, or similar ARM64/x86 boards using Docker Compose.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| RAM | 4 GB | 8 GB+ |
| Storage | 16 GB SD card | 64 GB SSD/eMMC |
| OS | 64-bit Linux (Ubuntu/Debian) | Ubuntu 22.04+ or Debian 12+ |
| Network | Ethernet or Wi-Fi | Gigabit Ethernet |

### Supported Boards

- Raspberry Pi 4 (4GB/8GB) / Pi 5 (4GB/8GB)
- Orange Pi 5 / 5 Plus
- Khadas Edge2 / VIM4
- NVIDIA Jetson Nano / Orin
- Any x86_64 or ARM64 SBC running 64-bit Linux

---

## Step 1: Prepare the SBC Operating System

Flash a 64-bit Linux image to your SD card/eMMC/SSD. Recommended: **Ubuntu Server 22.04/24.04** or **Raspberry Pi OS (64-bit)**.

```bash
# After booting, update the system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y curl git wget htop
```

---

## Step 2: Install Docker & Docker Compose

```bash
# Install Docker via official convenience script
curl -fsSL https://get.docker.com | sudo sh

# Add your user to the docker group (avoids needing sudo)
sudo usermod -aG docker $USER

# Log out and log back in, then verify
docker --version
docker compose version
```

> **ARM64 boards (Raspberry Pi, Orange Pi):** Docker images will automatically pull the ARM64 variants. No extra configuration needed.

---

## Step 3: Clone the Repository

```bash
git clone https://github.com/<your-org>/multilingual-voice-rag.git
cd multilingual-voice-rag
```

---

## Step 4: Configure Environment Variables

The system works **fully offline with mock providers** by default. No API keys are required for demo/testing.

```bash
# Copy the example env file
cp backend/.env.example backend/.env
```

### Option A: Offline / Demo Mode (Default)

Leave `.env` as-is. All providers run in mock mode:

```
STT_PROVIDER=mock      # Deterministic mock speech-to-text
LLM_PROVIDER=mock      # Deterministic extractive mock LLM
DATASET_PROVIDER=synthetic  # 17 hand-written multilingual passages
```

### Option B: Real Providers (Optional)

Edit `backend/.env` to enable real AI providers:

```bash
# For real speech-to-text via Sarvam AI
STT_PROVIDER=sarvam
SARVAM_API_KEY=your-sarvam-api-key

# For real LLM (OpenAI-compatible API)
LLM_PROVIDER=real
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

### SBC Performance Tuning (Optional)

For boards with limited RAM (4GB), add these to `backend/.env`:

```bash
# Reduce model loading and retrieval concurrency
TOP_K_DENSE=10
TOP_K_BM25=10
TOP_K_RERANK=3
MSMARCO_MAX_DOCS=200
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## Step 5: Build and Launch

```bash
# Build all images and start all services
docker compose up --build -d
```

This builds three containers:

| Service | Description | Port |
|---|---|---|
| `qdrant` | Vector database | 6333, 6334 |
| `backend` | FastAPI RAG pipeline | 8000 |
| `frontend` | React SPA + nginx reverse proxy | 5173 → 80 |

> **First build note:** The backend Dockerfile downloads ~800MB of Python dependencies and ~500MB of ML models. On a Raspberry Pi 4, this takes approximately 20-40 minutes. On Pi 5 or Orange Pi 5, it takes ~10-20 minutes.

---

## Step 6: Verify All Services Are Running

```bash
# Check container status
docker compose ps

# Check backend health
curl http://localhost:8000/api/v1/health

# Check Qdrant
curl http://localhost:6333/collections

# View logs
docker compose logs -f
```

Expected health response:

```json
{
  "status": "healthy",
  "vector_store": "connected",
  "embedder": "ready"
}
```

---

## Step 7: Access the Application

Open a browser on any device on the same network:

| URL | Description |
|---|---|
| `http://<SBC-IP>:5173` | Frontend UI |
| `http://<SBC-IP>:8000/docs` | Backend API docs (Swagger) |
| `http://<SBC-IP>:6333/dashboard` | Qdrant dashboard |

Find your SBC's IP address:

```bash
hostname -I
```

---

## Step 8: Test the Pipeline

### Text Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Retrieval Augmented Generation?"}'
```

### Voice Query (with mock transcript)

```bash
curl -X POST http://localhost:8000/api/v1/voice \
  -F "mock_transcript=Tell me about multilingual AI"
```

### Via Frontend UI

1. Open `http://<SBC-IP>:5173`
2. Type a query in the text input and press Enter
3. Click the microphone button to record audio (browser permission required)

---

## Resource Usage Estimates

| Service | RAM (mock mode) | RAM (real providers) | CPU |
|---|---|---|---|
| Qdrant | ~150 MB | ~150 MB | Low |
| Backend (startup) | ~800 MB | ~800 MB | High |
| Backend (idle) | ~500 MB | ~600 MB | Low |
| Backend (query) | ~600 MB | ~700 MB | Medium-High |
| Frontend (nginx) | ~10 MB | ~10 MB | Negligible |
| **Total (idle)** | **~660 MB** | **~760 MB** | -- |
| **Total (peak)** | **~960 MB** | **~1.1 GB** | -- |

> On a 4GB board, this leaves plenty of headroom. On a 2GB board, it will be tight -- consider using swap.

---

## Enable Swap (for 2GB boards)

```bash
# Create a 2GB swap file
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## Useful Commands

```bash
# Start services
docker compose up -d

# Stop services
docker compose down

# Rebuild after code changes
docker compose up --build -d

# View logs (all services)
docker compose logs -f

# View logs (backend only)
docker compose logs -f backend

# Restart a single service
docker compose restart backend

# Check resource usage
docker stats

# Remove everything (including data volumes)
docker compose down -v
```

---

## Persistent Data

| Data | Location | Description |
|---|---|---|
| Qdrant vectors | Docker volume `qdrant_data` | Vector embeddings + indexed chunks |
| Processed chunks | `./data/processed/chunks.jsonl` | Chunked document data |
| Raw datasets | `./data/raw/` | Source dataset files |
| Backend models | Docker image layers | Downloaded at build time, baked into image |

To back up vector data:

```bash
docker compose exec qdrant tar czf /tmp/qdrant-backup.tar.gz /qdrant/storage
docker cp voice-rag-qdrant:/tmp/qdrant-backup.tar.gz ./backups/
```

---

## Networking Tips

### Access from Other Devices on LAN

Ensure the SBC firewall allows incoming connections on ports 5173 and 8000:

```bash
# UFW example
sudo ufw allow 5173/tcp
sudo ufw allow 8000/tcp
sudo ufw allow 6333/tcp
```

### Change Frontend Port

Edit `docker-compose.yml` line 48:

```yaml
ports:
  - "80:80"    # Change 5173 to 80 for direct browser access
```

### Reverse Proxy with Traefik/Nginx (Optional)

For production, put an nginx reverse proxy in front with HTTPS:

```nginx
server {
    listen 443 ssl;
    server_name rag.yourdomain.com;

    location / {
        proxy_pass http://localhost:5173;
    }

    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_read_timeout 120s;
        client_max_body_size 25M;
    }
}
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Build fails on ARM64 | Ensure 64-bit OS. Run `uname -m` -- should show `aarch64` |
| Out of memory during build | Add swap (see above) or build on a more powerful machine and transfer images |
| Backend healthcheck fails | Check `docker compose logs backend` -- models may still be downloading on first run |
| Qdrant connection refused | Wait for Qdrant healthcheck to pass; backend retries automatically |
| Frontend shows "not ready" | Backend is still starting (model loading). Check health: `curl localhost:8000/api/v1/health` |
| Slow queries | Normal on Pi 4 (~3-8s). Pi 5 is faster (~1-3s). Use `all-MiniLM-L6-v2` for lighter load |
| Port already in use | Run `docker compose down` first, or change ports in `docker-compose.yml` |

---

## Architecture Overview

```
Browser ──> Frontend (nginx:80) ──/api──> Backend (FastAPI:8000) ──> Qdrant (6333)
                                        │
                                        ├── Embeddings (sentence-transformers)
                                        ├── BM25 (in-memory)
                                        ├── Reranker (cross-encoder)
                                        └── LLM / STT (mock or real)
```

All ML models run on CPU. The pipeline follows: **Query → Guardrails → Hybrid Search (Dense + BM25) → RRF Fusion → Reranking → LLM Generation → Grounding Check → Response**.

---

## Full Restart from Scratch

```bash
# Stop and remove all containers, networks, and volumes
docker compose down -v

# Remove built images
docker image prune -af

# Rebuild from scratch
docker compose up --build
```
