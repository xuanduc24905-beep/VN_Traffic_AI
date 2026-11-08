# Kế hoạch triển khai MLOps cho `vn_traffic_ai`

**Repo mới:** `vn_traffic_mlops` (song song, không thay thế `vn_traffic_ai`)
**Target:** Local WSL2
**Data scope:** V5 + V6 + V7 + raw sources (~10GB lên MinIO)
**Author:** xuanduc <xuanduc24905@gmail.com>

---

## Stack tổng quan

| Lớp | Công nghệ | Port |
|---|---|---|
| Object storage | MinIO (S3-compatible) | 9000 (API), 9001 (console) |
| Metadata DB | PostgreSQL | 5432 |
| Experiment tracking + Model registry | MLflow | 5000 |
| Orchestration | Prefect | 4200 |
| Distributed inference | Ray | 8265 |
| Serving | FastAPI | 8000 |
| UI demo | Streamlit | 8501 |
| Monitoring | Prometheus + Grafana | 9090, 3000 |
| Data version | DVC (remote = MinIO) | — |

---

## PHASE 0 — Chuẩn bị môi trường

### 0.1 Kiểm tra hệ thống
```bash
free -h                              # cần >= 8GB RAM rảnh
df -h ~                              # cần >= 30GB trống
docker --version
docker compose version
```

### 0.2 Kiểm tra port rảnh
```bash
for p in 9000 9001 5432 5000 4200 8000 8265 8501 9090 3000; do
  echo -n "Port $p: "
  ss -tln | grep -q ":$p " && echo "BUSY" || echo "free"
done
```

Nếu port busy → note lại, tôi sẽ đổi trong `docker-compose.yml`.

### 0.3 Cài tool CLI cần thiết
```bash
# MinIO client (upload data)
curl https://dl.min.io/client/mc/release/linux-amd64/mc \
  -o ~/.local/bin/mc && chmod +x ~/.local/bin/mc

# Python env cho DVC
pip install --user 'dvc[s3]' mlflow boto3 s3fs
```

---

## PHASE 1 — Scaffold project mới

### 1.1 Tạo repo local
```bash
cd ~
mkdir vn_traffic_mlops && cd vn_traffic_mlops
git init
git config user.name "xuanduc"
git config user.email "xuanduc24905@gmail.com"
git config --get user.name    # verify = xuanduc
```

### 1.2 Tạo cấu trúc folder
```bash
mkdir -p {src/{data,models,training,inference,serving,pipelines},configs/{data,model},notebooks,tests,infra,scripts,docs}
touch src/__init__.py src/{data,models,training,inference,serving,pipelines}/__init__.py
```

### 1.3 Viết `.gitignore`
Bao gồm: `.env`, `data/`, `runs/`, `logs/`, `*.pt`, `*.onnx`, `__pycache__/`, `.venv/`, `.dvc/cache/`, `mlruns/`, `prefect-data/`.

### 1.4 Viết `pyproject.toml` (thay `requirements.txt`)
Dependencies: `torch`, `ultralytics`, `dvc[s3]`, `mlflow`, `prefect`, `ray`, `fastapi`, `uvicorn`, `s3fs`, `boto3`, `psycopg2-binary`, `evidently`, `prometheus-client`.

### 1.5 Commit skeleton
```bash
git add .
git commit -m "init: scaffold vn_traffic_mlops project structure"
```

---

## PHASE 2 — Dựng infra (Docker Compose)

### 2.1 Viết `.env` (KHÔNG commit)
```
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=<strong-password>
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=<strong-password>
POSTGRES_DB=mlflow
MLFLOW_S3_ENDPOINT_URL=http://minio:9000
AWS_ACCESS_KEY_ID=${MINIO_ROOT_USER}
AWS_SECRET_ACCESS_KEY=${MINIO_ROOT_PASSWORD}
```

### 2.2 Viết `docker-compose.yml`
Services:
- `minio` — image `minio/minio`, volume `./infra/minio-data`
- `postgres` — image `postgres:16`, volume `./infra/pg-data`
- `mlflow` — custom build với `psycopg2` + `boto3`, backend Postgres, artifact store MinIO
- `createbuckets` — init container tạo bucket `vn-traffic`, `mlflow-artifacts`

### 2.3 Start stack
```bash
docker compose up -d
docker compose ps                    # tất cả Up
docker compose logs -f mlflow        # kiểm tra kết nối Postgres
```

### 2.4 Verify
- MinIO console: `http://localhost:9001` → login → thấy bucket `vn-traffic` và `mlflow-artifacts`.
- MLflow UI: `http://localhost:5000` → trang trắng OK.

### 2.5 Commit
```bash
git add docker-compose.yml infra/mlflow.Dockerfile .env.example
git commit -m "infra: thêm MinIO + Postgres + MLflow stack"
```

---

## PHASE 3 — Upload data lên MinIO

### 3.1 Config `mc` alias
```bash
mc alias set local http://localhost:9000 admin <password>
mc ls local/                         # thấy bucket
```

### 3.2 Upload theo thứ tự (tổng ~10GB)

| Thứ tự | Nguồn local | Đích MinIO | Size |
|---|---|---|---|
| 1 | `~/vn_traffic_ai/data/raw_soict2024/` | `local/vn-traffic/raw/soict2024/` | 29MB |
| 2 | `~/vn_traffic_ai/data/raw_soict_night/` | `local/vn-traffic/raw/soict_night/` | 57MB |
| 3 | `~/vn_traffic_ai/data/raw_vn_cctv/` | `local/vn-traffic/raw/vn_cctv/` | 119MB |
| 4 | `~/vn_traffic_ai/data/raw_vn_cantho/` | `local/vn-traffic/raw/vn_cantho/` | 4.7MB |
| 5 | `~/vn_traffic_ai/data/raw_vnv3/` | `local/vn-traffic/raw/vnv3/` | 6.4MB |
| 6 | `~/vn_traffic_ai/data/pseudo_v6/` | `local/vn-traffic/pseudo/v6/` | 233MB |
| 7 | `~/vn_traffic_ai/data/merged_v6/` | `local/vn-traffic/datasets/v6/` | 85MB |
| 8 | `~/vn_traffic_ai/data/merged_v7/` | `local/vn-traffic/datasets/v7/` | 54MB |
| 9 | `~/vn_traffic_ai/data/test_videos/` | `local/vn-traffic/test_videos/` | 5.5MB |
| 10 | `~/vn_traffic_ai/data/merged_v5/` | `local/vn-traffic/datasets/v5/` | 9.2GB (lâu nhất) |
| 11 | `~/vn_traffic_ai/weights/` (best only) | `local/vn-traffic/models/baseline/` | ~200MB |

Lệnh mẫu:
```bash
mc mirror ~/vn_traffic_ai/data/merged_v5/ local/vn-traffic/datasets/v5/ --overwrite
```

### 3.3 Verify
```bash
mc du local/vn-traffic/              # tổng ~10GB
mc ls local/vn-traffic/datasets/
```

---

## PHASE 4 — Setup DVC version control

### 4.1 Init
```bash
cd ~/vn_traffic_mlops
dvc init
dvc remote add -d minio s3://vn-traffic/dvc
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify minio access_key_id admin
dvc remote modify minio secret_access_key <password> --local
```

### 4.2 Track datasets (reference, không download)
```bash
dvc import-url s3://vn-traffic/datasets/v5 data/v5 --no-exec
dvc import-url s3://vn-traffic/datasets/v6 data/v6 --no-exec
dvc import-url s3://vn-traffic/datasets/v7 data/v7 --no-exec
```

### 4.3 Commit DVC files
```bash
git add .dvc/ data/*.dvc .dvcignore
git commit -m "data: version control V5/V6/V7 via DVC-MinIO"
```

---

## PHASE 5 — Migrate code core

### 5.1 Copy từ `vn_traffic_ai`
```bash
cp -r ~/vn_traffic_ai/src/* src/
cp -r ~/vn_traffic_ai/configs/* configs/
cp ~/vn_traffic_ai/tests/* tests/
cp ~/vn_traffic_ai/data/*.yaml configs/data/
```

### 5.2 Refactor dataset loader đọc từ MinIO
File: `src/data/s3_dataset.py` — dùng `s3fs` để stream ảnh thay vì mount local.
```python
import s3fs
fs = s3fs.S3FileSystem(
    endpoint_url="http://localhost:9000",
    key="admin",
    secret=os.environ["MINIO_PASSWORD"],
)
```

### 5.3 Thêm MLflow logging vào train script
File: `src/training/train_yolo.py` — thêm:
```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("yolov8-v5")
with mlflow.start_run():
    mlflow.log_params({"lr": lr, "epochs": epochs, "dataset": "v5"})
    # ... train ...
    mlflow.log_metrics({"mAP50": map50, "mAP50-95": map5095})
    mlflow.log_artifact("runs/train/weights/best.pt")
```

### 5.4 Smoke test
Train 1 epoch trên V5 subset → verify MLflow có record → verify artifact trên MinIO.

### 5.5 Commit
```bash
git add src/ configs/ tests/
git commit -m "feat: migrate training code với MLflow tracking + S3 dataset loader"
```

---

## PHASE 6 — Prefect pipeline

### 6.1 Viết flow `src/pipelines/train_flow.py`
Stages:
1. `fetch_data` — `dvc pull data/v5`
2. `validate_data` — check video-aware split, class balance
3. `train` — gọi `src/training/train_yolo.py`
4. `evaluate` — mAP trên test set
5. `register_model` — nếu mAP > threshold → push vào MLflow Model Registry

### 6.2 Deploy flow
```bash
prefect server start &
prefect deploy src/pipelines/train_flow.py:train_flow -n yolo-v5 -p default
```

### 6.3 Trigger
```bash
prefect deployment run 'train_flow/yolo-v5'
```

---

## PHASE 7 — Serving + UI

### 7.1 FastAPI
File: `src/serving/api.py` — load model từ MLflow Model Registry stage=`production`, expose `/predict` nhận ảnh base64.

### 7.2 Streamlit UI
Copy `ui/streamlit_app.py` → `src/serving/ui.py` → đổi để gọi FastAPI thay vì load model trực tiếp.

### 7.3 Thêm vào `docker-compose.yml`
Services `api` (FastAPI) + `ui` (Streamlit).

### 7.4 Commit + test
```bash
docker compose up -d api ui
curl -X POST localhost:8000/predict -F "image=@sample.jpg"
```

---

## PHASE 8 — Distributed pseudo-label với Ray

### 8.1 Ray cluster trong Docker
Service `ray-head` trong `docker-compose.yml`.

### 8.2 Viết job `src/inference/pseudo_label_ray.py`
Parallel inference trên CCTV frames → upload kết quả về `s3://vn-traffic/pseudo/v7/`.

### 8.3 Trigger qua Prefect
Thêm stage vào `train_flow`.

---

## PHASE 9 — Monitoring

### 9.1 Prometheus scrape
- FastAPI `/metrics` (dùng `prometheus-fastapi-instrumentator`)
- MinIO built-in metrics
- MLflow custom exporter

### 9.2 Grafana dashboard
- API latency p50/p95/p99
- Model confidence distribution theo ngày
- Dataset size growth

### 9.3 Data drift với Evidently
Weekly job: so phân bố class V5 vs CCTV incoming → alert nếu drift > threshold.

---

## PHASE 10 — CI/CD + Push GitHub

### 10.1 Tạo repo trên GitHub
User tự tạo: `github.com/xuanduc/vn_traffic_mlops` (private hoặc public tùy).

### 10.2 GitHub Actions workflow
File: `.github/workflows/ci.yml`
- Lint (ruff)
- Test (pytest)
- Build Docker images
- Push vào GHCR

### 10.3 Push lần đầu
```bash
# verify identity
git config user.name          # phải = xuanduc
git config user.email         # phải = xuanduc24905@gmail.com

git remote add origin git@github.com:xuanduc/vn_traffic_mlops.git
git push -u origin main
```

**Rule:** commit message không nhắc Claude/AI/assistant, không có trailer `Co-Authored-By`.

---

## Checklist tiến độ

- [ ] Phase 0: Môi trường sẵn sàng
- [ ] Phase 1: Project scaffold + commit đầu
- [ ] Phase 2: Docker stack chạy (MinIO + Postgres + MLflow)
- [ ] Phase 3: Data upload MinIO (~10GB)
- [ ] Phase 4: DVC track datasets
- [ ] Phase 5: Code migrate + MLflow logging + smoke test
- [ ] Phase 6: Prefect pipeline chạy end-to-end
- [ ] Phase 7: FastAPI + Streamlit chạy từ Docker
- [ ] Phase 8: Ray pseudo-label distributed
- [ ] Phase 9: Prometheus + Grafana + Evidently
- [ ] Phase 10: GitHub Actions CI + push repo

---

## Quyết định đã chốt

| Câu hỏi | Chốt |
|---|---|
| Deploy ở đâu | Local WSL2 |
| Data migrate | V5 + V6 + V7 + raw (~10GB) |
| Repo cũ | Giữ nguyên, chạy song song |
| Big-data stack | MinIO + Postgres + Ray (skip Spark/Hadoop/Kafka initially) |

---

## Ước lượng thời gian

| Phase | Thời gian |
|---|---|
| 0 + 1 + 2 | 2-3h |
| 3 (upload data) | 2-4h (tùy SSD/mạng) |
| 4 + 5 | 3-4h |
| 6 | 2-3h |
| 7 | 2-3h |
| 8 | 2h |
| 9 | 2-3h |
| 10 | 1-2h |
| **Tổng** | **16-24h** (chia 4-6 buổi) |
