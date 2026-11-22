# Hướng dẫn cài đặt & chạy trên máy khác

Tài liệu này giúp người khác clone repo và chạy được project **VN Traffic AI** (detection → tracking → counting) trên máy của họ. Mô tả dự án, kết quả và methodology xem [README.md](README.md).

Có 3 mức, chọn theo nhu cầu:

| Mức | Làm được gì | Cần |
|---|---|---|
| **A. Demo** | Chạy Streamlit / CLI đếm xe trên video | Code + 1 file weights + 1 video |
| **B. Eval** | Chạy lại đánh giá mAP trên tập test | A + dataset test (tải/dựng bằng script) |
| **C. Train** | Train lại các model V5/V5+/V7/... | B + dataset train (~15 GB) + GPU |

> Hầu hết người dùng chỉ cần **mức A**.

---

## 1. Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|---|---|
| OS | Ubuntu 20.04+, Windows 10/11 qua **WSL2**, hoặc macOS 12+ |
| Python | 3.10 – 3.12 (đã chạy thử trên 3.12) |
| GPU | NVIDIA ≥ 6 GB VRAM + driver CUDA 11.8+ (khuyến nghị). Không có GPU vẫn chạy demo được nhưng chậm ~30× |
| RAM | ≥ 8 GB |
| Ổ đĩa | Mức A: ~3 GB · Mức B: ~5 GB · Mức C: ≥ 30 GB |
| Khác | `git`, `ffmpeg` |

> Windows native chưa được kiểm thử vì script `.sh` dùng bash — hãy dùng WSL2.

---

## 2. Cài đặt (mức A — demo)

### 2.1. Cài công cụ hệ thống

```bash
# Ubuntu / WSL2
sudo apt update && sudo apt install -y git ffmpeg python3-venv python3-pip
```

```bash
# macOS
brew install git ffmpeg python
```

### 2.2. Clone repo

```bash
git clone https://github.com/xuanduc24905-beep/VN_Traffic_AI.git
cd VN_Traffic_AI
```

### 2.3. Tạo môi trường ảo

Dùng venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Hoặc conda:

```bash
conda create -n vn_traffic python=3.11 -y
conda activate vn_traffic
```

### 2.4. Cài PyTorch (làm TRƯỚC `requirements.txt`)

Chọn **một** lệnh phù hợp với máy:

```bash
# GPU NVIDIA, CUDA 12.4
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# GPU NVIDIA, CUDA 11.8
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

# Không có GPU / macOS
pip install torch==2.5.1 torchvision==0.20.1
```

Kiểm tra driver GPU nếu dùng CUDA: `nvidia-smi`.

### 2.5. Cài thư viện còn lại

```bash
pip install -r requirements.txt
```

### 2.6. Kiểm tra cài đặt

```bash
python -c "import torch, ultralytics, cv2; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| ultralytics', ultralytics.__version__)"
```

`CUDA: True` nghĩa là GPU dùng được. `False` thì pipeline vẫn chạy trên CPU.

---

## 3. Tải weights

Repo **không commit file `.pt`**. Tải về từ GitHub Releases rồi đặt đúng đường dẫn:

```bash
# Model V5+ (imgsz=960, ~22 MB)
mkdir -p runs/detect/train_v8s_v5plus_960/weights
wget -O runs/detect/train_v8s_v5plus_960/weights/best.pt \
  https://github.com/xuanduc24905-beep/VN_Traffic_AI/releases/download/v1.0-weights/best_v5plus_960.pt

# Model V7 (imgsz=1280, ~22 MB) — tùy chọn
mkdir -p runs/detect/train_v8s_v7_1280/weights
wget -O runs/detect/train_v8s_v7_1280/weights/best.pt \
  https://github.com/xuanduc24905-beep/VN_Traffic_AI/releases/download/v1.0-weights/best_v7_1280.pt

# COCO baseline (để so sánh trong dashboard)
mkdir -p weights
wget -O weights/yolov8s.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt
```

Kiểm tra checksum (Linux):

```bash
sha256sum runs/detect/train_v8s_v5plus_960/weights/best.pt
# b1e9b1e583a3fa0d0dfccdd502fd234e670a46a318998bfb6e81db7ecd6d0a0d
sha256sum runs/detect/train_v8s_v7_1280/weights/best.pt
# 199848ec28025b1ee3b76993780d109ccdb40c52a06da9122ef161569230721b
```

> Streamlit tự quét `weights/**/*.pt` và `runs/**/best.pt`, nên đặt file đúng các thư mục trên là dashboard tự thấy.
>
> Model cuối của tiểu luận (`r_clspw`) chưa nằm trong release `v1.0-weights`. Muốn dùng, chủ repo cần upload `runs/detect/r_clspw/weights/best.pt` lên Releases rồi bổ sung link vào đây.

---

## 4. Chạy demo

Chuẩn bị 1 video giao thông (`.mp4/.avi/.mov`), ví dụ đặt ở `data/test_videos/raw/demo_traffic.mp4`.

### Cách 1 — Dashboard Streamlit

```bash
streamlit run ui/streamlit_app.py
```

Mở http://localhost:8501, upload video, chọn weights ở sidebar, bấm chạy. Cổng bận thì thêm `--server.port 8502`.

Chạy trên server/WSL không có trình duyệt thì thêm `--server.headless true` rồi mở URL từ máy host.

### Cách 2 — CLI

```bash
python -m src.pipeline.run \
  --video data/test_videos/raw/demo_traffic.mp4 \
  --weights runs/detect/train_v8s_v5plus_960/weights/best.pt \
  --tracker bytetrack.yaml \
  --imgsz 960 \
  --counting-config configs/counting_zones.json \
  --video-key demo_traffic \
  --out-csv results/tables/demo_events.csv \
  --out-video results/videos/demo_out.mp4
```

Không có GPU: thêm `--imgsz 640` để chạy nhanh hơn. Vạch đếm của từng video khai báo trong [configs/counting_zones.json](configs/counting_zones.json) (khóa `--video-key`); video mới thì thêm 1 mục tương ứng.

### Chạy unit test

```bash
pytest tests -q
```

---

## 5. Mức B — chạy lại đánh giá (tùy chọn)

### 5.1. Sửa đường dẫn tuyệt đối (BẮT BUỘC)

Nhiều file `data/*.yaml` và script đang ghi cứng `/home/xuand/vn_traffic_ai`. Trên máy khác phải thay bằng đường dẫn repo của bạn, chạy ở thư mục gốc repo:

```bash
OLD=/home/xuand/vn_traffic_ai
NEW="$(pwd)"
grep -rlZ "$OLD" --include='*.yaml' --include='*.sh' --include='*.py' \
  --exclude-dir=.git --exclude-dir=merged --exclude-dir='merged_*' . \
  | xargs -0 sed -i "s#$OLD#$NEW#g"
```

macOS dùng `sed -i ''` thay cho `sed -i`.

Cách khác không sửa file: tạo symlink để đường dẫn cũ trỏ tới repo:

```bash
sudo mkdir -p /home/xuand
sudo ln -s "$(pwd)" /home/xuand/vn_traffic_ai
```

### 5.2. Dataset đánh giá

Dataset ảnh **không có trong git** (`data/merged_*/`, `data/raw*/`, `*.zip` bị ignore). Có hai nguồn nhỏ dùng để eval:

| Bộ | Cách có |
|---|---|
| VN Vehicles Aerial (635 ảnh val+test) | `python scripts/data_prep/download_vn_aerial.py` (cần file `data/vietnam-vehicles.ndjson` từ Ultralytics Platform) |
| DETRAC bus/truck subset | Đã nằm trong git: `data/detrac_bustruck_subset/` |
| Tập V5/V5+ (SOICT24, CanTho, VNv3) | Tải dataset gốc từ Roboflow/SOICT rồi dựng bằng `scripts/data_prep/import_*.py` và `rebuild_v5_by_video.py` — hỏi chủ repo để lấy bản đã dựng |

Sau khi có data, chạy:

```bash
python scripts/eval/eval_all_models_full.py
```

---

## 6. Mức C — train lại

Cần dataset train đã merge (V5+ khoảng 15 GB) và GPU ≥ 12 GB VRAM để dùng `batch=16, imgsz=960`.

```bash
python scripts/training/v5_plus_960.py        # V5+ (~2 giờ trên RTX 3500 Ada)
bash scripts/phase4_v7/run_pipeline.sh        # V7: import → build → audit leak → train 1280 → eval
```

Thiếu VRAM thì giảm `batch` (16→8) hoặc `imgsz` (960→640) trong script.

---

## 7. Xử lý lỗi thường gặp

| Triệu chứng | Cách xử lý |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Chạy từ **thư mục gốc repo**; hoặc `export PYTHONPATH=.` |
| `ModuleNotFoundError: torch` | Chưa làm bước 2.4, hoặc chưa kích hoạt venv/conda |
| `torch.cuda.is_available()` = False dù có GPU | Cài sai bản torch (CPU) — gỡ `pip uninstall torch torchvision` rồi cài lại đúng CUDA ở 2.4; kiểm tra `nvidia-smi`. Với WSL2 cần driver NVIDIA bản Windows hỗ trợ WSL |
| `CUDA out of memory` | Giảm `--imgsz` hoặc `batch`; bật FP16 trong sidebar |
| `lap` build lỗi | `sudo apt install build-essential` rồi `pip install lap`; hoặc `pip install lapx` |
| Video output trắng trong Streamlit | Chưa cài `ffmpeg` |
| `FileNotFoundError: .../best.pt` | Chưa tải weights (mục 3) hoặc sai đường dẫn |
| `FileNotFoundError: /home/xuand/...` | Chưa thay đường dẫn tuyệt đối (mục 5.1) |
| `Không tìm thấy file .pt nào` trong Streamlit | Đặt weights vào `weights/` hoặc `runs/detect/<tên>/weights/best.pt` |
| `ImportError: libGL.so.1` (opencv) | `sudo apt install -y libgl1 libglib2.0-0` |
| Streamlit không mở được từ WSL2 | Mở `http://localhost:8501` trên trình duyệt Windows; hoặc dùng IP của WSL |

---

## 8. Tóm tắt nhanh (copy–paste, Ubuntu + GPU CUDA 12.4)

```bash
sudo apt update && sudo apt install -y git ffmpeg python3-venv python3-pip libgl1
git clone https://github.com/xuanduc24905-beep/VN_Traffic_AI.git && cd VN_Traffic_AI
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
mkdir -p runs/detect/train_v8s_v5plus_960/weights
wget -O runs/detect/train_v8s_v5plus_960/weights/best.pt \
  https://github.com/xuanduc24905-beep/VN_Traffic_AI/releases/download/v1.0-weights/best_v5plus_960.pt
streamlit run ui/streamlit_app.py
```
