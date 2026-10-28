"""
Bước 4: TRAIN model YOLOv8s trên dataset VN.

Đây là bước lâu nhất — ~2 giờ trên GPU RTX 3500 Ada 12GB, 40 epoch.
CPU sẽ mất ~2-3 ngày. Nếu không có GPU, chạy test với số epoch nhỏ (epochs=1)
để xem flow trước, sau đó mượn máy có GPU.

Flow bên trong Ultralytics khi gọi model.train(...):
    1. Load pretrained weights yolov8s.pt (COCO 80 class)
       → Ultralytics tự download từ GitHub release lần đầu
    2. Reset detection head (lớp cuối) từ 80 class → 4 class của ta
       → gọi là "transfer learning" — giữ backbone đã học feature chung
    3. Mỗi epoch:
         a. Lặp qua tất cả ảnh train, chia thành batch (mặc định batch=16)
         b. Mỗi batch:
            - Forward: model dự đoán bbox + class
            - Loss: so sánh với ground truth → ra 3 loss (box, cls, dfl)
            - Backward: tính gradient, update weights
         c. Cuối epoch: chạy validation set, log mAP
         d. Nếu mAP > best → save best.pt. Luôn save last.pt
    4. Sau khi hết epoch (hoặc early stop):
         - Save confusion matrix, PR curve, loss curves
         - Return kết quả cuối

Output nằm ở runs/detect/<name>/
    ├── weights/
    │   ├── best.pt   ← dùng cái này cho inference
    │   └── last.pt
    ├── results.csv   ← loss + mAP mỗi epoch
    ├── confusion_matrix.png
    ├── results.png   ← biểu đồ loss/mAP (ultralytics auto plot)
    └── train_batch0.jpg ... (sample batch với label overlay)

Chạy: python tutorial/4_train.py
"""

from pathlib import Path
from ultralytics import YOLO  # thư viện wrapper YOLOv8


# ─── Config — chỉnh ở đây, không cần sửa code bên dưới ────────────────────────

# Model nền (pretrained COCO). Ultralytics sẽ tự download nếu chưa có.
# Options:
#   yolov8n.pt — nano (3.2M params, nhanh nhất, mAP COCO thấp)
#   yolov8s.pt — small (11.2M params, cân bằng) ← dùng cái này
#   yolov8m.pt — medium (25.9M, chậm hơn ~2x, mAP cao hơn ~5-10%)
#   yolov8l.pt — large (43.7M)
#   yolov8x.pt — extra-large (68.2M)
MODEL = "yolov8s.pt"

# File data.yaml từ bước 3
DATA_YAML = "tutorial/3_data.yaml"

# Số epoch — 1 epoch = model đi hết tất cả ảnh train 1 lần.
# Với dataset nhỏ (~3000 ảnh), 40 epoch thường đủ. Dataset lớn (>50k ảnh) chỉ cần 10-20.
# Nếu dừng sớm quá → model chưa học hết (underfit). Dừng muộn quá → có thể overfit.
EPOCHS = 40

# Batch size — số ảnh model xử lý cùng lúc.
# Lớn hơn → tận dụng GPU tốt, loss gradient ít noise. Nhưng tốn VRAM.
# Rule-of-thumb với yolov8s:
#   imgsz=640  → batch 32 trên 12GB VRAM
#   imgsz=960  → batch 16 trên 12GB
#   imgsz=1280 → batch 8 trên 12GB
# Nếu OOM (out of memory) → giảm batch xuống còn 1/2.
BATCH = 16

# Image size — ảnh được resize về imgsz×imgsz trước khi vào model.
# Lớn hơn → bắt được object nhỏ tốt hơn (xe máy xa camera) nhưng chậm 2-4x.
# Phải chia hết cho 32 (do model stride). Các giá trị hay dùng: 416, 640, 768, 960, 1280.
IMGSZ = 960

# Learning rate ban đầu. Ultralytics default 0.01, nhưng với fine-tune từ pretrained
# nên hạ xuống 0.001-0.005 để không "quên" feature đã học (catastrophic forgetting).
# Nếu loss không giảm → thử tăng. Nếu loss nhảy loạn → giảm.
LR0 = 0.005

# Learning rate scheduler: lrf = lr_final / lr_initial.
# Mặc định cosine schedule: lr giảm dần từ lr0 → lr0*lrf theo đường cong cosine.
# lrf = 0.01 nghĩa là epoch cuối lr = lr0 * 0.01 = 0.00005.
LRF = 0.01

# Patience: nếu mAP val không cải thiện trong N epoch liên tiếp → stop sớm (early stopping).
# Tránh train quá lâu khi model đã hội tụ.
PATIENCE = 10

# Device: 0 = GPU đầu tiên, "cpu" = CPU, [0, 1] = multi-GPU.
# Ultralytics tự detect nếu để trống, nhưng nên chỉ rõ để tránh nhầm.
DEVICE = 0

# Tên folder output trong runs/detect/. Nếu đã tồn tại, ultralytics tự thêm số suffix.
RUN_NAME = "tutorial_train_v8s"

# Project folder — chứa tất cả runs. Mặc định runs/detect/.
PROJECT = "runs/detect"


# ─── Advanced config — không chỉnh nếu chưa hiểu ──────────────────────────────

# Workers: số process load data song song. 8 là ổn cho SSD, giảm xuống 4 nếu HDD.
WORKERS = 8

# AMP (Automatic Mixed Precision): dùng float16 cho forward → nhanh 1.5x trên
# GPU Ada/Ampere/Turing. Bật mặc định. Tắt nếu GPU cũ (< GTX 2070) hoặc gặp NaN loss.
AMP = True

# Augmentation — ultralytics có 10+ augment, dưới đây là 3 cái quan trọng.
# Mặc định đều bật nên không cần set, trừ khi bạn muốn tune.
# mosaic: ghép 4 ảnh thành 1 → tăng diversity, model học "nhìn nhiều scene cùng lúc"
# mixup:  pha trộn 2 ảnh → regularization, dùng cho dataset nhỏ
# hsv_h/s/v: thay màu/độ sáng → model robust với camera khác nhau

# Random seed để kết quả reproducible. Nhưng ultralytics không hoàn toàn
# deterministic ngay cả với seed cố định (do CUDA ops). Chạy 2 lần có thể lệch 0.5-1% mAP.
SEED = 42


def main():
    print("=" * 70)
    print(f"TRAIN YOLOv8s cho VN Traffic Detection")
    print("=" * 70)
    print(f"  Pretrained: {MODEL}")
    print(f"  Data:       {DATA_YAML}")
    print(f"  Epochs:     {EPOCHS}")
    print(f"  Batch:      {BATCH}")
    print(f"  Image size: {IMGSZ}")
    print(f"  LR:         {LR0} → {LR0 * LRF} (cosine)")
    print(f"  Device:     {DEVICE}")
    print(f"  Output:     {PROJECT}/{RUN_NAME}/")
    print("=" * 70)

    # Bước 1: Load model với pretrained weights
    # Nếu MODEL là file.pt → load weights.
    # Nếu MODEL là file.yaml (vd "yolov8s.yaml") → tạo model từ scratch (không có pretrain).
    # Luôn dùng .pt để fine-tune — train scratch cần dataset rất lớn (COCO 118k ảnh).
    print("\n[1] Load pretrained model...")
    model = YOLO(MODEL)
    # In info model để biết có bao nhiêu params
    model.info(verbose=False)

    # Bước 2: Train
    # Hàm model.train() là wrapper lớn, nó làm toàn bộ:
    #   - Reset head sang nc=4 (lấy từ data.yaml)
    #   - Setup optimizer (SGD mặc định, có thể đổi sang AdamW)
    #   - Setup data loader với augmentation
    #   - Loop EPOCHS lần, mỗi epoch val + log
    #   - Save best.pt / last.pt
    #   - Vẽ results.png, confusion_matrix.png
    print("\n[2] Bắt đầu train — có thể mất 1-6 giờ, cứ để chạy...")
    print("    (Progress bar hiển thị bên dưới)")
    print()

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        lr0=LR0,
        lrf=LRF,
        patience=PATIENCE,
        device=DEVICE,
        workers=WORKERS,
        amp=AMP,
        seed=SEED,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=False,   # True = overwrite nếu folder tồn tại. False = thêm suffix "2", "3"...
        verbose=True,     # In log chi tiết mỗi epoch
        plots=True,       # Vẽ sẵn results.png, confusion_matrix.png khi xong
        save=True,        # Save weights (best.pt + last.pt)
        val=True,         # Chạy validation mỗi epoch
    )

    # Bước 3: In kết quả cuối
    # results là object DetMetrics — chứa mAP, precision, recall, loss cuối cùng
    print("\n" + "=" * 70)
    print("✓ TRAIN XONG")
    print("=" * 70)

    # Tìm folder output thực sự (có thể có suffix nếu run cũ tồn tại)
    out_dir = Path(PROJECT) / RUN_NAME
    best_pt = out_dir / "weights" / "best.pt"

    print(f"\nKết quả:")
    print(f"  Best weights: {best_pt}")
    print(f"  Log CSV:      {out_dir / 'results.csv'}")
    print(f"  Biểu đồ:      {out_dir / 'results.png'}")
    print(f"  Confusion:    {out_dir / 'confusion_matrix.png'}")

    # Metrics cuối cùng (từ validation set)
    if hasattr(results, 'box'):
        print(f"\nMetrics trên val set (epoch cuối):")
        print(f"  mAP@0.5:      {results.box.map50:.3f}")
        print(f"  mAP@0.5:0.95: {results.box.map:.3f}")
        print(f"  Precision:    {results.box.mp:.3f}")
        print(f"  Recall:       {results.box.mr:.3f}")

    print(f"\n→ Chạy tiếp bước 5 để eval chi tiết trên test set:")
    print(f"    python tutorial/5_evaluate.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
