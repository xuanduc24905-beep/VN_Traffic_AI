"""
Bước 1: Kiểm tra môi trường trước khi train.

Script này check 5 thứ bắt buộc có trước khi train:
    1. Python version >= 3.10
    2. PyTorch cài đặt + đúng version
    3. CUDA (GPU) hoạt động hay không
    4. Ultralytics (YOLOv8 wrapper) cài đặt
    5. OpenCV đọc được ảnh

Nếu thiếu gì script sẽ in rõ, dừng lại — không chạy tiếp.

Chạy: python tutorial/1_check_env.py
"""

import sys          # module chuẩn, lấy thông tin Python
import platform     # lấy thông tin OS


def check_python():
    """Check Python version. YOLOv8 yêu cầu Python >= 3.8, khuyến nghị 3.10 hoặc 3.11."""
    v = sys.version_info  # tuple (major, minor, micro, ...)
    print(f"[1/5] Python version: {v.major}.{v.minor}.{v.micro}")
    print(f"       OS: {platform.platform()}")

    # Python < 3.8 không chạy được một số feature của ultralytics
    if v < (3, 8):
        print("       ❌ Python quá cũ, cần >= 3.8 (khuyến nghị 3.10/3.11)")
        sys.exit(1)
    print("       ✓ OK")


def check_torch():
    """Check PyTorch — framework deep learning mà YOLOv8 dùng bên dưới."""
    try:
        import torch
    except ImportError:
        print("[2/5] ❌ Chưa cài PyTorch. Chạy: pip install torch")
        sys.exit(1)

    print(f"[2/5] PyTorch version: {torch.__version__}")
    print("       ✓ OK")
    return torch


def check_cuda(torch):
    """
    Check CUDA — thư viện NVIDIA giúp PyTorch dùng GPU.

    Không có GPU vẫn train được (dùng CPU) nhưng chậm ~30 lần. Với dataset này
    (2800 ảnh × 40 epoch), CPU mất tầm 2-3 ngày; GPU 12GB mất 1-2 giờ.
    """
    if not torch.cuda.is_available():
        print("[3/5] ⚠ CUDA KHÔNG khả dụng — sẽ train trên CPU (rất chậm)")
        print("       Nếu máy có NVIDIA GPU, check:")
        print("         1. nvidia-smi — xem driver có hoạt động không")
        print("         2. torch có cài đúng CUDA version chưa")
        print("            (pip install torch --index-url https://download.pytorch.org/whl/cu118)")
        return False

    # Có GPU — in thông tin
    n_gpu = torch.cuda.device_count()  # số GPU trên máy
    print(f"[3/5] CUDA available: True")
    print(f"       Số GPU: {n_gpu}")

    for i in range(n_gpu):
        name = torch.cuda.get_device_name(i)                      # tên GPU
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3  # VRAM theo GB
        print(f"       GPU {i}: {name} ({total_mem:.1f} GB VRAM)")

    # Kiểm tra VRAM có đủ không — YOLOv8s với imgsz=960, batch=16 cần ~6GB
    main_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if main_vram < 6:
        print(f"       ⚠ VRAM chỉ {main_vram:.1f}GB — có thể cần giảm batch/imgsz trong bước 4")

    print("       ✓ OK")
    return True


def check_ultralytics():
    """
    Check Ultralytics — thư viện wrapper YOLOv8 (train/eval/inference bằng 1 dòng).
    Nếu không có, cài: pip install ultralytics
    """
    try:
        import ultralytics
    except ImportError:
        print("[4/5] ❌ Chưa cài ultralytics. Chạy: pip install ultralytics")
        sys.exit(1)

    print(f"[4/5] Ultralytics version: {ultralytics.__version__}")
    print("       ✓ OK")


def check_opencv():
    """Check OpenCV — thư viện đọc/ghi ảnh và video. Streamlit + inference cần cv2."""
    try:
        import cv2
    except ImportError:
        print("[5/5] ❌ Chưa cài OpenCV. Chạy: pip install opencv-python")
        sys.exit(1)

    print(f"[5/5] OpenCV version: {cv2.__version__}")
    print("       ✓ OK")


def main():
    print("=" * 60)
    print("KIỂM TRA MÔI TRƯỜNG TRƯỚC KHI TRAIN")
    print("=" * 60)

    check_python()
    torch = check_torch()
    has_gpu = check_cuda(torch)
    check_ultralytics()
    check_opencv()

    print("=" * 60)
    if has_gpu:
        print("✓ MÔI TRƯỜNG SẴN SÀNG — chạy tiếp bước 2:")
    else:
        print("⚠ MÔI TRƯỜNG OK nhưng không có GPU — train sẽ rất chậm")
    print("    python tutorial/2_prepare_data.py")
    print("=" * 60)


if __name__ == "__main__":
    # Dòng này nghĩa là: chỉ chạy main() khi file được chạy trực tiếp (python 1_check_env.py)
    # Không chạy khi file bị import từ file khác.
    main()
