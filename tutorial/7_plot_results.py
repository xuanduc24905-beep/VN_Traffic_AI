"""
Bước 7: Vẽ biểu đồ phân tích kết quả training.

Trong lúc train, Ultralytics tự ghi file `results.csv` ở folder output,
mỗi dòng = 1 epoch. Script này đọc CSV đó và vẽ:
    1. Loss curves (box_loss, cls_loss, dfl_loss)      — train vs val
    2. mAP curves (mAP@0.5, mAP@0.5:0.95)              — chỉ val
    3. Precision, Recall curves                         — chỉ val
    4. Learning rate schedule                           — xem lr có giảm đúng cosine không

Cách đọc biểu đồ:
    - Loss train phải GIẢM dần. Nếu phẳng → learning rate quá thấp.
    - Loss val đi theo train, có thể "lệch lên" sau một lúc = BẮT ĐẦU OVERFIT.
      → Nên stop ở epoch gần điểm val loss thấp nhất (ultralytics đã làm với patience).
    - mAP val phải TĂNG dần. Plateau = model đã hội tụ, train thêm vô ích.
    - lr0 phải GIẢM theo đường cong cosine (từ 0.005 → 0.00005).

Ultralytics cũng tự vẽ `results.png` sau khi train — script này vẽ chi tiết
và custom hơn (từng loss riêng, dùng matplotlib).

Chạy: python tutorial/7_plot_results.py
"""

from pathlib import Path
import pandas as pd              # đọc CSV, dễ hơn csv module chuẩn
import matplotlib.pyplot as plt   # vẽ biểu đồ


# ─── Config ───────────────────────────────────────────────────────────────────

# Đường dẫn folder training output. Sửa nếu RUN_NAME ở bước 4 khác.
RUN_DIR = Path("runs/detect/tutorial_train_v8s")

# Nơi lưu biểu đồ tổng hợp
OUT_DIR = Path("results/tutorial_plots")


def load_results(run_dir: Path) -> pd.DataFrame:
    """
    Đọc file results.csv do ultralytics ghi. Format:
        epoch, train/box_loss, train/cls_loss, train/dfl_loss,
               metrics/precision(B), metrics/recall(B),
               metrics/mAP50(B), metrics/mAP50-95(B),
               val/box_loss, val/cls_loss, val/dfl_loss,
               lr/pg0, lr/pg1, lr/pg2
    """
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {csv_path}. "
            f"Chạy bước 4 trước, hoặc sửa RUN_DIR cho đúng."
        )

    df = pd.read_csv(csv_path)
    # Column name có khoảng trắng thừa (bug của ultralytics ở một số version)
    df.columns = df.columns.str.strip()
    print(f"Load {len(df)} epoch từ {csv_path}")
    return df


def plot_losses(df: pd.DataFrame, out_path: Path):
    """
    Vẽ 3 loss (box, cls, dfl) × 2 (train, val) = 6 line trên 3 subplot.

    Giải thích 3 loss của YOLOv8:
        box_loss: lệch bbox dự đoán vs ground truth (CIoU loss)
        cls_loss: sai class (focal loss với label smoothing)
        dfl_loss: Distribution Focal Loss — regularize phân phối bbox regression
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, loss_name in zip(axes, ["box_loss", "cls_loss", "dfl_loss"]):
        # Train loss (bắt buộc có)
        ax.plot(df["epoch"], df[f"train/{loss_name}"],
                label=f"train {loss_name}", color="C0", linewidth=2)
        # Val loss (có từ ultralytics v8.0.x, một số version cũ không có)
        val_col = f"val/{loss_name}"
        if val_col in df.columns:
            ax.plot(df["epoch"], df[val_col],
                    label=f"val {loss_name}", color="C1", linewidth=2)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(loss_name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Training Loss Curves", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Loss curves: {out_path}")


def plot_metrics(df: pd.DataFrame, out_path: Path):
    """
    Vẽ mAP + Precision + Recall qua các epoch.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Subplot 1: mAP@0.5 và mAP@0.5:0.95
    ax = axes[0]
    ax.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.5", linewidth=2)
    ax.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_title("mAP qua các epoch (val set)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Đánh dấu epoch có mAP@0.5 cao nhất (= best.pt)
    best_idx = df["metrics/mAP50(B)"].idxmax()
    best_epoch = df["epoch"].iloc[best_idx]
    best_map = df["metrics/mAP50(B)"].iloc[best_idx]
    ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.5)
    ax.annotate(f"Best mAP@0.5\n= {best_map:.3f}\n@ epoch {int(best_epoch)}",
                xy=(best_epoch, best_map), xytext=(best_epoch + 2, best_map - 0.1),
                fontsize=9, color="red")

    # Subplot 2: Precision & Recall
    ax = axes[1]
    ax.plot(df["epoch"], df["metrics/precision(B)"], label="Precision", linewidth=2)
    ax.plot(df["epoch"], df["metrics/recall(B)"], label="Recall", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Precision & Recall (val set)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Metrics curves: {out_path}")


def plot_learning_rate(df: pd.DataFrame, out_path: Path):
    """
    Vẽ learning rate schedule — xem có giảm đúng cosine không.

    YOLOv8 có 3 param group:
        pg0: weights (nn.Parameter) — lr chính
        pg1: biases
        pg2: BN gamma (BatchNorm)
    Thường cả 3 chung schedule. pg0 là cái ta set qua lr0.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    for pg in ["lr/pg0", "lr/pg1", "lr/pg2"]:
        if pg in df.columns:
            ax.plot(df["epoch"], df[pg], label=pg, linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning rate schedule (cosine decay)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Dùng log scale vì lr thay đổi 100-1000x
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ LR schedule: {out_path}")


def print_summary(df: pd.DataFrame):
    """In tóm tắt kết quả training lên terminal."""
    print("\n" + "=" * 60)
    print("TÓM TẮT TRAINING")
    print("=" * 60)

    print(f"  Số epoch chạy: {len(df)}")
    print(f"  Epoch cuối:    {int(df['epoch'].iloc[-1])}")

    best_idx = df["metrics/mAP50(B)"].idxmax()
    print(f"\n  Best val mAP@0.5:      {df['metrics/mAP50(B)'].iloc[best_idx]:.3f}"
          f" (epoch {int(df['epoch'].iloc[best_idx])})")
    print(f"  Best val mAP@0.5:0.95: {df['metrics/mAP50-95(B)'].max():.3f}")
    print(f"  Final val precision:   {df['metrics/precision(B)'].iloc[-1]:.3f}")
    print(f"  Final val recall:      {df['metrics/recall(B)'].iloc[-1]:.3f}")

    print(f"\n  Train loss cuối:")
    print(f"    box: {df['train/box_loss'].iloc[-1]:.3f}")
    print(f"    cls: {df['train/cls_loss'].iloc[-1]:.3f}")
    print(f"    dfl: {df['train/dfl_loss'].iloc[-1]:.3f}")

    # So sánh loss đầu vs cuối để biết model có học không
    first_box = df['train/box_loss'].iloc[0]
    last_box = df['train/box_loss'].iloc[-1]
    print(f"\n  Box loss: {first_box:.3f} → {last_box:.3f} ({last_box - first_box:+.3f})")
    if last_box > first_box * 0.9:
        print(f"    ⚠ Loss gần như không giảm → model học kém. Check lr, data, hoặc overfit.")
    else:
        print(f"    ✓ Loss giảm đáng kể → model đã học")


def main():
    print("=" * 60)
    print("PLOT TRAINING RESULTS")
    print("=" * 60)

    # Load CSV
    df = load_results(RUN_DIR)

    # Tạo thư mục output
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Vẽ 3 biểu đồ
    plot_losses(df, OUT_DIR / "losses.png")
    plot_metrics(df, OUT_DIR / "metrics.png")
    plot_learning_rate(df, OUT_DIR / "lr_schedule.png")

    # In tóm tắt
    print_summary(df)

    print("\n" + "=" * 60)
    print(f"✓ HOÀN TẤT. Xem biểu đồ ở: {OUT_DIR}/")
    print()
    print("Các file khác đã có sẵn trong training folder:")
    print(f"  {RUN_DIR}/results.png            — biểu đồ tổng do ultralytics vẽ")
    print(f"  {RUN_DIR}/confusion_matrix.png   — ma trận nhầm lẫn class")
    print(f"  {RUN_DIR}/PR_curve.png           — Precision-Recall curve")
    print(f"  {RUN_DIR}/F1_curve.png           — F1 vs confidence")
    print(f"  {RUN_DIR}/train_batch0.jpg       — sample batch với label")
    print(f"  {RUN_DIR}/val_batch0_pred.jpg    — prediction trên val batch")
    print("=" * 60)


if __name__ == "__main__":
    main()
