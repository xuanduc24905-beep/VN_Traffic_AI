# Hướng dẫn: nâng điểm V5+ không cần huấn luyện lại

Nhánh này chỉ có ba file; mọi lệnh chạy từ **thư mục gốc dự án**.

| File | Việc | Cần GPU? |
|---|---|---|
| `scripts/eval/no_retrain_experiments.py` | Thử 7 cách tăng độ chính xác **không huấn luyện lại**; tính AP xe máy nhỏ và khoảng tin cậy | Nên có (CPU khoảng 0,2 giây/ảnh) |
| `src/evaluation/det_analysis.py` | Thư viện phân tích lỗi: kích thước, nhầm lớp, quét ngưỡng conf | Không |
| `notebooks/no_retrain_gpu_colab.ipynb` | Chạy `no_retrain_experiments.py` trên Colab | T4 |

## 0. Chuẩn bị

- Python có `ultralytics`, `pandas`, `opencv-python`, `pyyaml`, `matplotlib`.
- Trọng số: `runs/detect/train_v8s_v5plus_960/weights/best.pt` (GitHub Releases `v1.0-weights`) và `runs/detect/train_v8s_v6_pseudo/weights/best.pt` (Releases `v6.0-pseudo`).
- Dữ liệu val: `data/merged_v6/images/val` và `labels/val` (1.466 ảnh), lấy từ `dataset_v6_merged.tar.gz` của Releases `v6.0-pseudo`:
  ```bash
  tar -xzf dataset_v6_merged.tar.gz -C data/ merged_v6/images/val merged_v6/labels/val
  ```
- Nhãn val phải **có nội dung** (không phải file rỗng).

## 1. `no_retrain_experiments.py`

Bảy cách, đều dùng trọng số V5+ (riêng `wbf_v5p_v6` ghép thêm V6):

| Cách | Ý nghĩa |
|---|---|
| `base960` | Suy luận chuẩn imgsz 960 (mốc so sánh) |
| `sz1152`, `sz1280` | Tăng imgsz khi suy luận |
| `tta960` | Test-Time Augmentation của Ultralytics |
| `clahe960` | Tăng tương phản CLAHE trước khi suy luận |
| `slice960` | Cắt ảnh thành ô 640 chồng lấn (kiểu SAHI) |
| `wbf_v5p_v6` | Ghép V5+ và V6 bằng Weighted Boxes Fusion |
| (kèm `base960`) | Dòng `base960+per_class_conf`: ngưỡng conf riêng từng lớp, tính hậu kỳ |

```bash
# tất cả 7 cách, trên GPU
python scripts/eval/no_retrain_experiments.py --device 0
# chỉ vài cách; chạy thử 200 ảnh
python scripts/eval/no_retrain_experiments.py --methods base960 wbf_v5p_v6 --device 0 --max-images 200
# so thêm mô hình khác (suy luận chuẩn imgsz 960); mặc định chạy base960 và các mô hình đó
python scripts/eval/no_retrain_experiments.py --device 0 \
    --model-weights v6b=runs/detect/p1_D/weights/best.pt v6c=runs/detect/p1_C/weights/best.pt
# chỉ tính lại chỉ số từ dự đoán đã lưu (không chạy mô hình)
python scripts/eval/no_retrain_experiments.py --report-only
```

Tham số khác: `--data` (mặc định `data/merged_v6`), `--split val`, `--out` (mặc định `results/no_retrain`), `--bootstrap 300` (số lần lấy mẫu lại ảnh; 0 để tắt).

**Kết quả** ở `--out`: `summary.csv`, `summary.md`, và `cache/<cách>.pkl` (dự đoán đã lưu: chạy lại thì cách nào đã có `.pkl` được bỏ qua; muốn chạy lại thì xóa file đó).

**Cách đọc bảng:**
- `mAP50`, `mAP50-95`: kiểu COCO 101 điểm, **mỗi hộp một nhãn lớp**. Thấp hơn `model.val` của Ultralytics khoảng 0,02–0,03 vì `model.val` dùng NMS đa nhãn. Đừng đặt chung hai cách đo vào một bảng.
- `AP50_moto_small`, `AP50_moto_medium`: AP của riêng xe máy nhỏ (dưới 32×32 px) và cỡ vừa, **không phụ thuộc ngưỡng conf**, nên so sánh được giữa các cách. Cột `R_moto_small` phụ thuộc ngưỡng mỗi cách tự chọn, không so trực tiếp.
- `conf`, `P`, `R`, `F1`: ngưỡng chọn trên một nửa ảnh (tune), đo trên nửa còn lại (eval).
- `ΔmAP50_CI95`, `P(Δ>0)`: khoảng tin cậy 95% của mức chênh mAP50 so với `base960` (bootstrap theo ảnh) và tỉ lệ lần lấy mẫu mà cách đó tốt hơn. Khoảng không chứa 0 thì mức chênh đáng tin.
- `s/img`: thời gian mỗi ảnh (gồm cả đọc ảnh). Lấy số đo trên GPU cho báo cáo.

Hai hằng số `W_V5P`, `W_V6` ở đầu file gắn cứng đường dẫn trọng số V5+ và V6. Đánh giá mô hình khác thì dùng `--model-weights`, không cần sửa file.

## 2. `det_analysis.py`

Thư viện (chỉ cần numpy, pandas); các hàm chính:

- `collect_predictions(model, images_dir, labels_dir, conf_floor=0.05, imgsz=960, max_images=None)`: chạy model một lần ở conf thấp và lưu lại. `max_images` lấy **mẫu ngẫu nhiên có seed**, không phải N file đầu theo tên.
- `evaluate_records(records, conf_thr)`: ghép dự đoán với nhãn (IoU 0,5) ở một ngưỡng; mỗi nhãn là `tp`, `confused` (nhầm lớp) hoặc `fn` (bỏ sót), cùng các dự đoán thừa.
- `recall_by_size(df_gt, names)`: recall theo lớp và kích thước COCO (nhỏ dưới 32², vừa dưới 96², lớn).
- `confusion(...)`, `sweep_conf(records, thresholds)`, `worst_images(records, conf)`.

## 3. Notebook trên Colab (`no_retrain_gpu_colab.ipynb`)

1. Runtime, Change runtime type, **T4 GPU**.
2. Chép vào Drive (cùng cấu trúc dự án): `scripts/eval/no_retrain_experiments.py`, `src/evaluation/det_analysis.py`, hai trọng số, và notebook.
3. Mở notebook, sửa ô cấu hình: `PROJECT_DIR` (đường dẫn dự án trên Drive), `DATA_DIR = "/content/merged_v6"`. Đặt `MAX_IMAGES = 200` cho lần chạy thử.
4. Run all. Notebook tự tải dataset, **chỉ giải nén phần val** ra ổ Colab (không ghi lên Drive).

Chống crash RAM: mỗi cách chạy trong một tiến trình riêng, dự đoán lưu lên Drive ngay sau mỗi cách. Colab bị ngắt thì chạy lại notebook: cách nào đã xong được bỏ qua.

## 4. Gỡ lỗi nhanh

| Lỗi | Cách xử lý |
|---|---|
| `ModuleNotFoundError: src` | Chưa đứng ở thư mục gốc dự án |
| `THIẾU` ở ô kiểm tra file của notebook | Chép file vào đúng đường dẫn tương ứng |
| Dừng ở "Không có ảnh" hoặc "nhãn rỗng" | `DATA_DIR` phải chứa `images/val` và `labels/val` có nội dung |
| Colab crash hết RAM | `MAX_IMAGES = 200`; chạy từng cách (`--methods`); khởi động lại runtime |

Kết quả đo được (V5+, val 1.466 ảnh, GPU T4): ghép V5+ và V6 bằng WBF tăng mAP50 từ 0,829 lên 0,847 (khoảng tin cậy của mức chênh [+0,012; +0,022]) và AP xe máy nhỏ từ 0,724 lên 0,745, nhưng chậm gần gấp đôi (49 so với 26 ms/ảnh). Tăng imgsz chỉ khi suy luận làm giảm (imgsz 1280: −0,039).
