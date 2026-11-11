# Hướng dẫn chạy 4 pipeline cải tiến V5+

Script nằm ở `scripts/pipelines/`. Mọi lệnh chạy từ thư mục gốc dự án. **Các script này chưa được chạy với mô hình và dữ liệu thật**; đã kiểm thử bằng dữ liệu và mô hình giả (`python -m pytest tests/test_pipelines.py -q`, 20 bài thử đạt). Chạy thử nhanh (`--dry-run`, `--fraction 0.05 --epochs 2`) trước khi chạy thật.

| File | Việc |
|---|---|
| `p1_pseudo_small.py` | Sinh nhãn giả từ giáo viên; ba chế độ xử lý hộp nhỏ: `drop`, `keep`, `mask`; nhiều `--teacher` thì ghép WBF |
| `p2_build_datasets.py` | Dựng dataset: tập gốc + nhãn tay và/hoặc nhãn giả, **cùng ngân sách**; không chép lại tập gốc |
| `p4_tiles.py` | Cắt ảnh lớn thành ô để huấn luyện (val và test giữ nguyên ảnh đầy đủ) |
| `train.py` | Huấn luyện theo công thức: `v5plus`, `finetune`, `imgsz1280`, `multiscale`, `clspw`, `scale`, `p2`, `progressive`; hỗ trợ `--distill-model` |
| `common.py` | Hàm dùng chung (không chạy trực tiếp) |

Đánh giá mô hình mới: `scripts/eval/no_retrain_experiments.py --model-weights tên=best.pt` (xem `docs/guides/HUONG_DAN_NANG_DIEM_KO_CAN_TRAIN.md`).

## Nguyên tắc chung

- Chọn bằng **val**; chỉ đánh giá test một lần, sau khi chốt.
- Loại camera của tập test khỏi dữ liệu nhãn giả: `--exclude-cameras` (danh sách mã hoặc file mỗi dòng một mã). Mã camera nằm trong tên file ảnh `hcm_<phường>_<mã 6 số>_<ngày>_<giờ>.jpg`. Dùng thêm `--blacklist configs/hcm_cameras_blacklist.yaml` để bỏ các camera V5 dự đoán kém.
- Mỗi nhóm thí nghiệm chỉ đổi **một yếu tố** so với nhóm liền trước; dùng cùng `--seed` và cùng `--recipe`.
- Ảnh không nhãn lấy từ `data/raw/frames/hcm` (script thu snapshot nằm ở tag `v6.0-pseudo`: `git checkout v6.0-pseudo -- scripts/pseudo`).

## Pipeline 1: self-training giữ vật nhỏ (nhóm A đến E)

Câu hỏi: cách lọc nhãn giả theo kích thước hộp ảnh hưởng thế nào đến recall xe máy nhỏ?

| Nhóm | Giáo viên | Chế độ | Lệnh `--teacher` / tham số |
|---|---|---|---|
| A | V5 imgsz 640 | `drop` | `--teacher runs/detect/train_v8s_v5_honest/weights/best.pt --imgsz 640` |
| B | V5+ imgsz 960 | `drop` | `--teacher runs/detect/train_v8s_v5plus_960/weights/best.pt` |
| C | V5+ imgsz 960 | `keep` | như B |
| D | V5+ imgsz 960 | `mask` | như B |
| E | V5+ và V6 (WBF) | `mask` | hai lần `--teacher` |

```bash
# ví dụ nhóm D
python scripts/pipelines/p1_pseudo_small.py --images data/raw/frames/hcm \
    --teacher runs/detect/train_v8s_v5plus_960/weights/best.pt --mode mask \
    --blacklist configs/hcm_cameras_blacklist.yaml --exclude-cameras configs/test_cameras.txt \
    --out data/pseudo_D --device 0
python scripts/pipelines/p2_build_datasets.py --base data/merged_v5 --pseudo data/pseudo_D \
    --arms pseudo -n 100000 --out data/exp1_D
python scripts/pipelines/train.py --data data/exp1_D/pseudo.yaml --name p1_D --recipe v5plus --device 0
```

Ba chế độ (xem đầu `p1_pseudo_small.py`):
- `drop`: như V6. Hộp có diện tích nhỏ hơn `--small-frac` (0,25% ảnh) bị bỏ, nên xe máy xa bị học thành nền.
- `keep`: không lọc theo kích thước; hộp nhỏ chỉ cần đạt ngưỡng conf (hoặc `--small-thr`).
- `mask`: như `keep`, và hộp nghi ngờ (conf từ `--low-thr` đến dưới ngưỡng, nhỏ hơn `--mask-max-frac`) bị **tô xám** trong ảnh lưu ra để không bị học thành nền.

Ngưỡng theo lớp mặc định: xe máy 0,60, ô tô 0,45, bus 0,50, truck 0,50 (như round 1). Đổi bằng `--thr motorcycle=0.7,car=0.5,...`. Với nhiều giáo viên (nhóm E), độ tin cậy sau ghép WBF của hộp chỉ một mô hình thấy bị giảm một nửa, nên ngưỡng áp lên giá trị đã ghép.

Sau khi sinh nhãn: xem `manifest.json` (số hộp theo lớp, số hộp nhỏ được giữ, số hộp bị tô), **kiểm tay 300 đến 500 ảnh** để đo độ chính xác nhãn (strict dưới 70% thì làm lại bước lọc).

## Pipeline 2: nhãn tay và nhãn giả, cùng ngân sách

Câu hỏi: với cùng số ảnh, nhãn tay hơn nhãn giả bao nhiêu?

```bash
python scripts/pipelines/p2_build_datasets.py --base data/merged_v5 --out data/exp2 \
    --hand data/raw_vietnam_vehicles_v10 --remap-v10 --pseudo data/pseudo_D \
    --arms base,hand,pseudo --budget images -n 3000
for arm in base hand pseudo; do
  python scripts/pipelines/train.py --data data/exp2/$arm.yaml --name p2_$arm --recipe v5plus --device 0
done
```

- `--remap-v10` đổi bộ Roboflow 6 lớp về 4 lớp (container và xe tải thành truck, ô tô thành car, xe khách thành bus, xe máy thành motorcycle). **Ảnh có xe ba gác bị loại** (mặc định) để ba gác không bị học thành nền; giữ lại bằng `--keep-images-with-dropped`.
- `--budget boxes -n <số hộp>` cho ngân sách cùng số hộp, vì cùng số ảnh có thể khác số hộp. Thống kê in ra để so (hộp mỗi ảnh, tỉ lệ hộp nhỏ, hộp theo lớp).
- Dataset yaml liệt kê thư mục train của tập gốc và của phần thêm; **không chép lại** tập gốc. val và test luôn là của tập gốc.

## Pipeline 3: giáo viên ensemble và chưng cất

Hai cách, có thể kết hợp:

1. **Nhãn giáo viên ensemble** (nhóm E ở pipeline 1): hai lần `--teacher` (V5+ và V6).
2. **Chưng cất đặc trưng** của Ultralytics: giáo viên là một mô hình lớn hơn cùng họ YOLOv8, học sinh nhỏ.

```bash
# giáo viên lớn (cần so với V5+ trên val trước khi dùng)
python scripts/pipelines/train.py --data data/data_v5.yaml --name teacher_m --recipe v5plus \
    --init yolov8m.pt --device 0
# học sinh có chưng cất, và đối chứng cùng học sinh không chưng cất
python scripts/pipelines/train.py --data data/exp1_D/pseudo.yaml --name student_distill --recipe v5plus \
    --distill-model runs/detect/teacher_m/weights/best.pt --dis 6.0 --device 0
python scripts/pipelines/train.py --data data/exp1_D/pseudo.yaml --name student_plain --recipe v5plus --device 0
```

Điều kiện: giáo viên và học sinh **cùng họ** YOLO; giáo viên phải tốt hơn học sinh trên val (đo trước); cần bản Ultralytics có tham số `distill_model`. Cặp `student_distill` và `student_plain` là đối chứng bắt buộc, chạy 2 đến 3 seed.

## Pipeline 4: công thức huấn luyện cho vật nhỏ

```bash
# dữ liệu V5 giữ nguyên, đổi công thức (mỗi lần một thứ)
for r in imgsz1280 multiscale clspw scale p2 progressive; do
  python scripts/pipelines/train.py --data data/data_v5.yaml --name p4_$r --recipe $r --device 0
done
# huấn luyện trên ô cắt
python scripts/pipelines/p4_tiles.py --src data/merged_v5 --out data/merged_v5_tiles
python scripts/pipelines/train.py --data data/merged_v5_tiles/tiles.yaml --name p4_tiles --recipe v5plus --device 0
```

`progressive`: giai đoạn 1 ở imgsz 960 (`--epochs`, mặc định 40), giai đoạn 2 ở imgsz 1280 (`--epochs2`, mặc định 15) bắt đầu từ `best.pt` của giai đoạn 1. `clspw` và `multiscale` cần bản Ultralytics có các tham số này (`train.py` báo lỗi rõ nếu thiếu).

## Đánh giá và chạy tiếp

```bash
python scripts/eval/no_retrain_experiments.py --device 0 \
    --model-weights p1_B=runs/detect/p1_B/weights/best.pt p1_D=runs/detect/p1_D/weights/best.pt
```

Bảng cho `mAP50`, `AP50_moto_small` và khoảng tin cậy so với `base960` (V5+). `train.py` cũng tự ghi kết quả val vào `runs/detect/pipelines_summary.csv`. Chạy tiếp khi Colab ngắt: `train.py ... --name <tên> --resume`.

## Phụ: `scripts/training/v8_experiments.py` (CHƯA CHẠY HUẤN LUYỆN THẬT)

`train.py` dùng cấu hình nền và hàm đánh giá của file này. File còn chạy riêng được: 9 preset, mỗi preset đổi một nhóm tham số so với V5+.

**Trạng thái:** đã kiểm tra tên tham số của cả 9 preset hợp lệ với Ultralytics và kiến trúc P2 dựng được. **Chưa chạy huấn luyện lần nào.** Bước đầu luôn là chạy thử nhanh:

```bash
python scripts/training/v8_experiments.py --list
python scripts/training/v8_experiments.py --preset baseline --fraction 0.1 --epochs 3 --device 0
python scripts/training/v8_experiments.py --preset aug_small --epochs 40 --device 0   # tự đánh giá val
python scripts/training/v8_experiments.py --preset aug_small --final                  # đánh giá TEST, chỉ một lần, sau khi chốt
```

Preset: `baseline` (tái lập V5+), `freeze10`, `aug_small`, `aug_light`, `aug_occl`, `loss_cls`, `imgsz1280`, `p2_head`, `combo`. Kết quả ở `runs/detect/exp_<preset>/`, tổng hợp val ở `runs/detect/exp_summary.csv`.

So với `scripts/training/v5_plus_960.py`: cùng công thức, nhưng thêm preset, tự đánh giá val và tự sửa đường dẫn dữ liệu theo máy; **không có** chạy tiếp (resume) và `cache="disk"`. `baseline` giữ `copy_paste=0,3`; tham số này **không có tác dụng** với nhãn hộp (Ultralytics chỉ chạy copy-paste khi nhãn có polygon), các preset khác đặt 0. Với `train.py` thì dùng `--resume` để chạy tiếp.

## Hạn chế cần nhớ

- Chưa có tập test mới gán tay; không có nó thì không khẳng định được "hơn V5+", đặc biệt với bóng râm và che khuất.
- Công thức `finetune` (lr0 0,0005, 20 epoch) và các giá trị mặc định khác là **điểm khởi đầu chưa kiểm chứng**.
- Chế độ `mask` tạo vùng xám trên ảnh huấn luyện; tác dụng thật của nó là giả thuyết cần đo (nhóm D so với nhóm C).
- Mức chênh dự kiến chỉ khoảng 0,01 đến 0,02; cần khoảng tin cậy và nhiều seed.
