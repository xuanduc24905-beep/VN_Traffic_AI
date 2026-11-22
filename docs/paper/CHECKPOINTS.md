# Checkpoint dùng trong bài báo SSRC (Nhóm 02)

Tải tại GitHub Release **`paper-ssrc-v2-weights`**: https://github.com/xuanduc24905-beep/VN_Traffic_AI/releases/tag/paper-ssrc-v2-weights

Các file `.pt` không nằm trong git (thư mục `runs/` và `weights/trained/` bị `.gitignore`, và Faster R-CNN nặng 158 MB vượt giới hạn 100 MB của git).

## Danh sách

mAP@0,5 trên V5+ test theo công thức Ultralytics (YOLO: `model.val()`; Faster R-CNN: cùng công thức áp lên dự đoán của TorchVision, xem `scripts/verify/88_ultralytics_metric_on_preds.py`).

| File trong Release | Run gốc trong repo | Vai trò trong bài | mAP@0,5 | MB | SHA-256 (12 ký tự đầu) |
|---|---|---|---|---|---|
| `yolov8s_v5plus_baseline.pt` | `runs/detect/exp_baseline/weights/best.pt` | Cơ sở V5+ (Bảng II–VI) | 0.7930 | 21.5 | `8e3d5f89a8f7` |
| `yolov8s_cw.pt` | `runs/detect/r_clspw_v2/weights/best.pt` | YOLOv8s-CW, mô hình cuối (mọi bảng, phân tích lỗi, tốc độ) | 0.7984 | 21.5 | `609c2c99fbf7` |
| `yolov8s_p4_scale.pt` | `runs/detect/p4_scale_from_baseline/weights/best.pt` | p4_scale (Bảng II, III, V, VI) | 0.7871 | 21.5 | `59cdbdfc181b` |
| `yolov8s_p4_imgsz1280.pt` | `runs/detect/p4_imgsz1280_from_baseline/weights/best.pt` | p4_imgsz1280 (Bảng II, III, V, VI) | 0.7741 | 21.6 | `a2103f32c804` |
| `yolov8s_p2.pt` | `runs/detect/r_p2/weights/best.pt` | r_p2 (Bảng II, III, V, VI) | 0.7528 | 21.0 | `80daf0bba560` |
| `yolov8s_progressive.pt` | `runs/detect/r_progressive/weights/best.pt` | r_progressive, giai đoạn 2 (Bảng II, III, V, VI) | 0.7710 | 21.6 | `7475fb1cbdb3` |
| `yolov8s_ctrl_cw05_s0.pt` | `runs/detect/ctrl_cw05_s0/weights/best.pt` | Đối chứng cls_pw = 0,5, hạt giống 0 | 0.8031 | 21.5 | `699ebdcaa7ed` |
| `yolov8s_ctrl_cw05_s1.pt` | `runs/detect/ctrl_cw05_s1/weights/best.pt` | Đối chứng cls_pw = 0,5, hạt giống 1 | 0.7973 | 21.5 | `f9d12b62b026` |
| `yolov8s_ctrl_cw05_s2.pt` | `runs/detect/ctrl_cw05_s2/weights/best.pt` | Đối chứng cls_pw = 0,5, hạt giống 2 | 0.8020 | 21.5 | `3959f12236d9` |
| `yolov8s_ctrl_cw0_s0.pt` | `runs/detect/ctrl_cw0_s0/weights/best.pt` | Đối chứng cls_pw = 0, hạt giống 0 | 0.8035 | 21.5 | `c717dd1986a7` |
| `yolov8s_ctrl_cw0_s1.pt` | `runs/detect/ctrl_cw0_s1/weights/best.pt` | Đối chứng cls_pw = 0, hạt giống 1 | 0.7956 | 21.5 | `d4721126a2be` |
| `yolov8s_ctrl_cw0_s2.pt` | `runs/detect/ctrl_cw0_s2/weights/best.pt` | Đối chứng cls_pw = 0, hạt giống 2 | 0.8019 | 21.5 | `ecf8fc3a4e81` |
| `yolov8s_v5_2833_no_pseudo.pt` | `runs/detect/train_v8s_v5plus_960/weights/best.pt` | Thí nghiệm nhãn giả: 2.833 ảnh thật | 0.7812 | 21.5 | `b1e9b1e583a3` |
| `yolov8s_v6_pseudo402.pt` | `runs/detect/train_v8s_v6_pseudo/weights/best.pt` | Thí nghiệm nhãn giả: 2.833 ảnh thật + 402 ảnh nhãn giả | 0.7608 | 21.5 | `d914dcbf2239` |
| `fasterrcnn_r50fpn_v5plus.pt` | `weights/trained/frcnn_v5plus/best.pt` | Faster R-CNN R50-FPN train trên V5+ (Bảng II–VI) | 0.7945 | 158.1 | `5ac9c09863f6` |

## Kiểm tra file sau khi tải

```bash
sha256sum *.pt
```

SHA-256 đầy đủ:

```
8e3d5f89a8f7411e557283b8355e03fa6203730fe84296a17d6cf864e776c92e  yolov8s_v5plus_baseline.pt
609c2c99fbf79aabe534b42dcbbf7dd41184bc16ef4aa46c189066ab6496a519  yolov8s_cw.pt
59cdbdfc181b053f5ea7c242f6d6b849c83077c5b872222b6394f7c78ec80ca4  yolov8s_p4_scale.pt
a2103f32c804067785ade292657622b4086ae7349031c23d2c21d5d6a703237d  yolov8s_p4_imgsz1280.pt
80daf0bba5605a887763e38fcf5813f3ce8cef4c70624f1f378e027a0909711c  yolov8s_p2.pt
7475fb1cbdb3de4666c374e4b6e58d816c0aeda60ae00c2ea5c9d3b75f125342  yolov8s_progressive.pt
699ebdcaa7ed4a8b3173dd3c8238232b4986fd92a1838d5b7e18916bd684468a  yolov8s_ctrl_cw05_s0.pt
f9d12b62b02605af737cc201bd34084043490470e9b83c6407fa83cd2e1ec654  yolov8s_ctrl_cw05_s1.pt
3959f12236d976af373ade4f4f0357c8d18fe186d1b30c84c49c17fcedf5827d  yolov8s_ctrl_cw05_s2.pt
c717dd1986a76dfaabc3c8ab86946e45cf5a8afe8aac17d67e3835961515578d  yolov8s_ctrl_cw0_s0.pt
d4721126a2beb0bb254c203ce8006247a820cc5f42f60608abc9c9dbe5241458  yolov8s_ctrl_cw0_s1.pt
ecf8fc3a4e815449d073a4ed34e3078950f4ac6147ffb40051d3cdaf42f93922  yolov8s_ctrl_cw0_s2.pt
b1e9b1e583a3fa0d0dfccdd502fd234e670a46a318998bfb6e81db7ecd6d0a0d  yolov8s_v5_2833_no_pseudo.pt
d914dcbf2239187c6b610a0d4c99161af4a8ede8d3aa8504503b29a414eeb1ca  yolov8s_v6_pseudo402.pt
5ac9c09863f6d86cd47f3ccbef0d0f5a34b3a45f73e28be6e98a435ebab7d99a  fasterrcnn_r50fpn_v5plus.pt
```

## Dùng lại

Đặt file về đúng đường dẫn ở cột "Run gốc" thì các script trong `scripts/verify/` chạy lại được nguyên vẹn. Ví dụ:

```bash
mkdir -p runs/detect/r_clspw_v2/weights weights/trained/frcnn_v5plus
cp yolov8s_cw.pt runs/detect/r_clspw_v2/weights/best.pt
cp fasterrcnn_r50fpn_v5plus.pt weights/trained/frcnn_v5plus/best.pt
```

YOLOv8s (Ultralytics 8.4.126):

```python
from ultralytics import YOLO
m = YOLO("yolov8s_cw.pt")
m.val(data="data/data_v5plus.yaml", split="test", imgsz=960, conf=0.001, iou=0.7, half=True)
```

Faster R-CNN (PyTorch 2.7.1, TorchVision 0.22.1):

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
ck = torch.load("fasterrcnn_r50fpn_v5plus.pt", map_location="cpu", weights_only=False)
m = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, min_size=800, max_size=1333,
                            box_score_thresh=0.001, box_detections_per_img=300, box_nms_thresh=0.5)
m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 5)  # 4 lớp + nền
m.load_state_dict(ck["model"]); m.eval()
```

Nhãn Faster R-CNN là lớp YOLO + 1 (0 là nền): 1 xe máy, 2 ô tô, 3 xe buýt, 4 xe tải. Cấu hình huấn luyện nằm trong `ck["cfg"]`, môi trường trong `ck["env"]`.

Khi so YOLO với Faster R-CNN, dự đoán YOLO phải lấy từ `model.val()` (NMS nhiều nhãn); `model.predict()` mặc định một nhãn mỗi hộp và làm mAP thấp đi khoảng 4,4 điểm (xem `outputs/verify/unified_eval/multilabel_isolation.json`).
