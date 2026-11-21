from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from docx import Document


ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "docs/tieuluan_V3/Nhom02_PhanLoaiPhuongTien_UPDATED5 (1).docx"
BACKUP = ROOT / "docs/tieuluan_V3/Nhom02_PhanLoaiPhuongTien_UPDATED5 (1).before_REPORT_NUMBERS.docx"
TMP = ROOT / "outputs/verify/docx_update_assets"


def set_paragraph(paragraph, text: str) -> None:
    """Replace paragraph contents while retaining paragraph and first-run formatting."""
    if paragraph.runs:
        paragraph.runs[0].text = text
        for run in paragraph.runs[1:]:
            run.text = ""
    else:
        paragraph.add_run(text)


def set_cell(cell, text: str) -> None:
    set_paragraph(cell.paragraphs[0], text)
    for p in cell.paragraphs[1:]:
        set_paragraph(p, "")


def set_row(table, row_index: int, values: list[str]) -> None:
    row = table.rows[row_index]
    if len(row.cells) != len(values):
        raise ValueError(f"table row has {len(row.cells)} cells, expected {len(values)}")
    for cell, value in zip(row.cells, values):
        set_cell(cell, value)


def append_row_like(table, values: list[str]) -> None:
    source_tr = table.rows[-1]._tr
    new_tr = deepcopy(source_tr)
    table._tbl.append(new_tr)
    set_row(table, len(table.rows) - 1, values)


def make_charts() -> tuple[Path, Path]:
    TMP.mkdir(parents=True, exist_ok=True)
    models = ["Baseline", "p4_scale", "p4_1280", "r_p2", "r_clspw_v2", "progressive"]
    v5 = [0.7930, 0.7871, 0.7741, 0.7528, 0.7984, 0.7710]
    aerial_val = [0.6200, 0.6476, 0.6377, 0.5525, 0.6368, 0.6287]
    aerial_test = [0.5996, 0.6374, 0.6008, 0.5380, 0.6018, 0.6100]

    p42 = TMP / "fig_4_2_map50_v5plus.png"
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=180)
    colors = ["#6c8ebf", "#82b366", "#d79b00", "#b85450", "#9673a6", "#4ba3a5"]
    bars = ax.bar(models, v5, color=colors)
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Kết quả trên V5+ test (1.500 ảnh)")
    ax.set_ylim(0.70, 0.82)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, labels=[f"{x:.4f}" for x in v5], padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(p42, bbox_inches="tight")
    plt.close(fig)

    p43 = TMP / "fig_4_3_cross_domain.png"
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=180)
    x = range(len(models))
    width = 0.25
    ax.bar([i - width for i in x], v5, width, label="V5+ test")
    ax.bar(x, aerial_val, width, label="VN Aerial val")
    ax.bar([i + width for i in x], aerial_test, width, label="VN Aerial test")
    ax.set_xticks(list(x), models)
    ax.set_ylabel("mAP@0.5")
    ax.set_title("So sánh trong miền và ngoài miền")
    ax.set_ylim(0.50, 0.83)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(p43, bbox_inches="tight")
    plt.close(fig)
    return p42, p43


def main() -> None:
    if not BACKUP.exists():
        shutil.copy2(DOCX, BACKUP)

    doc = Document(DOCX)
    paragraphs = doc.paragraphs

    replacements = {
        7: "According to the verified evaluation results, r_clspw_v2 achieves mAP@0.5 = 0.7984 and mAP@0.5:0.95 = 0.4283 on the V5+ test set, the best in-domain result among the examined recipes. Faster R-CNN achieves 0.6408 and 0.3605, respectively. Relative to the fair exp_baseline, truck AP@0.5 increases by 2.32 percentage points, bus remains nearly unchanged (+0.04), and motorcycle/car decrease only 0.03/0.15 points. On the out-of-domain VN Aerial test set, p4_scale reaches 0.6374 mAP@0.5, higher than r_clspw_v2 at 0.6018. Model-only latency for r_clspw_v2 is 9.69 ms/image (103.2 FPS); the end-to-end counting pipeline reaches 37.8 FPS at cold start and 80.4 FPS after warm-up.",
        8: "r_clspw_v2 is selected for in-domain deployment, while p4_scale is preferable for the evaluated aerial domain. YOLO and Faster R-CNN are compared on the same 1,500-image V5+ test set. Counting ground truth is available for only one short video and was measured with r_clspw v1; it must be rerun with v2 before broader deployment claims are made.",
        14: "Theo kết quả đánh giá đã kiểm chứng, r_clspw_v2 đạt mAP@0,5 = 0,7984 và mAP@0,5:0,95 = 0,4283 trên test V5+, cao nhất trong nhóm recipe đã khảo sát. Faster R-CNN đạt tương ứng 0,6408 và 0,3605. So với baseline exp_baseline, AP@0,5 của truck tăng 2,32 điểm phần trăm, bus gần như giữ nguyên (+0,04 điểm), còn motorcycle và car chỉ giảm 0,03 và 0,15 điểm. Trên VN Aerial test ngoài miền huấn luyện, p4_scale đạt 0,6374 mAP@0,5, cao hơn r_clspw_v2 với 0,6018. Độ trễ model-only của r_clspw_v2 là 9,69 ms/ảnh (103,2 FPS); pipeline đếm đạt 37,8 FPS ở lần chạy cold-start và 80,4 FPS sau khi mô hình đã warm-up.",
        15: "r_clspw_v2 phù hợp mục tiêu in-domain, còn p4_scale tốt hơn trên VN Aerial đã khảo sát. YOLO và Faster R-CNN được so sánh trên cùng tập test V5+ gồm 1.500 ảnh. Kết quả đếm có GT hiện chỉ gồm một video và được đo với r_clspw v1; cần chạy lại bằng v2 trước khi khái quát.",
        101: "Bảng 4.8 Số TP, FP, FN, Precision và Recall theo lớp của r_clspw_v2 trên test V5+ (conf = 0,25; IoU ghép = 0,5). 55",
        102: "Bảng 4.9 Số TP / FP / FN của r_clspw_v2 theo kích thước hộp (chuẩn COCO) và theo lớp trên test V5+. 55",
        103: "Bảng 4.10 Kết quả đếm phương tiện của r_clspw v1 trên video demo_traffic.mp4. 56",
        117: "Nhóm lựa chọn YOLOv8 làm mô hình chính vì có tốc độ xử lý cao phù hợp với video thời gian thực và cộng đồng nghiên cứu lớn với mã nguồn mở Ultralytics được bảo trì tích cực. Faster R-CNN với backbone ResNet50-FPN được chọn làm mô hình đối chứng do đại diện cho hướng tiếp cận hai giai đoạn kinh điển. Hai mô hình được đánh giá trên cùng tập test V5+ gồm 1.500 ảnh và cùng phần cứng để bảo đảm điều kiện đánh giá nhất quán.",
        129: "Tinh chỉnh YOLOv8s từ trọng số pretrained COCO, kiểm chứng năm chiến lược cải tiến trên nền V5+ và chọn r_clspw_v2 làm mô hình cuối; xây dựng Faster R-CNN ResNet50-FPN làm mô hình đối chứng và so sánh trên cùng test V5+.",
        135: "Câu hỏi 1. Trên cùng test V5+ gồm 1.500 ảnh, r_clspw_v2 và Faster R-CNN ResNet50-FPN khác nhau thế nào về mAP, kết quả theo lớp và tốc độ suy luận?",
        136: "Câu hỏi 2. r_clspw_v2 với cls_pw=0,5 thay đổi AP từng lớp, Precision và Recall thế nào so với baseline exp_baseline khi warm-start nhất quán?",
        138: "Câu hỏi 4. r_clspw_v2 có đạt mAP@0.5 ≥ 0,75 trên V5+ test không, và thứ hạng với p4_scale thay đổi thế nào trên VN Aerial ngoài miền?",
        139: "Câu hỏi 5. Pipeline theo vết và đếm có sai số, tốc độ end-to-end thế nào, và kết quả nào đã được xác minh cho checkpoint v1/v2?",
        154: "Huấn luyện hai mô hình YOLOv8s và Faster R-CNN, sau đó đánh giá trên cùng phần cứng và cùng tập test V5+ (Chương 4).",
        156: "Phân tích pipeline đếm với ByteTrack trên video demo có ground truth; đánh giá sai số đếm và tốc độ end-to-end, đồng thời tách rõ checkpoint dùng cho từng phép đo (Chương 5).",
        161: "Trong nhóm recipe V5+, r_clspw_v2 có mAP@0.5 cao nhất trên test hiện có: 0,7984; mAP@0.5:0.95 = 0,4283. Chênh lệch với Faster R-CNN lần lượt là 15,76 và 6,78 điểm phần trăm. Đây là lựa chọn theo kết quả so sánh hậu nghiệm, chưa phải ước lượng độc lập sau khi khóa cấu hình.",
        166: "Về cải tiến mô hình. Đề tài khảo sát năm recipe trên nền V5+, gồm augmentation không gian, tăng độ phân giải, P2, class weighting và progressive resolution. r_clspw_v2 với cls_pw=0,5 đạt mAP@0.5 = 0,7984, cao hơn baseline exp_baseline 0,54 điểm và p4_scale 1,13 điểm phần trăm. AP truck tăng 2,32 điểm, AP bus gần như giữ nguyên (+0,04 điểm), trong khi motorcycle/car chỉ giảm 0,03/0,15 điểm. Đóng góp là khảo sát thực nghiệm cơ chế trọng số lớp có sẵn, không đề xuất hàm mất mát mới.",
        167: "Đề tài đồng thời chỉ ra trade-off giữa độ chính xác trong miền và khả năng tổng quát hóa: r_clspw_v2 tốt nhất trên V5+, trong khi p4_scale tốt nhất trên VN Aerial test. Kết quả giúp lựa chọn mô hình theo bối cảnh camera.",
        175: "Chương 5 mô tả ứng dụng Streamlit, kết quả tốc độ đã đo cho r_clspw_v2 và giới hạn của phép đếm ground-truth hiện mới dùng r_clspw v1 trên một video.",
        298: "Nền tảng lập trình: Python 3.12 làm ngôn ngữ chính; PyTorch 2.7.1+cu118 với CUDA Runtime 11.8 cho tính toán GPU và huấn luyện mạng nơ-ron.",
        299: "Ultralytics 8.4.126 cung cấp YOLOv8, huấn luyện, đánh giá và theo vết. TorchVision 0.22.1+cu118 cung cấp Faster R-CNN ResNet50-FPN. Các checkpoint YOLO chính đã được chấm lại bằng model.val() với cùng cấu hình đánh giá; Faster R-CNN được đánh giá riêng bằng pycocotools trên cùng test V5+.",
        302: "Theo dõi thí nghiệm và tái lập: mỗi run YOLO lưu cấu hình args.yaml, results.csv, checkpoint và biểu đồ trong runs/detect/<run_name>; các run YOLO dùng seed=42. Notebook Faster R-CNN không đặt seed cố định, vì vậy báo cáo không giả định seed=42 cho mô hình này.",
        317: "Huấn luyện baseline exp_baseline và khảo sát p4_scale, p4_imgsz1280, r_p2, r_clspw_v2 và r_progressive. r_clspw_v2 warm-start từ exp_baseline/best.pt; r_p2 thay đổi kiến trúc và dùng phần trọng số COCO tương thích. Faster R-CNN được huấn luyện riêng trên V6 làm đối chứng.",
        318: "Trong mỗi run YOLO, best.pt được chọn theo fitness validation của Ultralytics 8.4.126, chủ yếu dựa trên mAP@0.5:0.95. Việc chọn recipe r_clspw_v2 theo kết quả test là quyết định hậu nghiệm khác với chọn checkpoint trong một run.",
        388: "Các số lượng của biến thể sạch là phương án dữ liệu riêng; không phải số lượng đã dùng để tạo kết quả r_clspw_v2 hiện có.",
        400: "Baseline V5+, p4_scale và r_clspw_v2 dùng imgsz=960. Các nhánh tăng độ phân giải và progressive có giai đoạn dùng 1.280 pixel; V5 honest dùng 640. Khi suy luận phải ghi rõ imgsz thực tế của từng checkpoint.",
        402: "Ràng buộc phần cứng: với GPU 12 GB VRAM, baseline exp_baseline ở imgsz=960 dùng batch=6. r_clspw_v2 fine-tune ở cùng độ phân giải với batch=16; nhánh imgsz=1.280 phải giảm batch theo cấu hình từng run.",
        410: "YOLOv8s được chọn để khảo sát các recipe trên một kiến trúc cơ sở và có thể tích hợp với bộ theo vết. Các tệp args.yaml, results.csv và checkpoint của các run chính đã được lưu trong runs/detect/ và dùng để đối chiếu cấu hình, thời gian huấn luyện và kết quả.",
        414: "Giả thuyết: dữ liệu V5+ đa dạng hơn kết hợp ảnh 960 pixel sẽ cải thiện khả năng phát hiện vật nhỏ và các lớp ít mẫu so với V5 honest. Cấu hình thực: runs/detect/train_v8s_v5_honest/args.yaml và runs/detect/exp_baseline/args.yaml.",
        428: "Tập V5+ mất cân bằng lớp, vì vậy r_clspw đặt cls_pw=0,5. Ultralytics 8.4.126 dùng số instance train để tính trọng số (1/n_c)^cls_pw, sau đó chuẩn hóa để trung bình bốn trọng số bằng 1. Các trọng số là [0,571; 0,860; 1,629; 0,940] theo thứ tự motorcycle, car, bus, truck. BCE classification loss của từng lớp được nhân với trọng số tương ứng; box loss và DFL không đổi.",
        430: "Giả thuyết: giảm đóng góp tương đối của lớp phổ biến và tăng đóng góp của lớp hiếm trong BCE sẽ cải thiện AP bus/truck và mAP tổng. Mô hình cuối r_clspw_v2 warm-start công bằng từ exp_baseline/best.pt. Cấu hình thực: runs/detect/r_clspw_v2/args.yaml.",
        436: "Checkpoint trong từng run được chọn theo validation. Việc chọn recipe r_clspw_v2 đã tham khảo V5+ test và VN Aerial, nên là lựa chọn hậu nghiệm; test không cập nhật trọng số nhưng đã tham gia quyết định cấu hình. Quy trình tiếp theo cần dùng validation độc lập theo camera để chọn recipe, khóa cấu hình rồi mới đánh giá test.",
        438: "Để trả lời Câu hỏi nghiên cứu 1, đề tài xây dựng mô hình Faster R-CNN ResNet50-FPN làm đối chứng và đánh giá trên cùng test V5+ gồm 1.500 ảnh như YOLO.",
        446: "YOLOv8s và Faster R-CNN được đánh giá trên cùng test V5+ gồm 1.500 ảnh và cùng GPU. YOLO được tính bằng Ultralytics model.val(), còn Faster R-CNN bằng pycocotools với cùng quy ước bốn lớp và các ngưỡng đánh giá tương ứng.",
        447: "Tốc độ model-only được đo trên cùng GPU với 20 ảnh warm-up và 200 ảnh đo, tách khỏi tốc độ end-to-end. r_clspw_v2 dùng imgsz=960, FP16; Faster R-CNN dùng kích thước mặc định 800/1.333, FP32.",
        503: "Mô hình đối chứng: Faster R-CNN ResNet50-FPN với cấu hình chuẩn TorchVision, được đánh giá trên cùng test V5+ với YOLO.",
        501: "Ảnh và augmentation được thiết lập theo từng recipe: r_clspw_v2 dùng imgsz=960, cls_pw=0,5; scale=0,7 và translate=0,15 thuộc p4_scale. Không gán cấu hình của p4_scale cho r_clspw_v2.",
        502: "Mô hình chính: YOLOv8s r_clspw_v2 với class-weighted loss, warm-start từ exp_baseline và được chọn sau khi so sánh năm recipe trên V5+ test.",
        505: "Đánh giá detection bằng mAP/AP theo lớp trên V5+ và VN Aerial; phân rã TP/FP/FN của r_clspw_v2 theo lớp và kích thước; đánh giá sai số đếm trên video demo và tốc độ end-to-end.",
        509: "Chương này kiểm chứng năm cải tiến đã nêu ở §3.5 trên test V5+, đánh giá khả năng tổng quát hóa trên VN Aerial, đối chiếu mô hình cuối với Faster R-CNN và báo cáo kết quả đếm. Các chỉ số được đối chiếu từ kết quả chạy trong outputs/verify/ và log của từng run.",
        512: "Thực nghiệm kiểm chứng chạy trên GPU NVIDIA RTX 3500 Ada Generation Laptop 12 GB, Python 3.12, PyTorch 2.7.1+cu118, TorchVision 0.22.1+cu118 và Ultralytics 8.4.126. Các recipe YOLO dùng AMP và checkpoint tốt nhất của từng run.",
        519: "Ultralytics lưu box loss, classification loss, DFL và chỉ số validation theo epoch trong results.csv. Hình 4.1 được dựng lại từ runs/detect/exp_baseline/results.csv và phản ánh baseline V5+ đúng 7.237 ảnh.",
        522: "Baseline exp_baseline chạy đủ 40 epoch và đạt checkpoint tốt nhất ở epoch 26. r_clspw_v2 early-stop tại epoch 10/15 (patience=8), checkpoint tốt nhất ở epoch 2. r_p2 chạy đủ 25 epoch; progressive gồm hai giai đoạn 20 và 10 epoch.",
        525: "Số epoch và thời gian trong bảng được lấy từ cột time (giây tích lũy) ở dòng cuối results.csv. Baseline exp_baseline chạy 40 epoch trong 1 giờ 27 phút 30 giây; r_clspw_v2 chạy 10 epoch trong 24 phút 38 giây. Chi phí fine-tune không so trực tiếp với huấn luyện từ COCO nếu ngân sách khác nhau.",
        529: "r_clspw_v2 đạt mAP@0.5 cao nhất (0,7984), cao hơn baseline 0,54 điểm và p4_scale 1,13 điểm. Mô hình cũng có recall cao nhất trong nhóm (0,7445) và được chọn làm mô hình cuối cho bối cảnh in-domain.",
        534: "r_clspw_v2 đạt AP truck cao nhất (0,6907), AP bus gần tương đương baseline; r_progressive tốt nhất ở motorcycle còn baseline tốt nhất ở car. Cải thiện truck đủ để đưa mAP tổng của r_clspw_v2 lên cao nhất, trong khi các lớp khác gần như giữ ổn định.",
        537: "So với baseline exp_baseline, p4_scale giảm mAP@0.5 từ 0,7930 xuống 0,7871 (−0,59 điểm) trên V5+ test, nhưng tăng từ 0,5996 lên 0,6374 (+3,78 điểm) trên VN Aerial test. Recipe không cải thiện in-domain nhưng cho khả năng cross-domain tốt nhất; do đồng thời warm-start, giảm learning rate và đổi batch, không quy toàn bộ chênh lệch cho riêng scale hoặc translate.",
        539: "p4_imgsz1280 đạt mAP@0.5 = 0,7741, thấp hơn baseline 1,89 điểm dù recall tăng từ 0,7283 lên 0,7427. AP truck giảm từ 0,6675 xuống 0,6049. Như vậy tăng đầu vào lên 1.280 pixel không cải thiện mục tiêu tổng thể trong cấu hình đã thử; precision giảm còn 0,7285.",
        541: "r_p2 đạt 0,7528 mAP@0.5, thấp hơn baseline 4,02 điểm. AP bus gần như không đổi (0,7448 so với 0,7458) nhưng AP truck giảm 12,27 điểm. Kết quả không ủng hộ giả thuyết thêm head stride 4 tự động cải thiện toàn bộ bài toán vật nhỏ.",
        543: "r_clspw_v2 tăng mAP@0.5 0,54 điểm so với baseline exp_baseline. AP truck tăng từ 0,6675 lên 0,6907 (+2,32 điểm), bus gần như giữ nguyên (+0,04), còn motorcycle/car chỉ giảm 0,03/0,15 điểm. Với warm-start nhất quán từ baseline, kết quả ủng hộ class-weighted loss cho lớp mất cân bằng, dù mức tăng tổng còn nhỏ.",
        546: "r_progressive đạt AP motorcycle cao nhất 0,8707 nhưng mAP@0.5 tổng chỉ 0,7710, thấp hơn baseline 2,20 điểm. AP truck giảm 5,28 điểm so với baseline. Giả thuyết curriculum 960→1.280 cải thiện đồng đều các lớp không được ủng hộ.",
        548: "V6 bổ sung 4.396 ảnh pseudo-label nhưng đạt mAP@0.5 = 0,7608, thấp hơn baseline exp_baseline 3,22 điểm. AP bus 0,7427 cũng thấp hơn nhẹ baseline 0,7458. Kết quả cho thấy bổ sung dữ liệu tự gán nhãn không bảo đảm cải thiện nếu phân bố nguồn và chất lượng nhãn không được kiểm soát.",
        555: "r_clspw_v2 có mAP@0.5 cao nhất trên V5+ (0,7984), còn p4_scale cao nhất trên VN Aerial test (0,6374). p4_scale có gap in-domain→aerial nhỏ nhất (14,97 điểm), vì vậy phù hợp hơn khi ưu tiên cross-domain; r_clspw_v2 phù hợp hơn với camera gần miền huấn luyện.",
        559: "r_clspw_v2 vượt Faster R-CNN 15,76 điểm mAP@0.5 và 6,78 điểm mAP@0.5:0.95 trên cùng 1.500 ảnh test. AP@0.5:0.95 theo lớp của Faster R-CNN là 0,2694 (motorcycle), 0,5115 (car), 0,3727 (bus) và 0,2886 (truck).",
        557: "4.6 So sánh r_clspw_v2 với Faster R-CNN",
        560: "4.7 Phân tích lỗi của r_clspw_v2",
        562: "AP@0.5 của r_clspw_v2 lần lượt là 0,8608 cho motorcycle, 0,8961 cho car, 0,7462 cho bus và 0,6907 cho truck. So với baseline, truck tăng 2,32 điểm, bus gần như giữ nguyên, còn motorcycle/car giảm không đáng kể trong ngưỡng dao động FP16.",
        563: "Ở ngưỡng conf=0,25, NMS IoU=0,7 và IoU ghép=0,5 trên 1.500 ảnh test, motorcycle có precision/recall 0,8731/0,7431; car 0,7247/0,9031; bus 0,7705/0,6574; truck 0,6593/0,6315. Motorcycle có 10.206 FN do chiếm phần lớn bbox trong test.",
        564: "Bảng 4.8 Số TP, FP, FN, Precision và Recall theo lớp của r_clspw_v2 trên test V5+ (conf = 0,25; IoU ghép = 0,5).",
        566: "Phân rã theo kích thước cho thấy recall motorcycle đạt 49,8% ở nhóm nhỏ, 79,3% ở nhóm vừa và 95,4% ở nhóm lớn; với car lần lượt là 75,5%, 90,1% và 93,7%. Bus đạt 57,9% ở nhóm vừa và 77,1% ở nhóm lớn; truck đạt 48,2% và 86,1%. Tập test chỉ gồm cam_09/cam_10 ban ngày nên không thể kết luận chênh lệch ngày–đêm.",
        567: "Bảng 4.9 Số TP / FP / FN của r_clspw_v2 theo kích thước hộp (chuẩn COCO) và theo lớp trên test V5+.",
        568: "Ma trận nhầm lẫn cho thấy nhầm lớp chủ yếu giữa car và bus (207 trường hợp hai chiều), car và truck (135), bus và truck (103). Motorcycle ít nhầm lớp (47 trường hợp với các lớp còn lại); lỗi chính là 10.206 FN và 4.280 FP nền.",
        570: "Pipeline đếm (ByteTrack, conf=0,3, IoU=0,5, imgsz=960, FP16) được đo trên demo_traffic.mp4 với checkpoint r_clspw v1 trước khi retrain v2. Vạch ngang ở y=474 và đếm cả hai chiều. Vì chưa chạy lại phép đếm ground-truth với v2, kết quả Bảng 4.10 chỉ được dùng để minh họa pipeline.",
        571: "Bảng 4.10 Kết quả đếm phương tiện của r_clspw v1 trên video demo_traffic.mp4.",
        575: "Trên cùng test V5+, r_clspw_v2 đạt 0,7984 mAP@0.5 so với 0,6408 của Faster R-CNN, chênh 15,76 điểm. Ở mAP@0.5:0.95, chênh lệch là 6,78 điểm. Độ trễ model-only tương ứng là 9,69 ms/ảnh và 47,34 ms/ảnh, nhưng đầu vào và precision khác nhau.",
        577: "r_clspw_v2 vượt baseline 0,54 điểm mAP@0.5, chủ yếu nhờ AP truck tăng 2,32 điểm trong khi các lớp khác gần như giữ ổn định. Kết quả warm-start nhất quán ủng hộ class weighting cho lớp mất cân bằng; p4_scale thấp hơn trong miền nhưng cao nhất trên VN Aerial test.",
        580: "Truck có AP và precision thấp nhất (0,6907 và 0,6593). Theo phân rã kích thước, recall thấp nhất xuất hiện ở vật nhỏ/vừa, đặc biệt motorcycle nhỏ (49,8%) và truck vừa (48,2%). Nhầm lớp chủ yếu là car–bus, car–truck và bus–truck. Do tập test chỉ có ảnh ban ngày, chưa thể kết luận về lỗi ban đêm.",
        582: "r_clspw_v2 đạt 0,7984 mAP@0.5 trên V5+ test; p4_scale đạt 0,6374 trên VN Aerial test, cao hơn r_clspw_v2 với 0,6018. YOLOv8s-COCO zero-shot đạt 0,4514 trên V5+ test và 0,4893 trên Aerial test. Như vậy r_clspw_v2 cao hơn zero-shot 34,70 điểm trên V5+ và 11,25 điểm trên Aerial test.",
        584: "Trên demo_traffic.mp4 (87 lượt GT), phép đếm với r_clspw v1 cho 86 lượt, sai số 1,15%. Benchmark tốc độ riêng với r_clspw_v2 đạt 37,8 FPS cold-start, 80,4 FPS warm khi không ghi video và 45,1 FPS warm khi ghi MP4. Cần thêm video có GT và chạy lại đếm với v2 trước khi khái quát.",
        586: "Kết quả xác nhận r_clspw_v2 là mô hình cuối cho V5+ với mAP@0.5 = 0,7984; p4_scale phù hợp hơn khi ưu tiên cross-domain aerial. P2, tăng độ phân giải đơn thuần, progressive curriculum và pseudo-labeling không cải thiện mục tiêu in-domain trong cấu hình đã thử.",
        589: "Chương này mô tả ứng dụng Streamlit theo mã nguồn hiện có. r_clspw_v2 là mô hình cuối cho in-domain và đã được thêm vào danh sách ưu tiên tự nạp; p4_scale được giữ làm lựa chọn cross-domain.",
        613: "Sidebar liệt kê các checkpoint YOLO khả dụng. Danh sách ưu tiên hiện đặt r_clspw_v2 ở vị trí đầu, tiếp theo là exp_baseline, p4_scale, V6 pseudo và V5 honest. Bộ nạp hiện tại chưa tích hợp Faster R-CNN.",
        647: "Hình 5.2 minh họa bố cục giao diện Streamlit. Checkpoint mặc định hiện là r_clspw_v2; kết quả tốc độ và giới hạn kiểm thử được trình bày tại §5.4.",
        658: "Với checkpoint r_clspw_v2, độ trễ model-only FP16 là 9,69 ms/ảnh (103,2 FPS), nhanh khoảng 4,9 lần Faster R-CNN (47,34 ms/ảnh, 21,1 FPS); hai mô hình dùng kích thước đầu vào và precision khác nhau. Pipeline đạt 37,8 FPS cold-start, 80,4 FPS warm khi không ghi và 45,1 FPS warm khi ghi MP4. Chuyển mã H.264 preset fast mất khoảng 1,25 giây cho video mẫu.",
        675: "Ứng dụng hiện có tải video, lựa chọn checkpoint YOLO, một vạch ngang/dọc, theo vết, thống kê và xuất kết quả. r_clspw_v2 là checkpoint ưu tiên; pipeline đạt 37,8 FPS cold-start và 80,4 FPS warm. Kết quả đếm 1,15% sai số mới chỉ được xác minh với r_clspw v1 trên một video, nên cần chạy lại v2 và mở rộng bộ GT.",
        681: "Đóng góp 3 — So sánh định lượng YOLOv8s và Faster R-CNN. Trên cùng test V5+ 1.500 ảnh, YOLOv8s r_clspw_v2 đạt mAP@0.5 = 0,7984, vượt Faster R-CNN 15,76 điểm (0,6408).",
        682: "Đóng góp 4 — Khảo sát class weighting. r_clspw_v2 với cls_pw=0,5 đạt 0,7984 mAP@0.5, tăng 0,54 điểm so với baseline fair; AP truck tăng 2,32 điểm, bus gần như giữ nguyên và motorcycle/car không giảm đáng kể. p4_scale tốt nhất trên VN Aerial test.",
        683: "Đóng góp 5 — Cài đặt ứng dụng demo. Streamlit có pipeline phát hiện, theo vết, đếm qua một vạch và xuất CSV/video. Với r_clspw_v2, pipeline đạt 37,8 FPS cold-start và 80,4 FPS warm; phép đếm sai số 1,15% hiện mới được xác minh bằng r_clspw v1 trên một video.",
        687: "Hạn chế 2 — Khoảng cách cross-domain còn lớn. r_clspw_v2 đạt 0,6018 mAP@0.5 trên VN Aerial test, thấp hơn V5+ test 19,66 điểm. Bộ cross-domain chỉ có 635 ảnh và một kiểu góc nhìn trên cao, chưa đại diện đầy đủ cho mọi camera.",
        688: "Hạn chế 3 — Chưa có nhãn track ID để đo IDF1/MOTA/HOTA. Phép đếm có GT chỉ gồm một video và dùng r_clspw v1; cần chạy lại với v2. Validation cũ còn trùng camera, còn quy trình chọn recipe đã tham khảo test.",
        690: "Hạn chế 5 — V6 pseudo chưa cải thiện mAP tổng. Kết quả 0,7608 thấp hơn baseline exp_baseline 0,7930. Mất cân bằng lớp, sai nhãn giả và lệch miền là các khả năng cần kiểm tra.",
        696: "Hướng 3 — Tối ưu phát hiện vật nhỏ sau negative result P2. r_p2 chỉ đạt 0,7528 mAP@0.5, thấp hơn baseline. Nên thử tile inference, sampling theo kích thước hoặc huấn luyện dài hơn và đo AP theo small/medium/large.",
        703: "Đề tài đã chuẩn hóa V5+, khảo sát các recipe YOLOv8s, so sánh Faster R-CNN và xây dựng ứng dụng demo. r_clspw_v2 đạt 0,7984 mAP@0.5 trên V5+ test và p4_scale đạt 0,6374 trên VN Aerial test. Việc chọn recipe đã tham khảo test/validation cũ nên kết luận vẫn cần được củng cố bằng quy trình chọn cấu hình trên validation độc lập theo camera.",
        704: "Checkpoint, log huấn luyện, kết quả đánh giá và benchmark tốc độ của các run chính đã được đối chiếu. Hạn chế còn lại là phép đếm GT mới có một video và chưa được chạy lại với r_clspw_v2, cùng thiếu đánh giá tracking theo ID.",
        719: "[12]\tUltralytics, “Detection loss implementation,” source code, v8.4.126. Accessed: Nov. 16, 2026. [Online]. Available: https://github.com/ultralytics/ultralytics/blob/v8.4.126/ultralytics/utils/loss.py",
        724: "[17]\tUltralytics, “YOLOv8 detection model configuration,” source code, v8.4.126. Accessed: Nov. 16, 2026. [Online]. Available: https://github.com/ultralytics/ultralytics/blob/v8.4.126/ultralytics/cfg/models/v8/yolov8.yaml",
        739: "[32]\tUltralytics, “Detection metrics and checkpoint fitness,” source code, v8.4.126. Accessed: Nov. 16, 2026. [Online]. Available: https://github.com/ultralytics/ultralytics/blob/v8.4.126/ultralytics/utils/metrics.py",
    }
    for idx, text in replacements.items():
        set_paragraph(paragraphs[idx], text)

    tables = doc.tables
    set_row(tables[4], 4, ["Độ trễ suy luận đo được (RTX 3500 Ada, batch 1)", "9,69 ms/ảnh (≈103 FPS), imgsz 960, FP16", "47,34 ms/ảnh (≈21 FPS), 800/1.333, FP32"])

    set_row(tables[12], 4, ["batch", "16", "6"])
    if len(tables[12].rows) == 10:
        append_row_like(tables[12], ["epochs thực chạy", "40 (best@33)", "40 (best@26)"])
        append_row_like(tables[12], ["close_mosaic", "10", "10"])
        append_row_like(tables[12], ["data", "data_v5.yaml → merged_v5", "_data_v5plus_local.yaml → merged_v5plus"])
        append_row_like(tables[12], ["Checkpoint", "runs/detect/train_v8s_v5_honest/weights/best.pt", "runs/detect/exp_baseline/weights/best.pt"])
    set_row(tables[13], 2, ["imgsz / batch", "960 / 6", "960 / 6"])
    set_row(tables[14], 3, ["batch", "6", "6"])
    set_row(tables[16], 3, ["imgsz / batch", "960 / 6", "960 / 16"])

    set_row(tables[18], 12, ["augmentation", "flip ngang 0,5 + brightness/contrast + hue shift", "Không chỉ có flip; notebook có augmentation màu"])
    set_row(tables[18], 13, ["seed", "Không đặt cố định", "Notebook không gọi torch.manual_seed(42)"])
    if len(tables[18].rows) == 15:
        append_row_like(tables[18], ["Dữ liệu train", "data/merged_v6 (7.229 ảnh)", "Gồm V5 base + 4.396 ảnh pseudo-label"])
        append_row_like(tables[18], ["Dữ liệu eval", "V5+ test (1.500 ảnh)", "Cùng tập test dùng cho YOLO"])
    for row in list(tables[18].rows):
        if row.cells[0].text.strip() == "Dữ liệu train":
            row._tr.getparent().remove(row._tr)

    set_row(tables[19], 1, ["V5+ baseline", "YOLOv8s, 960 pixel", "runs/detect/exp_baseline/args.yaml"])
    set_row(tables[19], 5, ["r_clspw_v2", "cls_pw=0,5; warm-start fair", "runs/detect/r_clspw_v2/args.yaml"])

    set_row(tables[20], 1, ["V5+ baseline (exp_baseline)", "40", "1 giờ 27 phút 30 giây"])
    set_row(tables[20], 5, ["r_clspw_v2, mô hình cuối", "10", "24 phút 38 giây"])

    table21 = [
        ["Mô hình", "mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall"],
        ["V5+ baseline (exp_baseline)", "0,7930", "0,4279", "0,7643", "0,7283"],
        ["p4_scale", "0,7871", "0,4236", "0,7581", "0,7327"],
        ["p4_imgsz1280", "0,7741", "0,4175", "0,7285", "0,7427"],
        ["r_p2", "0,7528", "0,3944", "0,7409", "0,7075"],
        ["r_clspw_v2, mô hình cuối", "0,7984", "0,4283", "0,7648", "0,7445"],
        ["r_progressive", "0,7710", "0,4148", "0,7405", "0,7211"],
    ]
    for i, row in enumerate(table21): set_row(tables[21], i, row)

    table22 = [
        ["Mô hình", "motorcycle", "car", "bus", "truck"],
        ["V5+ baseline (exp_baseline)", "0,8611", "0,8976", "0,7458", "0,6675"],
        ["p4_scale", "0,8608", "0,8854", "0,7218", "0,6802"],
        ["p4_imgsz1280", "0,8612", "0,8886", "0,7417", "0,6049"],
        ["r_p2", "0,8376", "0,8838", "0,7448", "0,5448"],
        ["r_clspw_v2", "0,8608", "0,8961", "0,7462", "0,6907"],
        ["r_progressive", "0,8707", "0,8778", "0,7209", "0,6147"],
    ]
    for i, row in enumerate(table22): set_row(tables[22], i, row)

    table23 = [
        ["Lớp", "Trọng số loss", "AP baseline", "AP r_clspw_v2", "Δ tuyệt đối"],
        ["motorcycle", "0,571", "0,8611", "0,8608", "−0,03 điểm"],
        ["car", "0,860", "0,8976", "0,8961", "−0,15 điểm"],
        ["bus", "1,629", "0,7458", "0,7462", "+0,04 điểm"],
        ["truck", "0,940", "0,6675", "0,6907", "+2,32 điểm"],
    ]
    for i, row in enumerate(table23): set_row(tables[23], i, row)
    if len(tables[23].rows) == 5:
        append_row_like(tables[23], ["mAP@0.5 tổng", "—", "0,7930", "0,7984", "+0,54 điểm"])

    table24 = [
        ["Mô hình", "V5+ test", "VN Aerial val", "VN Aerial test", "Gap đến Aerial test"],
        ["V5+ baseline (exp_baseline)", "0,7930", "0,6200", "0,5996", "19,34 điểm"],
        ["p4_scale", "0,7871", "0,6476", "0,6374", "14,97 điểm"],
        ["p4_imgsz1280", "0,7741", "0,6377", "0,6008", "17,33 điểm"],
        ["r_p2", "0,7528", "0,5525", "0,5380", "21,48 điểm"],
        ["r_clspw_v2", "0,7984", "0,6368", "0,6018", "19,66 điểm"],
        ["r_progressive", "0,7710", "0,6287", "0,6100", "16,10 điểm"],
    ]
    for i, row in enumerate(table24): set_row(tables[24], i, row)

    set_row(tables[25], 2, ["YOLOv8s r_clspw_v2", "0,7984", "0,4283"])
    set_row(tables[25], 3, ["Chênh lệch tuyệt đối", "+15,76 điểm", "+6,78 điểm"])

    table26 = [
        ["Lớp", "TP", "FP", "FN", "Precision", "Recall"],
        ["motorcycle", "29.517", "4.290", "10.206", "0,8731", "0,7431"],
        ["car", "7.241", "2.751", "777", "0,7247", "0,9031"],
        ["bus", "497", "148", "259", "0,7705", "0,6574"],
        ["truck", "538", "278", "314", "0,6593", "0,6315"],
    ]
    for i, row in enumerate(table26): set_row(tables[26], i, row)

    table27 = [
        ["Kích thước", "motorcycle", "car", "bus", "truck"],
        ["Nhỏ (< 32²)", "3.599 / 1.457 / 3.625", "293 / 1.225 / 95", "0 / 1 / 6", "0 / 0 / 0"],
        ["Vừa (32²–96²)", "25.007 / 2.737 / 6.537", "4.984 / 1.138 / 549", "245 / 71 / 178", "248 / 89 / 267"],
        ["Lớn (> 96²)", "911 / 86 / 44", "1.964 / 105 / 133", "252 / 20 / 75", "290 / 46 / 47"],
    ]
    for i, row in enumerate(table27): set_row(tables[27], i, row)

    table29 = [
        ["Recipe", "Giả thuyết chính", "Kết quả thực tế", "Kết luận"],
        ["p4_scale", "Augmentation mạnh tăng robustness", "−0,59 V5+; +3,78 Aerial test", "Ủng hộ cross-domain; không ủng hộ in-domain"],
        ["p4_imgsz1280", "Độ phân giải cao tăng mAP", "−1,89 V5+; +0,12 Aerial test", "Không ủng hộ in-domain"],
        ["r_p2", "Head stride 4 giúp vật nhỏ", "−4,02 V5+; −6,16 Aerial test", "Không ủng hộ"],
        ["r_clspw_v2", "Class weighting giúp lớp hiếm", "+0,54 V5+; truck +2,32", "Ủng hộ; mức tăng nhỏ"],
        ["r_progressive", "Curriculum 960→1.280", "−2,20 V5+; +1,04 Aerial test", "Không ủng hộ in-domain"],
    ]
    for i, row in enumerate(table29): set_row(tables[29], i, row)

    table30 = [
        ["Mô hình", "Vai trò nghiên cứu", "Trạng thái ứng dụng"],
        ["r_clspw_v2", "mAP@0.5 V5+ cao nhất: 0,7984", "Ưu tiên số 1; mặc định in-domain"],
        ["p4_scale", "mAP@0.5 VN Aerial test cao nhất: 0,6374", "Ưu tiên số 3; lựa chọn cross-domain"],
        ["V5+ baseline (exp_baseline)", "Baseline 7.237 ảnh: 0,7930", "Ưu tiên số 2; fallback"],
        ["V6 pseudo", "Thử nghiệm bổ sung; không phải mô hình cuối", "Có trong danh sách ưu tiên"],
        ["Faster R-CNN", "Đối chứng Chương 4", "Không có adapter UI"],
    ]
    for i, row in enumerate(table30): set_row(tables[30], i, row)
    if len(tables[30].rows) == 6:
        append_row_like(tables[30], ["V5 honest", "Fallback từ COCO", "Có trong danh sách ưu tiên"])

    table32 = [
        ["Đại lượng", "Điều kiện đo", "Kết quả"],
        ["Độ trễ YOLOv8s r_clspw_v2", "imgsz 960, batch 1, FP16; warm-up 20, đo 200 ảnh", "9,69 ms/ảnh (median 8,52; p95 14,30) ≈ 103,2 FPS"],
        ["Độ trễ Faster R-CNN ResNet50-FPN", "800/1.333, batch 1, FP32; cùng quy trình", "47,34 ms/ảnh (median 46,82; p95 51,01) ≈ 21,1 FPS"],
        ["FPS end-to-end, không ghi video", "demo_traffic.mp4, cold-start", "37,8 FPS (4,71 giây / 178 frame)"],
        ["FPS end-to-end, không ghi video", "demo_traffic.mp4, warm", "80,4 FPS (2,21 giây / 178 frame)"],
        ["FPS end-to-end, có ghi MP4", "demo_traffic.mp4, warm, mp4v", "45,1 FPS (3,95 giây / 178 frame)"],
    ]
    for i, row in enumerate(table32): set_row(tables[32], i, row)

    fig42, fig43 = make_charts()
    image_replacements = {
        520: ROOT / "outputs/verify/fig_4_1_training_curves_exp_baseline.png",
        530: fig42,
        552: fig43,
    }
    for p_idx, image_path in image_replacements.items():
        blips = paragraphs[p_idx]._p.xpath('.//a:blip')
        if len(blips) != 1:
            raise RuntimeError(f"expected one image in paragraph {p_idx}, found {len(blips)}")
        rid = blips[0].get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
        doc.part.related_parts[rid]._blob = image_path.read_bytes()

    doc.save(DOCX)
    print(f"Updated: {DOCX}")
    print(f"Backup:  {BACKUP}")


if __name__ == "__main__":
    main()
