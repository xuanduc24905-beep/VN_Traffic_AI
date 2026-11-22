# Phản hồi góp ý vòng 2 — Bài báo SSRC, Nhóm 02

Kính gửi cô,

Nhóm em cảm ơn cô đã góp ý. Nhóm em gửi kèm bản sửa `Nhom02_PhuongTienGiaoThong_BaiBao_SSRC_v3.docx` (bản `_danhdau` tô xanh lá các chỗ nhóm em sửa) và thư mục kết quả `KetQua_Nhom02_vong3.zip`. Chữ xanh dương của cô nhóm em đã giữ nguyên và chuyển về màu đen. Dưới đây nhóm em trả lời lần lượt từng điểm cô yêu cầu xác nhận.

---

## 1. Bảng III và mức lệch 4,7 điểm giữa hai công cụ (điểm quan trọng nhất)

**Cô nghi đúng, và kết luận chính của bài phải đổi.** Mức lệch 4,7 điểm không do pycocotools mà do cách nhóm em sinh dự đoán YOLO.

- Nhóm em sinh dự đoán YOLO bằng `predict()`. Hàm này mặc định chạy **NMS một nhãn**, tức mỗi hộp chỉ giữ một lớp. Faster R-CNN của TorchVision chạy NMS theo từng lớp, nên một vùng có thể giữ hộp của nhiều lớp. Còn `model.val()` của Ultralytics thì dùng NMS đa nhãn. Vì vậy so sánh trong bản v2 bị lệch về phía Faster R-CNN.
- Nhóm em kiểm tra cả ba điểm cô nêu:
  1. Tọa độ là pixel ảnh gốc dạng [x, y, w, h], lấy từ `predictions.json` của `val(save_json=True)`.
  2. `category_id` và `image_id` khớp tệp nhãn.
  3. `maxDets` = 300. Nhóm em tính mAP trực tiếp từ mảng precision vì `stats[0]` mặc định chỉ lấy 100 hộp.
- Làm đúng cách cô gợi ý (`val(save_json=True)` → pycocotools) cho các kết quả sau:

| Cách sinh dự đoán YOLOv8s-CW | Ultralytics | pycocotools |
|---|---|---|
| `val()`, NMS đa nhãn (mặc định của val) | 0,7984 | **0,7950** |
| `val()`, NMS một nhãn | 0,7670 | 0,7641 |
| `predict()`, NMS một nhãn (bản v2) | – | 0,7513 |
| `predict()`, NMS đa nhãn | – | 0,7813 |

  Trên cùng một tệp dự đoán, hai công cụ chỉ lệch khoảng 0,3 điểm, với cả YOLO (0,7984 so với 0,7950) lẫn Faster R-CNN (0,7945 so với 0,7916). Con số 0,7945 là kết quả chấm dự đoán của Faster R-CNN bằng hàm tính AP của Ultralytics. Nhóm em đồng ý với cô bỏ dòng này khỏi Bảng III vì nó không phải kết quả của `model.val()`. Nhóm em chỉ nhắc con số này ở Mục VI, điểm 3. Xem các tệp `1_pycocotools/multilabel_isolation.json` và `val_json_pycoco.json`.
- **Kết quả mới (Bảng IV, bootstrap 1.000 lần):**
  - Trong miền: YOLOv8s-CW 0,7950 so với Faster R-CNN 0,7916, hiệu số +0,34 điểm [−0,46; +1,29]. **Hai mô hình ngang nhau.** Ở bản v2, Faster R-CNN hơn 4,03 điểm.
  - VN Aerial: YOLOv8s-CW hơn ở cả mAP@0,5 (+7,13 điểm trên tập xác thực, +7,66 điểm trên tập kiểm thử) và mAP@0,5:0,95. Câu "Faster R-CNN khớp hộp chặt hơn ngoài miền" vì vậy không còn đúng và đã được bỏ.
- Nhóm em đã viết lại Tóm tắt/Abstract, đoạn Bảng IV, Thảo luận (điểm thứ nhất và điểm 3, điểm 5) và Kết luận theo số liệu mới. Điểm 3 nay nói rõ: kết quả so sánh phụ thuộc cách hậu xử lý. Nếu dùng `predict()` mặc định khi triển khai thì độ chính xác thực tế gần 0,7513. Phần chênh 1,3 điểm còn lại giữa `predict()` và `val()` (cùng NMS một nhãn) nhóm em chưa tách được nguyên nhân và đã ghi rõ điều này trong bài.
- Số liệu đối chứng cls_pw theo pycocotools cũng đổi:
  - Chênh lệch theo từng hạt giống: −0,04 / +0,17 / +0,01 điểm (bản v2: −0,04 / +0,29 / +0,13).
  - Trung bình cls_pw = 0,5 so với cls_pw = 0: +0,05 điểm.
  - Riêng việc tinh chỉnh thêm: +0,73 điểm (trung bình ba hạt giống; bản v2 ghi 1,12 điểm là của một hạt giống).
  - Đóng góp 4 sửa "dưới 0,3 điểm" thành "dưới 0,2 điểm".

## 2. Ký hiệu ±

Ký hiệu ± đúng là **độ lệch chuẩn mẫu qua ba hạt giống** (`np.std(..., ddof=1)`, script 84). Kiểm tra lại: cls_pw = 0,5 cho 0,8031; 0,7973; 0,8020, nên SD = 0,0031 (sai số chuẩn sẽ là 0,0018). Chữ cô sửa giữ nguyên.

## 3. Lệch 395 hộp giữa ma trận nhầm lẫn và tập kiểm thử

Lỗi nằm ở script phân tích lỗi của nhóm em (`31_error_analysis.py`). Tập kiểm thử thật có 49.446 hộp (39.807 / 8.031 / 756 / 852), khớp tệp nhãn và tệp GT của pycocotools. Script có hai lỗi:

1. Một lệnh `break` đặt sai chỗ làm thoát cả vòng đếm FN sau ví dụ minh họa đầu tiên, nên mất **97 hộp bỏ sót** (84 xe máy, 13 ô tô).
2. Hộp thật bị nhầm lớp được ghi **hai lần**, một lần ở ô lệch chéo và một lần ở hàng "Nền".

Vì vậy tổng ra 49.349 + 492 = 49.841. Nhóm em đã sửa để mỗi hộp thật chỉ nằm ở đúng một ô, rồi chạy lại và vẽ lại Hình 3. Tổng các cột nay đúng bằng 49.446. Thay đổi trong bài:

- Bảng VII: FN xe máy 10.290 (cũ 10.206), recall 0,7415. FN ô tô 790 (cũ 777), recall 0,9016. TP và FP không đổi.
- Recall xe máy theo kích thước: 49,5% / 79,2% / 95,4%.
- Số trường hợp nhầm lớp: ô tô–xe buýt 145, ô tô–xe tải 106, xe buýt–xe tải 65, xe máy 47.

Nhóm em cũng thêm một câu nói rõ tổng các cột bằng 49.446. Kết quả trước khi sửa vẫn giữ trong `4_phan_tich_loi/` để cô đối chiếu.

## 4. Khoảng tin cậy [+1,50; +5,17] trên VN Aerial xác thực

Đây **không phải lỗi chép số**. Nhóm em chạy lại bootstrap trên đúng bộ dự đoán cũ (NMS một nhãn) và ra cận trên 0,05165, trong khi hiệu số trên tập kiểm thử là 0,05171. Hai số trùng nhau khi làm tròn đến 0,01 điểm (tệp `2_bootstrap/bootstrap_ci_du_doan_1nhan_BAN_CU.json`). Với dự đoán mới, khoảng này là [+5,49; +8,85]. Mục V nay ghi rõ phương pháp phân vị (phân vị 2,5% và 97,5% của 1.000 hiệu số).

## 5. Lượt nhãn giả 0,7812

- Lượt 0,7812 là `runs/detect/train_v8s_v5plus_960`. Tên thư mục gây nhầm: lượt này thật ra huấn luyện trên `data_v5.yaml`, tức 2.833 ảnh nhãn thật, với các thiết lập COCO, ảnh 960, batch 16, lr0 0,005, seed 42. Đây **không phải** lượt "V5 honest" (ảnh 640, lr 0,01, mAP 0,665) trong báo cáo đồ án.
- Lượt có nhãn giả là `train_v8s_v6_pseudo`, dùng 2.833 ảnh thật cộng **402** ảnh nhãn giả (manifest: 526 khung lấy mẫu, giữ 402 khung, 1.079 hộp, ngưỡng 0,75). Nhóm em xác nhận con số 402 qua thư mục lượt chạy. Ultralytics lưu ảnh lô huấn luyện `train_batch6090` ở đầu giai đoạn tắt mosaic, tức tại bước (40 − 10) × số batch mỗi epoch. Từ đó suy ra có 203 batch/epoch, tương ứng khoảng 3.235 ảnh = 2.833 + 402 với batch 16.
- Con số 4.396 ở bản trước là số ảnh `hcm_*` của thư mục `merged_v6` **dựng lại ngày 07/11, sau lượt chạy**. Chưa có lượt nào huấn luyện trên thư mục này, nên con số 4.396 là sai.
- Nhãn giả được sinh bằng chính lượt 0,7812. Lượt này có 26/40 epoch vì **dừng sớm** (patience 15, điểm lưu tốt nhất ở epoch 11), không phải do bị ngắt. Hai lượt cùng quy tắc dừng sớm nhưng số epoch thực chạy khác nhau. Bài đã ghi rõ điều này và giữ nguyên câu viết thận trọng.
- Nhóm em sẽ cập nhật báo cáo đồ án cho khớp (mục V6 pseudo).

## 6. Tài liệu [3], [4], [5]

Nhóm em đã tra Crossref theo DOI và đọc bài gốc:

- **[3]** Trinh T., Nguyen K., *Can Tho Univ. J. Sci.* 14(3):45–52, 2022: đúng. Nhận định trong mục Khoảng trống khớp bài gốc (Mục 3.2): *"both training and test set contain images at daytime and night but … were extracted from different videos which have completely different scenes. Samples in validation set and training set were extracted from same videos but were different frames."* Bài gốc dùng 6 video và so sánh YOLOX, YOLOF, YOLACT, có nhắc RetinaNet.
- **[4]** Dinh D.-L., Nguyen H.-N., Thai H.-T., Le K.-H., *J. Adv. Transp.* 2021, Art. 5551976: đúng. Tóm tắt bài gốc nêu bộ VDD của Việt Nam và hệ thống đếm trên thiết bị biên.
- **[5]** Vuong X. C., Mou R.-F., Vu T. T., *Transport Problems* 17(4):17–26, 2022: đúng. Bài gốc dùng YOLOv4 và SORT với camera cố định ở Hà Nội.

## 7. Thành viên Vũ Thăng Tiến

*[Nhóm em điền: xác nhận bạn Tiến có/không đứng tên bài báo, và lý do]*

---

**Thư mục kết quả gửi kèm** (`KetQua_Nhom02_vong3.zip`):

| Thư mục | Nội dung |
|---|---|
| `1_pycocotools/` | `unified_eval.json` (bản mới) và bản cũ dùng `predict()`, kèm `multilabel_isolation.json`, `val_json_pycoco.json`, `yolo_val_preds_log.json`, `ultralytics_metric_*.json` |
| `2_bootstrap/` | `bootstrap_ci.json`, `bootstrap_clspw_seeds.json` (B = 1.000, phân vị) và bản chạy lại trên dự đoán cũ |
| `3_doi_chung_cls_pw/` | `results.csv` và `args.yaml` của sáu lượt đối chứng, của cơ sở V5+ và của YOLOv8s-CW |
| `4_phan_tich_loi/` | Kết quả phân tích lỗi trước và sau khi sửa, kèm Hình 3 mới |
| `5_toc_do/` | Số đo tốc độ |
| `6_nhan_gia/` | `results.csv` và `args.yaml` của hai lượt nhãn giả, kèm manifest 402 ảnh |
| `7_faster_rcnn/` | `config.json`, `env.json` (Tesla T4) và nhật ký 8 epoch |
| `scripts/` | Các script đánh giá đã dùng |

Nhóm em xin cảm ơn cô.
