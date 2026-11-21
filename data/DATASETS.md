# Các bộ dữ liệu trong repo

Mọi bộ dùng nhãn YOLO 4 lớp: `0 motorcycle`, `1 car`, `2 bus`, `3 truck`. Mỗi bộ có `images/{train,val,test}` và
`labels/{train,val,test}`. Các file YAML trong `data/` đang ghi đường dẫn tuyệt đối của máy gốc
(`/home/xuand/vn_traffic_ai/...`); khi clone về máy khác, sửa dòng `path:` cho đúng thư mục của bạn.

| Thư mục | YAML | Train / val / test | Dùng cho |
|---|---|---|---|
| `merged_v5plus` | `data_v5plus.yaml` | 7.237 / 1.466 / 1.500 | Bộ V5+ của bài báo SSRC và báo cáo (YOLOv8s-CW `r_clspw_v2`) |
| `merged_v5fix` | `data_v5fix.yaml` | 7.179 / 1.000 / 1.500 | V5 + v10 đã sửa + UA-DETRAC (mô hình D1, C, D2) |
| `merged_v5fix_mio` | `data_v5fix_mio.yaml` | 10.179 / 1.000 / 1.500 | V5fix + 3.000 ảnh MIO-TCD (mô hình M1) |
| `merged_v5fix_mio_mask` | `data_v5fix_mio_mask.yaml` | 10.179 / 1.000 / 1.500 | Như V5fix+MIO nhưng tô xám 1.870 xe thiếu nhãn ở 4 nguồn Việt Nam |
| `merged_v5fix_mio_ctmask` | `data_v5fix_mio_ctmask.yaml` | 10.179 / 1.000 / 1.500 | Như V5fix+MIO nhưng tô xám 35% bên trái ảnh Cần Thơ |
| `ext_benchmark` | `data_ext_benchmark.yaml` | 250 khung (chỉ dùng đánh giá) | Tập đánh giá ngoài từ video thực tế; **nhãn nháp** do yolov8x sinh, chưa sửa tay |
| `vn_vehicles_aerial` | `data_vn_aerial.yaml` | – / 486 / 149 | Đánh giá ngoài miền (góc nhìn trên cao) |

Val và test của các bộ V5fix giống hệt nhau (val = SOICT `cam_07`, test = SOICT `cam_09`, `cam_10`), nên so sánh
trực tiếp được giữa các mô hình. Test của V5fix trùng test của V5+. Không dùng `ext_benchmark` để huấn luyện.

Cách dựng lại từng bộ: `scripts/data_prep/build_v5plus_merged.py`, `build_v5plus_clean.py`, `build_v5fix.py`,
`build_v5fix_mio.py`, `build_v5fix_mio_mask.py`, `build_v5fix_mio_ctmask.py`, `build_ext_benchmark.py`.
Thống kê và danh sách ảnh: `reports/v5fix/manifest.json`, `reports/v5fix_mio/*.json`, `reports/v5plus_clean/`.

## Nguồn và giấy phép

Các bộ trên là sản phẩm phái sinh, giữ nguyên điều khoản của từng nguồn. Khi dùng lại, hãy trích dẫn nguồn gốc.

| Nguồn | Có trong | Giấy phép / điều khoản |
|---|---|---|
| SOICT 2024, SOICT Night, Vehicle Vietnam-CanTho, Vietnamese Vehicle v3, Vietnam Vehicles v10 (Roboflow Universe) | mọi bộ V5+ / V5fix | CC BY 4.0 (ghi trong README của từng bộ trên Roboflow) |
| MIO-TCD Localization (Miovision) | `merged_v5fix_mio*` (ảnh tiền tố `mio_`) | CC BY-NC-SA 4.0: chỉ dùng phi thương mại, ghi nguồn, chia sẻ phái sinh theo cùng giấy phép. Do đó các bộ `merged_v5fix_mio*` cũng chỉ dùng phi thương mại |
| UA-DETRAC (Wen và cộng sự, CVIU 2020) | `merged_v5fix*` (ảnh tiền tố `detrac_`) | Trang dự án yêu cầu trích dẫn bài báo khi sử dụng; chưa tìm thấy văn bản giấy phép chính thức. Chỉ dùng cho nghiên cứu |
| VN Aerial (Ultralytics Platform) | `vn_vehicles_aerial` | Theo điều khoản của nguồn |
| Khung hình cắt từ video quay màn hình (YouTube, Windy, báo điện tử) | `ext_benchmark` | Thuộc bản quyền của chủ sở hữu video gốc; nhóm chỉ dùng cho mục đích nghiên cứu, đánh giá phi thương mại |

Ảnh trong `ext_benchmark` có tiền tố `india_` (2 clip camera trên cao ở Ấn Độ) và `vncctv_` (3 clip camera giao
thông Việt Nam). Danh sách clip nguồn và số khung: `ext_benchmark/manifest.json`.
