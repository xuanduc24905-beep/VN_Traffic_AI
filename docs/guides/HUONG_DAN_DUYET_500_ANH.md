# Hướng dẫn duyệt 500 ảnh pseudo-label V6

> Tài liệu dành cho **thành viên trong nhóm** giúp duyệt tay 500 ảnh.
> Không cần hiểu code, chỉ cần xem ảnh và đếm số lượng.

---

## 1. Mục tiêu

Mình đã dùng model AI tự động vẽ **bbox** (khung chữ nhật) quanh các xe trong 500 ảnh camera giao thông TP.HCM. Nhiệm vụ của bạn là **xem từng ảnh** và trả lời: model vẽ **đúng hay sai**?

Kết quả sẽ giúp nhóm biết model có đủ chính xác để dùng dataset này train tiếp không.

**Thời gian ước tính**: ~2–2.5 giờ. Có thể chia thành 2–3 buổi (mỗi buổi 1 giờ).

---

## 2. Chuẩn bị (1 lần duy nhất)

### Bước 1: Mở folder chứa ảnh

Đường dẫn: `data/pseudo_v6/review/` (trong thư mục project)

Mở bằng **1 trong 2 cách**:

**Cách A — VSCode** (khuyến nghị, xem ảnh nhanh):
- Mở VSCode
- File → Open Folder → chọn `data/pseudo_v6/review/`
- Click vào từng file `0001__...jpg`, `0002__...jpg`, ... → ảnh hiện ở panel bên phải

**Cách B — File Explorer (Nautilus, Files)**:
- Mở thư mục `data/pseudo_v6/review/` bằng file manager
- Bật chế độ xem icon lớn (ctrl + + hoặc view → large icons)
- Double-click từng ảnh để xem phóng to

### Bước 2: Mở 2 file để ghi kết quả

Trong cùng folder `data/pseudo_v6/review/`:

- **`bad_samples.txt`** — ghi tên ảnh SAI (sẽ giải thích ở Bước 4)
- **`RESULT_FORM.md`** — ghi tổng kết cuối cùng

Mở bằng Notepad / VSCode / bất kỳ text editor nào.

---

## 3. Ý nghĩa màu bbox

Trên mỗi ảnh có khung chữ nhật màu sắc khác nhau:

| Màu | Class | Nghĩa |
|---|---|---|
| 🔴 Đỏ | motorcycle | Xe máy |
| 🟢 Xanh lá | car | Xe con/ô tô 4 chỗ |
| 🟠 Cam | bus | Xe buýt |
| 🔵 Xanh dương | truck | Xe tải |

Tên class cũng được viết chữ trắng trên nền màu phía trên mỗi khung.

---

## 4. Cách phân loại từng ảnh (QUAN TRỌNG)

Với **mỗi ảnh**, nhìn tổng thể và chọn **1 trong 3 nhóm**:

### ✅ CORRECT — "Đúng hết"

- Tất cả khung bbox đúng loại xe (motorbike/car/bus/truck)
- Khung ôm sát xe, không lệch
- Không bỏ sót xe lớn (xe to mà không có khung bao)

**Ví dụ CORRECT**:
- Ảnh có 5 xe, model vẽ đủ 5 khung, đúng hết loại, khung ôm sát → ✅

### ⚠️ PARTIAL — "Đa số đúng"

- Có **1–2 khung sai loại** (ví dụ motorbike bị đánh thành car) nhưng còn lại đúng
- Hoặc model **bỏ sót vài xe nhỏ** nhưng xe lớn đánh đúng
- Hoặc khung **lệch nhẹ** nhưng vẫn bao quanh xe

**Ví dụ PARTIAL**:
- Ảnh có 10 xe, model chỉ vẽ 3 khung nhưng 3 cái đó đúng → PARTIAL
- Ảnh có 5 khung, 1 cái motorbike bị tag thành car → PARTIAL

### ❌ WRONG — "Sai nhiều quá"

- **≥3 khung sai loại** hoặc **đa số khung vẽ nhầm vào vật không phải xe** (cây, đèn giao thông, biển báo)
- Hoặc **class sai hoàn toàn** (ví dụ cả ảnh chỉ có xe máy mà toàn bị tag car)
- Hoặc khung **lệch hẳn ra ngoài đường**

**Ví dụ WRONG**:
- Ảnh ban đêm, model vẽ khung quanh đèn giao thông và tag là motorbike → WRONG
- Ảnh có 4 khung, 3 cái bị lệch vị trí → WRONG

---

## 5. Quy tắc phân biệt quan trọng

### Miss xe vs Sai xe

| Tình huống | Phân loại |
|---|---|
| Model **không vẽ** khung cho vài xe | PARTIAL (chấp nhận được) |
| Model **vẽ khung** nhưng **tag sai class** | PARTIAL (nếu ít) hoặc WRONG (nếu nhiều) |
| Model vẽ khung vào **vật không phải xe** | **WRONG** (nghiêm trọng hơn) |

**Nguyên tắc**: Model bỏ sót thì không nghiêm trọng bằng model đánh sai. Vì train sau này sẽ có ảnh khác bù lại cho xe bị miss, nhưng label sai thì model học sai luôn.

### Nếu không chắc

→ Phân vào **PARTIAL** (an toàn, không ảnh hưởng xấu).

---

## 6. Khi gặp ảnh WRONG — thao tác

1. Copy tên file ảnh (ví dụ: `0042__hcm_phuong-cat-lai_000026_20261105_111241.jpg`)
2. Mở file `bad_samples.txt` trong folder review
3. Dán tên ảnh vào 1 dòng mới + ghi chú ngắn **lý do** sau dấu `#`

**Ví dụ nội dung `bad_samples.txt`**:
```
0042__hcm_phuong-cat-lai_000026_20261105_111241.jpg   # tag cây xanh thành motorbike
0100__hcm_xa-tan-nhut_000133_20261105_112055.jpg      # 4 xe máy bị đánh thành car
0203__hcm_unknown_000456_20261105_113022.jpg          # khung lệch hẳn ra vỉa hè
```

**Chú ý**:
- **KHÔNG sửa khung** trên ảnh — mình không cần annotate lại, chỉ cần biết ảnh nào SAI để loại
- **KHÔNG xoá file** — chỉ cần ghi tên vào `bad_samples.txt`, script khác sẽ xoá tự động

---

## 7. Đếm kết quả

Trong lúc duyệt, bạn nên giữ 1 giấy ghi nháp hoặc file text có 3 cột:

```
CORRECT: |||| |||| |||| ...
PARTIAL: |||| |||
WRONG:   ||
```

Hoặc dùng counter online (tìm "tally counter" trên Google).

Mỗi ảnh xong → đánh 1 gạch vào đúng cột.

---

## 8. Sau khi duyệt xong 500 ảnh

Mở file `RESULT_FORM.md`, điền 3 con số vào bảng:

```markdown
| Phân loại | Số ảnh |
|---|---|
| CORRECT (tất cả box đúng) | ___ |
| PARTIAL (1-2 box sai, còn lại đúng) | ___ |
| WRONG (đa số box sai) | ___ |
```

**Kiểm tra**: CORRECT + PARTIAL + WRONG = **500** (nếu không đủ, kiểm tra lại).

Gửi file `RESULT_FORM.md` + `bad_samples.txt` cho nhóm trưởng (Đức).

---

## 9. Tips tăng tốc

- **Dùng phím mũi tên** trong VSCode để chuyển ảnh nhanh (←/→)
- **Không soi quá kỹ** — nhìn tổng thể 3-5 giây/ảnh là đủ, tin vào trực giác
- **Chia buổi** — ngồi lâu quá mắt mỏi, dễ nhầm. Nghỉ 10 phút sau mỗi 100 ảnh
- **Buổi đầu** sẽ chậm (30s/ảnh), về sau nhanh dần (10s/ảnh)

---

## 10. Nếu gặp vấn đề

- **Không mở được ảnh** → check lại đường dẫn `data/pseudo_v6/review/` có tồn tại không
- **Ảnh không có bbox** → chụp lại gửi nhóm, có thể do bug
- **Không chắc phân loại nào** → mặc định chọn PARTIAL (an toàn)
- **Vấn đề khác** → chụp màn hình gửi nhóm trưởng

---

## Tóm tắt 1 câu

> Mở folder, xem từng ảnh, đếm vào 3 cột (CORRECT / PARTIAL / WRONG), ảnh nào SAI ghi tên vào `bad_samples.txt`, cuối cùng điền tổng vào `RESULT_FORM.md`.

Cảm ơn bạn đã hỗ trợ 🙏
