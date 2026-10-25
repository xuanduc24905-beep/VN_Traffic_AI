# V5 Cross-eval — Final Summary

**Ngày:** 2026-10-26
**Test sets:**
- `v5_test`: 1500 frames từ 2 video UNSEEN (soict24_cam_09+10), VN urban, no leak
- `detrac_test` (subsample): 1139 frames từ 40 video DETRAC, highway TQ, car+bus only

## Bảng tổng hợp — V5 test (VN domain)

| Model | Training data | mAP50 | mAP50-95 | motor | car | bus | truck |
|---|---|---|---|---|---|---|---|
| yolov8s COCO (zero-shot) | COCO 80cls | 0.557 | 0.279 | 0.363 | **0.859** | 0.598 | 0.407 |
| Old DETRAC+Cantho merge | DETRAC 60k + Cantho 700 (leaky) | 0.616 | 0.309 | 0.652 | 0.851 | 0.493 | 0.469 |
| **V5 honest** | **V5 clean (2.8k, video-aware)** | **0.665** | **0.340** | 0.621 | 0.813 | **0.694** | **0.530** |

### Delta vs v8s COCO zero-shot (baseline)

| Model | Overall mAP50 | motor | car | bus | truck |
|---|---|---|---|---|---|
| **V5 honest** | **+0.108 (+19%)** | **+0.258 (+71%)** | -0.046 | **+0.096** | **+0.123** |
| Old DETRAC+Cantho | +0.059 (+11%) | +0.289 | -0.008 | -0.105 | +0.062 |

**Insight chính:**
- V5 honest cải tiến mạnh nhất ở **motorcycle (+71%)** — chính là class dominant của traffic VN mà COCO yếu.
- Car bị trade-off nhẹ (-0.046) nhưng gain motorcycle lớn hơn nhiều.
- DETRAC merge thực sự HURT bus (-0.10) — confirm van→truck mapping gây class confusion.

## V5 generalization — DETRAC test (cross-domain)

V5 train trên VN domain, test trên DETRAC Chinese highway (chỉ 2 class common car+bus):

| Metric | V5 on V5_test | V5 on DETRAC | Drop |
|---|---|---|---|
| car mAP50 | 0.813 | 0.616 | -0.197 |
| bus mAP50 | 0.694 | 0.480 | -0.214 |
| mean mAP50 | 0.754 | 0.548 | -0.206 |

→ V5 drop ~0.20 mAP50 khi cross-domain → **generalize được** (không phải overfit VN hoàn toàn), nhưng rõ ràng đặc trưng VN có vai trò.

## Story cho báo cáo (ready-to-use)

1. **Problem**: Model yolov8s pretrained COCO đạt mAP50 chỉ **0.56** trên VN traffic test, đặc biệt **motorcycle 0.36** — rất kém với class dominant của VN.

2. **Approach**: Finetune v8s trên dataset VN đã được **video-aware split** (no data leak, 2.8k frames thực sự đa dạng từ 11 video).

3. **Result**: V5 honest đạt mAP50 **0.665** (+19% so với COCO baseline), đặc biệt motorcycle **0.62** (+71%).

4. **Comparison với approach cũ**: Thử merge data DETRAC (60k Chinese highway) không giúp — DETRAC đã train đạt 0.62 nhưng vẫn **thua V5 honest** 0.665 và **hurt bus mAP** (0.49 vs 0.69) do class semantic mismatch (van→truck gây confusion).

5. **Takeaway**: Với traffic VN domain, **methodology (no leak) + quality data (video-diverse) > volume**. 2.8k frames VN sạch > 60k frames mixed domain.

## Files

- Scripts: [scripts/eval/](../../scripts/eval/)
  - `compare_models_on_v5test.py` — V5 vs old on V5 test
  - `eval_v8s_coco_zeroshot.py` — v8s zero-shot với class remap
  - `eval_v5_on_detrac.py` — V5 external eval trên DETRAC
- Raw JSON: `summary.json` ở mỗi subfolder của [results/v5_cross_eval/](.)
- Confusion matrices: `confusion_matrix_normalized.png` ở mỗi subfolder
