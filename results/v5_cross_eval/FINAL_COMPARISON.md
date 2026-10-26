# Final comparison — all models × all test sets

Mỗi cell là `mAP50 / mAP50-95`.

## Overall

| Model | V5 test (VN urban) | vn_cctv test (OOD CCTV) | DETRAC test (cross-domain) |
|---|---|---|---|
| v8s_coco_zeroshot | **0.557** / 0.279 | — | — |
| v5_honest | **0.665** / 0.340 | **0.181** / 0.121 | **0.548** / 0.361 |
| v5_plus_960 | **0.782** / 0.413 | **0.215** / 0.146 | **0.637** / 0.425 |
| v6_pseudo | **0.761** / 0.411 | **0.135** / 0.100 | **0.596** / 0.408 |

## Per-class mAP50 trên V5 test

| Model | motorcycle | car | bus | truck |
|---|---|---|---|---|
| v8s_coco_zeroshot | 0.363 | 0.859 | 0.598 | 0.407 |
| v5_honest | 0.621 | 0.813 | 0.694 | 0.530 |
| v5_plus_960 | 0.841 | 0.884 | 0.716 | 0.687 |
| v6_pseudo | 0.817 | 0.850 | 0.743 | 0.634 |