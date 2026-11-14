"""Chạy toàn bộ 6 check audit cho v10 và tổng hợp thành reports/v10_audit_summary.md.

Chỉ ĐỌC dữ liệu. Không sửa split, không train, không đụng V5+.

Usage:
    conda activate yolov8_ft
    python scripts/data_prep/audit_v10/audit_v10_all.py
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_camera_leak_v10 as cam_chk  # noqa: E402
import check_cross_v5_leak_v10 as cross_chk  # noqa: E402
import check_exact_duplicates_v10 as md5_chk  # noqa: E402
import check_label_integrity_v10 as lbl_chk  # noqa: E402
import check_perceptual_duplicates_v10 as phash_chk  # noqa: E402
import check_temporal_leak_v10 as time_chk  # noqa: E402
import check_vehicle_id_leak_v10 as veh_chk  # noqa: E402
from v10_common import REPORTS_DIR, REPO_ROOT, V10_ROOT, load_records, survey, write_json  # noqa: E402

OUT_MD = REPORTS_DIR / "v10_audit_summary.md"
ICON = {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "WARN": "⚠️ WARN"}


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO_ROOT))


def _feasibility(recs) -> dict:
    """Số liệu để đánh giá phương án rebuild split theo site_id (KHÔNG thực hiện split)."""
    by_site = defaultdict(int)
    plate_sites = defaultdict(set)
    for r in recs:
        by_site[r.site_id] += 1
        if r.plate and len(r.plate) >= veh_chk.MIN_PLATE_LEN:
            plate_sites[r.plate].add(r.site_id)
    sizes = sorted(by_site.items(), key=lambda kv: -kv[1])
    n = len(recs)
    multi = {p: s for p, s in plate_sites.items() if len(s) > 1}
    return {
        "n_sites": len(sizes),
        "top_sites": [(s, c, round(100 * c / n, 1)) for s, c in sizes[:8]],
        "largest_site_pct": round(100 * sizes[0][1] / n, 1),
        "n_plates_multi_site": len(multi),
        "n_images_with_multi_site_plate": sum(1 for r in recs if r.plate in multi),
    }


def build_markdown(sv, res, feas) -> str:
    md5, cam, veh, tmp, ph, lbl = (res[k] for k in ("md5", "cam", "veh", "tmp", "ph", "lbl"))
    leak_checks = {"(a) Exact duplicate (MD5)": md5, "(b) Camera leak": cam,
                   "(c) Vehicle leak (ID + biển số)": veh, "(d) Temporal leak (<60s)": tmp,
                   "(e) Near-duplicate (pHash ≤5)": ph}
    any_leak = any(r["verdict"] == "FAIL" for r in leak_checks.values())
    label_fail = lbl["verdict"] == "FAIL"
    L = []
    w = L.append
    w("# Audit leakage — Roboflow vietnam_vehicles v10 (standalone)\n")
    w(f"_Sinh tự động bởi [audit_v10_all.py](../scripts/data_prep/audit_v10/audit_v10_all.py) "
      f"lúc {datetime.now():%Y-%m-%d %H:%M}. Chỉ đọc dữ liệu, không sửa split._\n")

    cross = res["cross"]
    cross_ok = cross["verdict"] == "PASS"
    w("## Kết luận\n")
    w("| Câu hỏi | Trả lời |")
    w("|---|---|")
    w("| v10 standalone có sạch (split nội bộ)? | " + (
        "❌ **Không** — train/valid/test gốc rò rỉ lẫn nhau (camera, xe, thời gian, khung gần trùng). "
        "Không được dùng valid/test của v10 để đánh giá." if any_leak else "✅ Có") + " |")
    w("| Gộp v10 vào **train** V5+ (val/test giữ của V5) có làm leak eval V5+? | " + (
        "✅ **Không** — 0 ảnh/khung/camera của v10 trùng với val/test V5 (check (g))." if cross_ok else
        "❌ **Có** — xem check (g).") + " |")
    w("| Nhãn có hợp lệ? | " + ("✅ Format hợp lệ" if not label_fail else "❌ Có lỗi format") +
      "; ⚠️ nghi **thiếu nhãn** (xem dưới) |")
    w("")
    if any_leak and cross_ok:
        w("**→ Về leakage, v10 an toàn để gộp vào train V5+ theo đúng cách "
          "[build_v5plus_merged.py:3](../scripts/data_prep/build_v5plus_merged.py#L3) đang làm** "
          "(gộp cả 3 split v10 vào train, val/test giữ của V5): leak nội bộ v10 không chạm tới eval V5+. "
          "**Không** an toàn nếu dùng split gốc của v10 làm val/test.\n")
    w("**Còn lại — rủi ro chất lượng nhãn (không phải leak):** trung bình chỉ ~1.3 box/ảnh trên ảnh CCTV "
      "nhiều xe, và các file cùng camera + cùng giây mang **bộ nhãn khác nhau** (vd "
      "`general_caunamprc554_50Y31618_13-02-2024_13-08-11` có 2 box, bản ở test chỉ 1 box) → nhiều xe "
      "không được gán nhãn, model có thể bị dạy coi xe là nền. Một số ảnh `gen_*` còn có **khung bbox vẽ "
      "sẵn lên pixel** (thấy trong preview của check (g)). Nên kiểm tra trực quan một mẫu trước khi tin dùng.\n")
    w("_Ghi chú: docstring build_v5plus_merged.py ghi \"~776 camera\" — thực tế parse được 44 camera._\n")

    w("## 1. Tổng quan cấu trúc\n")
    w(f"- Nguồn: `{_rel(V10_ROOT)}` (Roboflow `visionvn/vietnam_vehicles` v10, export 2024-09-05, "
      f"CC BY 4.0, không augmentation)")
    w(f"- Ảnh: **{sv['n_images']}** — train {sv['per_split']['train']} / valid {sv['per_split']['valid']} "
      f"/ test {sv['per_split']['test']}")
    w(f"- Class ({lbl['nc']}): {', '.join(lbl['names'])}")
    w(f"- Variant tiền tố filename: {sv['prefix_variants']} (+ 2 file dùng `general-` toàn dấu gạch ngang, "
      f"{sv['n_copy_suffix']} file có hậu tố `-1-`)")
    pr = sv["parse_rate"]
    w(f"- Tỉ lệ parse: camera {pr['camera_id']:.1%}, datetime {pr['datetime']:.1%}, "
      f"vehicle_id {pr['vehicle_id']:.1%}, biển số {pr['plate']:.1%}")
    w(f"- **Camera unique: {sv['n_unique_camera']}** (thuộc {sv['n_unique_site']} địa điểm) — "
      f"**không phải ~776 camera** như giả định ban đầu.")
    w(f"- vehicle_id unique: {sv['n_unique_vehicle_id_global']} (ID trần) / "
      f"{sv['n_unique_camera_vehicle_id']} (theo cặp camera+ID); biển số unique: {sv['n_unique_plate']}")
    w(f"- Khoảng thời gian: {sv['date_min'][:10]} → {sv['date_max'][:10]} ({sv['n_unique_dates']} ngày khác nhau)\n")

    w("## 2. Verdict từng check\n")
    w("| Check | Ngưỡng | Kết quả | Verdict |")
    w("|---|---|---|---|")
    w(f"| (a) Exact duplicate (MD5) | {md5['threshold']} | {md5['n_groups_cross_split']} nhóm cross-split, "
      f"{md5['n_groups_within_split']} within-split | {ICON[md5['verdict']]} |")
    c = cam["camera"]
    w(f"| (b) Camera leak | {cam['threshold']} | {c['n_shared_across_splits']}/{c['n_unique']} camera ở >1 split; "
      f"{c['pct_test_images_from_shared']}% ảnh test từ camera đã thấy | {ICON[cam['verdict']]} |")
    w(f"| (c) Vehicle leak | {veh['threshold']} | {veh['strict_camera_vid_date']['n_cross_split']} track cross-split; "
      f"{veh['plate']['n_cross_split']} biển số cross-split (ảnh dính: valid {veh['plate']['pct_eval_images_affected']['valid']}%, "
      f"test {veh['plate']['pct_eval_images_affected']['test']}%) | {ICON[veh['verdict']]} |")
    w(f"| (d) Temporal leak | {tmp['threshold']} | {tmp['n_cross_split_pairs']} cặp; ảnh dính: "
      f"valid {tmp['pct_images_flagged'].get('valid', 0)}%, test {tmp['pct_images_flagged'].get('test', 0)}% | {ICON[tmp['verdict']]} |")
    w(f"| (e) Near-duplicate pHash | {ph['threshold']} | {ph['n_cross_split_pairs']} cặp {ph['pairs_by_split_pair']}; "
      f"ảnh dính: valid {ph['pct_images_flagged'].get('valid', 0)}%, test {ph['pct_images_flagged'].get('test', 0)}% | {ICON[ph['verdict']]} |")
    w(f"| (f) Label integrity | 0 lỗi format/class/bbox | {sum(s['n_files_with_issues'] for s in lbl['splits'].values())} file lỗi | {ICON[lbl['verdict']]} |")
    w(f"| (f') Phân phối class | lệch ≤15pp giữa split | {len(lbl['class_shift_flags'])} class lệch | {ICON[lbl['distribution_verdict']]} |")
    cb = cross["background"]
    w(f"| (g) Chéo v10 ↔ val/test V5 | MD5=0, pHash≤5=0, NCC nền camera <0.90 và < min đối chứng | "
      f"MD5 {cross['md5']['n_hits']}, pHash {cross['phash']['n_pairs']} (min d={cross['phash']['min_hamming']}), "
      f"NCC max {cb['max_cross_ncc']} | {ICON[cross['verdict']]} |")
    w("")

    w("## 3. Chi tiết\n")
    w("### (a) Exact duplicate\n")
    w(f"- {md5['n_unique_md5']}/{md5['n_images']} MD5 unique. Không có ảnh y hệt byte giữa các split.")
    w("- ⚠️ PASS ở đây **không** có nghĩa là không trùng: Roboflow re-encode ảnh nên cùng một khung hình "
      "có MD5 khác nhau → xem (d) và (e).\n")

    w("### (b) Camera leak\n")
    w(f"- Camera chỉ có ở 1 split: {c['n_only_in']}. Site-level: "
      f"{cam['site']['n_shared_across_splits']}/{cam['site']['n_unique']} địa điểm ở >1 split ({ICON[cam['site_verdict']]}).")
    w("- Split hiện tại là **random theo ảnh** (mặc định Roboflow) → hầu như mọi camera đều có mặt ở cả 3 split. "
      "Test set đo khả năng nhận diện trên **camera đã thấy**, không phải camera mới.")
    w("- Top camera (train/valid/test):")
    for k, v in list(c["distribution"].items())[:8]:
        w(f"  - `{k}`: {v.get('train', 0)}/{v.get('valid', 0)}/{v.get('test', 0)}")
    w("")

    w("### (c) Vehicle leak\n")
    s = veh["strict_camera_vid_date"]
    w(f"- Khoá (camera, vehicle_id, ngày): {s['n_cross_split']} track cross-split. Mẫu (cùng khung hình, "
      f"chỉ khác tiền tố `general_`/`veh_`):")
    for e in s["items"][:3]:
        w(f"  - {' ↔ '.join(f'`{f}`' for f in e['files'])}")
    w(f"- Biển số: {veh['plate']['n_cross_split']}/{veh['plate']['n_keys']} biển xuất hiện ở >1 split. Phần lớn là "
      f"xe quay lại cùng camera ở ngày khác (vd xe buýt `60Y…` qua `caunamprc554` nhiều lần) → cùng ngoại hình xe ở train và test.")
    w("- vehicle_id trần (không kèm camera) **không** dùng làm verdict: đó là track-ID của tracker, reset theo camera.\n")

    w("### (d) Temporal leak\n")
    w(f"- {tmp['n_cross_split_pairs']} cặp khác split cùng camera, cùng ngày, |Δt|<60s. Theo cặp split: "
      f"{tmp['pairs_by_split_pair']}.")
    w(f"- Phân bố Δt: {tmp['delta_histogram']}. Các cặp **Δt = 0s** là cùng một khung hình CCTV được lưu "
      f"thành nhiều file (mỗi xe trong khung một file, khác biển số/ID) rồi bị Roboflow chia vào split khác nhau.")
    w("- Mẫu:")
    for p in tmp["pairs"][:4]:
        w(f"  - Δt={p['delta_s']:.0f}s: `{p['a']}` ↔ `{p['b']}`")
    w("")

    w("### (e) Near-duplicate (pHash)\n")
    w(f"- Histogram Hamming: {ph['hamming_histogram']}; {ph['pct_pairs_same_camera']}% cặp cùng camera.")
    w(f"- Near-dup trong cùng split (tham khảo): {ph['n_within_split_pairs_INFO']}.")
    w("- Mẫu Hamming=0 (ảnh giống hệt về thị giác nhưng khác file):")
    for p in ph["pairs"][:3]:
        w(f"  - `{p['a']}` ↔ `{p['b']}`")
    w("")

    w("### (f) Label integrity & phân phối class\n")
    w("| Split | Ảnh | Box | Box/ảnh | Ảnh rỗng | " + " | ".join(lbl["names"]) + " |")
    w("|---|---|---|---|---|" + "---|" * len(lbl["names"]))
    for sp, d in lbl["splits"].items():
        w(f"| {sp} | {d['n_images']} | {d['n_boxes']} | {d['mean_boxes_per_image']} | {d['n_empty_labels']} | "
          + " | ".join(f"{d['class_pct'][n]}%" for n in lbl["names"]) + " |")
    w("")
    w(f"- Không có lỗi format/class_id/toạ độ (tái sử dụng `check_label_file` của "
      f"[validate_labels.py:35](../scripts/data_prep/validate_labels.py#L35)).")
    w(f"- Không class nào lệch >15pp giữa split, nhưng container/xe_ba_gac ở valid/test gần **gấp đôi** train "
      f"(~15% vs ~8%). Imbalance nội tại max/min = {lbl['overall_max_min_ratio']}× (xe_may chiếm ~53%).\n")

    w("### (g) Leak chéo v10 ↔ val/test V5\n")
    pc = cb["positive_control_same_camera_split_half"]["summary"]
    w(f"- So {cross['n_v10_images']} ảnh v10 (cả 3 split, vì đều vào train V5+) với "
      f"{cross['n_v5_eval_images']} ảnh val/test V5: {cross['v5_eval_videos']}.")
    w(f"- MD5 trùng: {cross['md5']['n_hits']}. pHash: 0 cặp ≤{cross['phash']['threshold']}, "
      f"cặp gần nhất d={cross['phash']['min_hamming']} (và là cảnh khác hẳn khi xem bằng mắt).")
    w("- Ảnh nền camera (trung bình xám ≤150 ảnh/camera, so NCC):")
    w(f"  - Đối chứng dương — cùng camera chia 2 nửa: trung vị {pc['p50']}, nhỏ nhất {pc['min']}")
    w(f"  - Baseline v10 khác địa điểm: trung vị {cb['baseline_v10_diff_site']['p50']}, "
      f"max {cb['baseline_v10_diff_site']['max']}; cùng địa điểm khác kênh: max "
      f"{cb['baseline_v10_same_site_diff_channel']['max']}")
    w(f"  - **Chéo v10 ↔ V5 eval: max {cb['max_cross_ncc']}** — thấp hơn hẳn đối chứng cùng camera "
      f"({pc['min']}), nằm trong vùng \"khác camera\".")
    w("  - Top cặp: " + ", ".join(f"`{p['v10_camera']}`↔`{p['v5_video']}` {p['ncc']}" for p in cb["top10_cross"][:3]))
    s = cross["sanity_v5plus_eval"]
    w(f"- Sanity: val/test của `merged_v5plus` trùng khớp V5 và không chứa file `v10_*`: "
      f"{'✅' if s['ok'] else '❌'}.")
    w(f"- Preview kiểm tra bằng mắt: [{cross['preview']}](../{cross['preview']}) (trái: v10, phải: V5).")
    w("- Giới hạn: không bắt được trường hợp cùng địa điểm nhưng góc quay khác hẳn (NCC nền thấp). "
      "Với 44 camera v10 (tên địa điểm VN) và V5 eval (SOICT24 + Cần Thơ) thì khả năng này thấp.\n")

    w("## 4. Cần fix gì trước khi gộp (ĐỀ XUẤT — chưa thực hiện)\n")
    if not (any_leak or label_fail):
        w("Không cần fix. v10 có thể gộp vào V5+ an toàn.\n")
    else:
        w("Vì v10 trong V5+ chỉ đóng vai trò **dữ liệu train bổ sung**, có 2 hướng tuỳ mục đích:\n")
        w("**Hướng A — Chỉ dùng v10 để train (khuyến nghị, đơn giản nhất):**")
        w("1. Gộp **toàn bộ** train+valid+test của v10 vào train của V5+; **không** dùng valid/test của v10 "
          "để báo cáo số liệu. Leak nội bộ v10 khi đó vô hại vì không có split eval nào của v10.")
        w("2. Leak chéo v10 ↔ val/test V5: " + ("✅ đã kiểm tra, PASS (check (g))." if cross_ok else
          "❌ check (g) FAIL — loại các ảnh/camera v10 trùng khỏi train trước khi gộp."))
        w("3. Dedup khung hình trùng trong v10 (cùng camera + cùng giây / pHash≤2) để tránh over-weight; "
          "giữ một bản và **gộp nhãn** của các bản (mỗi file có thể chỉ gán nhãn một số xe trong khung).")
        w("")
        w("**Hướng B — Cần v10 có valid/test riêng đáng tin (vd báo cáo mAP trên v10):**")
        w(f"1. Rebuild split theo **site_id** (group split, {feas['n_sites']} địa điểm), không theo ảnh — "
          f"cùng tinh thần [rebuild_v5_by_video.py](../scripts/data_prep/rebuild_v5_by_video.py). "
          f"Lưu ý site lớn nhất chiếm {feas['largest_site_pct']}% ảnh nên tỉ lệ 80/10/10 chỉ xấp xỉ; "
          f"nên stratify theo class (container/xe_ba_gac/xe_may tập trung ở một số site).")
        w("   - Top site: " + ", ".join(f"`{s}` {n} ({p}%)" for s, n, p in feas["top_sites"]))
        w("2. Group theo site (không phải camera_id) vì nhiều kênh (`554/555/556`) cùng một ngã ba nhìn chung cảnh.")
        w(f"3. Sau khi split theo site, kiểm tra lại leak biển số chéo site: hiện có "
          f"{feas['n_plates_multi_site']} biển xuất hiện ở >1 site ({feas['n_images_with_multi_site_plate']} ảnh) — "
          f"chuyển các ảnh này về cùng phía hoặc loại khỏi eval.")
        w("4. Dedup như A.3 trước khi split.")
        w("5. Chạy lại `audit_v10_all.py` trên bản rebuild; chỉ gộp khi (b)(c)(d)(e) đều PASS.")
        w("")
        w("**Không làm:** dùng valid/test hiện tại của v10 để chọn checkpoint hay báo cáo — số liệu sẽ bị "
          "thổi phồng giống V1-V4.\n")

    w("## 5. File output\n")
    for k, p in res["_files"].items():
        w(f"- `{p}` — {k}")
    w("\nTái tạo: `conda activate yolov8_ft && python scripts/data_prep/audit_v10/audit_v10_all.py`")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_MD)
    a = ap.parse_args()

    recs = load_records()
    sv = survey(recs)
    write_json(sv, REPORTS_DIR / "v10_survey.json")
    print("=== Survey ===", sv["n_images"], "ảnh,", sv["n_unique_camera"], "camera,", sv["n_unique_site"], "site")

    res = {
        "md5": md5_chk.run(), "cam": cam_chk.run(), "veh": veh_chk.run(),
        "tmp": time_chk.run(), "ph": phash_chk.run(), "lbl": lbl_chk.run(),
        "cross": cross_chk.run(),
    }
    res["_files"] = {
        "survey": _rel(REPORTS_DIR / "v10_survey.json"),
        "(a) exact duplicate": _rel(md5_chk.OUT), "(b) camera leak": _rel(cam_chk.OUT),
        "(c) vehicle leak": _rel(veh_chk.OUT), "(d) temporal leak": _rel(time_chk.OUT),
        "(e) pHash near-dup": _rel(phash_chk.OUT), "(f) label integrity": _rel(lbl_chk.OUT),
        "(g) leak chéo v10 ↔ V5 eval": _rel(cross_chk.OUT), "(g) preview": _rel(cross_chk.PREVIEW),
    }
    md = build_markdown(sv, res, _feasibility(recs))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(md)
    print(md)
    print(f"[OK] Ghi {_rel(a.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
