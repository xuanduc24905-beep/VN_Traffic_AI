"""Người 3 — Gom số liệu từ pipeline tracking+counting DataFrame + stats.

Input format (event log CSV):
    frame, time_sec, line, track_id, class_name

Các hàm chính:
- events_to_df: chuẩn hoá + thêm cột thời gian
- summarize: tóm tắt tổng thể
- counts_per_bucket: thống kê theo bucket giây (linh hoạt)
- counts_per_minute / per_hour: alias tiện dụng
- flow_rate: xe/phút, xe/giờ theo class
- peak_period: tìm bucket đông nhất
- cumulative_counts: đếm tích luỹ theo thời gian
"""
import pandas as pd
from pathlib import Path


def events_to_df(events) -> pd.DataFrame:
    """Nhận list dict hoặc DataFrame chuẩn hoá + enrich time columns."""
    df = pd.DataFrame(events) if not isinstance(events, pd.DataFrame) else events.copy()
    if df.empty:
        return df
    df["time_sec"] = df["time_sec"].astype(float)
    df["time_min"] = (df["time_sec"] // 60).astype(int)
    df["time_hour"] = (df["time_sec"] // 3600).astype(int)
    return df


def summarize(df: pd.DataFrame, duration_sec: float | None = None) -> dict:
    """Tóm tắt tổng thể + per-class + per-line + per-direction."""
    if duration_sec is not None and duration_sec < 0:
        raise ValueError("duration_sec must be >= 0 when provided")
    if df.empty:
        return {"total": 0, "per_class": {}, "per_line": {},
                "per_direction": {},
                "per_class_per_line": {},
                "per_class_per_direction": {},
                "duration_sec": 0}
    out = {
        "total": len(df),
        "per_class": df["class_name"].value_counts().to_dict(),
        "per_line": df["line"].value_counts().to_dict(),
        "per_class_per_line": (df.groupby(["line", "class_name"]).size()
                               .unstack(fill_value=0).to_dict("index")),
        "duration_sec": (float(duration_sec) if duration_sec is not None
                         else float(df["time_sec"].max())),
    }
    if "direction" in df.columns:
        out["per_direction"] = df["direction"].value_counts().to_dict()
        out["per_class_per_direction"] = (
            df.groupby(["direction", "class_name"]).size()
              .unstack(fill_value=0).to_dict("index"))
    else:
        out["per_direction"] = {}
        out["per_class_per_direction"] = {}
    return out


def counts_by_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Bảng {line × direction} tổng số lượt. Rỗng nếu không có cột direction."""
    if df.empty or "direction" not in df.columns:
        return pd.DataFrame()
    return (df.groupby(["line", "direction"]).size()
              .unstack(fill_value=0)
              .sort_index())


def counts_per_bucket(df: pd.DataFrame, bucket_sec: int,
                      class_order: list[str] | None = None) -> pd.DataFrame:
    """Pivot theo bucket giây. Hàng = bucket index, cột = class, giá trị = số lượt.

    bucket_sec = 60 theo phút. = 3600 theo giờ. = 300 theo 5 phút.
    """
    if bucket_sec <= 0:
        raise ValueError("bucket_sec must be > 0")
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["bucket"] = (df["time_sec"] // bucket_sec).astype(int)
    pivot = (df.groupby(["bucket", "class_name"]).size()
             .unstack(fill_value=0)
             .sort_index())
    if class_order:
        for c in class_order:
            if c not in pivot.columns:
                pivot[c] = 0
        pivot = pivot[class_order]
    return pivot


def counts_per_minute(df: pd.DataFrame, class_order=None) -> pd.DataFrame:
    return counts_per_bucket(df, 60, class_order)


def counts_per_hour(df: pd.DataFrame, class_order=None) -> pd.DataFrame:
    return counts_per_bucket(df, 3600, class_order)


def flow_rate(df: pd.DataFrame, unit: str = "minute",
              duration_sec: float | None = None) -> dict:
    """Tốc độ trung bình xe/đơn vị thời gian.

    unit ∈ {"minute", "hour"}.
    """
    if unit not in {"minute", "hour"}:
        raise ValueError("unit must be 'minute' or 'hour'")
    if duration_sec is not None and duration_sec <= 0:
        raise ValueError("duration_sec must be > 0 when provided")
    if df.empty:
        return {"unit": f"xe/{unit}", "total": 0.0, "per_class": {}}
    # Callers that know the video duration should pass it. Falling back to the
    # event span preserves the old API but can overestimate sparse traffic.
    duration = (float(duration_sec) if duration_sec is not None else
                float(df["time_sec"].max() - df["time_sec"].min()))
    if duration <= 0:
        duration = 1.0
    factor = 60.0 if unit == "minute" else 3600.0
    scale = factor / duration
    per_class = df["class_name"].value_counts().to_dict()
    return {
        "unit": f"xe/{unit}",
        "duration_sec": round(duration, 2),
        "total": round(len(df) * scale, 2),
        "per_class": {k: round(v * scale, 2) for k, v in per_class.items()},
    }


def peak_period(df: pd.DataFrame, bucket_sec: int = 60) -> dict:
    """Tìm bucket có tổng số lượt lớn nhất."""
    if df.empty:
        return {}
    pivot = counts_per_bucket(df, bucket_sec)
    if pivot.empty:
        return {}
    totals = pivot.sum(axis=1)
    peak_idx = int(totals.idxmax())
    return {
        "bucket_sec": bucket_sec,
        "peak_bucket": peak_idx,
        "peak_start_sec": peak_idx * bucket_sec,
        "peak_end_sec": (peak_idx + 1) * bucket_sec,
        "peak_total": int(totals.max()),
        "peak_per_class": pivot.loc[peak_idx].to_dict(),
    }


def cumulative_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Đếm tích luỹ theo thời gian. Hàng = time_sec, cột = class."""
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("time_sec")
    onehot = pd.get_dummies(df["class_name"])
    onehot.index = df["time_sec"].values
    return onehot.cumsum()


def save_csv(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True, encoding="utf-8-sig")


def format_bucket_label(bucket_sec: int) -> str:
    """Trả về tên bucket cho biểu đồ."""
    if bucket_sec >= 3600:
        return f"{bucket_sec // 3600}h"
    if bucket_sec >= 60:
        return f"{bucket_sec // 60} phút"
    return f"{bucket_sec} giây"
