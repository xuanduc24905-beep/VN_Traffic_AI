"""Test cho stats.aggregator."""
import pandas as pd
import pytest
from src.stats.aggregator import (
    events_to_df, summarize, counts_per_bucket, counts_per_minute,
    counts_per_hour, flow_rate, peak_period, cumulative_counts,
    format_bucket_label,
)


def _make_events():
    """Sinh 6 event mock trong 120 giây."""
    return [
        {"frame": 0, "time_sec": 5, "line": "L", "track_id": 1, "class_name": "car"},
        {"frame": 30, "time_sec": 15, "line": "L", "track_id": 2, "class_name": "car"},
        {"frame": 60, "time_sec": 30, "line": "L", "track_id": 3, "class_name": "bus"},
        {"frame": 90, "time_sec": 65, "line": "L", "track_id": 4, "class_name": "car"},
        {"frame": 120, "time_sec": 90, "line": "L", "track_id": 5, "class_name": "car"},
        {"frame": 150, "time_sec": 110, "line": "L", "track_id": 6, "class_name": "bus"},
    ]


def test_events_to_df_adds_time_columns():
    df = events_to_df(_make_events())
    assert "time_min" in df.columns
    assert "time_hour" in df.columns
    assert df.loc[df["time_sec"] == 65, "time_min"].iloc[0] == 1


def test_counts_per_bucket_minute():
    df = events_to_df(_make_events())
    pivot = counts_per_bucket(df, 60)
    # bucket 0 = 0-60s 2 car + 1 bus
    # bucket 1 = 60-120s 2 car + 1 bus
    assert pivot.loc[0, "car"] == 2
    assert pivot.loc[0, "bus"] == 1
    assert pivot.loc[1, "car"] == 2
    assert pivot.loc[1, "bus"] == 1


def test_counts_per_hour_single_bucket():
    df = events_to_df(_make_events())
    pivot = counts_per_hour(df)
    # Tất cả trong 1 giờ đầu tiên
    assert len(pivot) == 1
    assert pivot.loc[0, "car"] == 4
    assert pivot.loc[0, "bus"] == 2


def test_flow_rate_units():
    df = events_to_df(_make_events())
    fr_min = flow_rate(df, "minute")
    fr_hour = flow_rate(df, "hour")
    # duration = 110 - 5 = 105s xe/phút = 6 * 60 / 105 ≈ 3.43
    assert abs(fr_min["total"] - 6 * 60 / 105) < 0.1
    # xe/giờ = xe/phút * 60
    assert abs(fr_hour["total"] - fr_min["total"] * 60) < 1


def test_peak_period_finds_max():
    df = events_to_df(_make_events())
    peak = peak_period(df, 60)
    # 2 bucket đều = 3 sẽ trả về bucket đầu tiên đạt max
    assert peak["peak_total"] == 3


def test_peak_period_sparse_bucket_uses_label_not_position():
    df = events_to_df([
        {"frame": 300, "time_sec": 305, "line": "L",
         "track_id": 1, "class_name": "car"},
    ])
    peak = peak_period(df, 60)
    assert peak["peak_bucket"] == 5
    assert peak["peak_per_class"] == {"car": 1}


def test_cumulative_counts_monotonic():
    df = events_to_df(_make_events())
    cum = cumulative_counts(df)
    # Cột 'car' cuối cùng phải là 4
    assert cum["car"].iloc[-1] == 4
    assert cum["bus"].iloc[-1] == 2
    # Monotonic non-decreasing
    for col in cum.columns:
        assert (cum[col].diff().fillna(0) >= 0).all()


def test_format_bucket_label():
    assert format_bucket_label(30) == "30 giây"
    assert format_bucket_label(60) == "1 phút"
    assert format_bucket_label(900) == "15 phút"
    assert format_bucket_label(3600) == "1h"


def test_empty_df():
    empty = pd.DataFrame()
    assert counts_per_bucket(empty, 60).empty
    assert peak_period(empty, 60) == {}
    assert flow_rate(empty)["total"] == 0.0


def test_flow_rate_accepts_real_video_duration():
    df = events_to_df(_make_events())
    result = flow_rate(df, "minute", duration_sec=120)
    assert result["duration_sec"] == 120
    assert result["total"] == 3.0
    assert summarize(df, duration_sec=120)["duration_sec"] == 120


def test_invalid_bucket_and_flow_unit():
    df = events_to_df(_make_events())
    with pytest.raises(ValueError):
        counts_per_bucket(df, 0)
    with pytest.raises(ValueError):
        flow_rate(df, "day")
