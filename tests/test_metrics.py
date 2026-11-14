import pandas as pd
import pytest

from src.evaluation.metrics import (
    counting_accuracy, mae, mape, report, tracking_proxy_from_events,
)


def test_accuracy_perfect():
    assert counting_accuracy(100, 100) == 1.0


def test_accuracy_partial():
    assert counting_accuracy(90, 100) == 0.9


def test_accuracy_over():
    # pred vượt 20% 1 - 0.2 = 0.8
    assert abs(counting_accuracy(120, 100) - 0.8) < 1e-9


def test_mae_mape():
    assert mae([100, 50], [110, 45]) == (10 + 5) / 2
    # (10/110 + 5/45) / 2 * 100
    expected = (10 / 110 + 5 / 45) / 2 * 100
    assert abs(mape([100, 50], [110, 45]) - expected) < 1e-6


def test_report_shape():
    r = report({"motorcycle": 120, "car": 40},
               {"motorcycle": 130, "car": 45, "bus": 3})
    assert r["total_pred"] == 160
    assert r["total_gt"] == 178
    assert "MAPE_percent" in r
    assert "MAE" in r


def test_empty_report():
    assert report({}, {}) == {
        "per_class": {}, "total_pred": 0, "total_gt": 0,
        "MAE": 0.0, "MAPE_percent": 0.0, "overall_accuracy": 1.0,
    }


def test_metric_lengths_are_validated():
    with pytest.raises(ValueError):
        mae([1], [])
    with pytest.raises(ValueError):
        mape([1], [])


def test_track_lifetime_requires_per_frame_track_log(tmp_path):
    events = tmp_path / "events.csv"
    pd.DataFrame([
        {"frame": 10, "track_id": 1, "class_name": "car"},
    ]).to_csv(events, index=False)
    result = tracking_proxy_from_events(str(events), {"car": 1})
    assert result["overall"]["avg_track_lifetime_frames"] is None

    tracks = tmp_path / "tracks.csv"
    pd.DataFrame([
        {"frame": 10, "track_id": 1},
        {"frame": 11, "track_id": 1},
        {"frame": 12, "track_id": 1},
    ]).to_csv(tracks, index=False)
    result = tracking_proxy_from_events(str(events), {"car": 1}, str(tracks))
    assert result["overall"]["avg_track_lifetime_frames"] == 3.0
