"""GPU-free tests for the runtime pipeline orchestration."""
import csv

import numpy as np
import pytest

from src.pipeline import run as pipeline
from src.tracking.counter import Line


class _Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _Boxes:
    def __init__(self, xyxy, ids, classes, confidence):
        self.xyxy = _Tensor(xyxy)
        self.id = None if ids is None else _Tensor(ids)
        self.cls = _Tensor(classes)
        self.conf = _Tensor(confidence)


class _Result:
    names = {0: "car", 1: "truck"}

    def __init__(self, xyxy, ids, classes, confidence):
        self.boxes = _Boxes(xyxy, ids, classes, confidence)
        self.orig_img = np.zeros((100, 200, 3), dtype=np.uint8)


def test_run_pipeline_writes_crossing_csv_atomically(tmp_path, monkeypatch):
    results = [
        _Result([[95, 20, 105, 40]], [1], [0], [0.9]),
        _Result([[95, 40, 105, 60]], [1], [0], [0.8]),
    ]
    monkeypatch.setattr(
        pipeline, "probe_video",
        lambda _: pipeline.VideoMetadata(25.0, 200, 100, 2),
    )
    monkeypatch.setattr(pipeline, "track_stream", lambda *args, **kwargs: iter(results))

    output = tmp_path / "events.csv"
    events, totals, names = pipeline.run_pipeline(
        "fake.mp4", "fake.pt", [Line("gate", (0, 50), (200, 50))],
        out_csv=str(output),
    )

    assert len(events) == 1
    assert totals == {"gate": {"ltr": {0: 1}}}
    assert names == {0: "car", 1: "truck"}
    with output.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["class_name"] == "car"
    assert not list(tmp_path.glob("*.tmp"))


def test_probe_video_rejects_missing_source(tmp_path):
    with pytest.raises(ValueError, match="Không mở được video"):
        pipeline.probe_video(str(tmp_path / "missing.mp4"))
