"""Unit tests for robust line-crossing state."""
import json

import pytest

from src.tracking.counter import Counter, Line, _segments_cross, load_lines


def _box(center_y, center_x=100, size=10):
    half = size / 2
    return [center_x - half, center_y - half,
            center_x + half, center_y + half]


def _center_counter(line=None, **kwargs):
    line = line or Line(name="L", p1=(0, 100), p2=(200, 100))
    return Counter(lines=[line], anchor_mode="center", **kwargs)


def test_single_crossing_and_no_double_count():
    counter = _center_counter()
    counter.update(0, [_box(50)], [1], [0])
    events = counter.update(1, [_box(150)], [1], [0])
    assert events == [{"line": "L", "direction": "ltr",
                       "track_id": 1, "class_id": 0}]
    counter.update(2, [_box(50)], [1], [0])
    assert counter.counts["L"]["ltr"][0] == 1
    assert counter.total() == 1


def test_multi_class_nested_counts_schema():
    counter = _center_counter()
    counter.update(0, [_box(50, 80), _box(50, 180)], [1, 2], [0, 1])
    counter.update(1, [_box(150, 80), _box(150, 180)], [1, 2], [0, 1])
    assert counter.to_dict() == {"L": {"ltr": {0: 1, 1: 1}}}


def test_hysteresis_ignores_jitter_inside_band():
    counter = _center_counter(hysteresis_px=5)
    counter.update(0, [_box(80)], [1], [0])
    for frame, y in enumerate([98, 102, 99, 101], start=1):
        assert counter.update(frame, [_box(y)], [1], [0]) == []
    assert counter.total() == 0
    assert len(counter.update(5, [_box(120)], [1], [0])) == 1


def test_long_tracking_gap_does_not_create_phantom_crossing():
    counter = _center_counter(max_gap_frames=3)
    counter.update(0, [_box(50)], [1], [0])
    assert counter.update(10, [_box(150)], [1], [0]) == []
    assert counter.total() == 0


def test_wrong_direction_does_not_block_later_valid_crossing():
    line = Line(name="L", p1=(0, 100), p2=(200, 100),
                count_direction="ltr")
    counter = _center_counter(line)
    counter.update(0, [_box(150)], [1], [0])
    assert counter.update(1, [_box(50)], [1], [0]) == []  # rtl rejected
    assert len(counter.update(2, [_box(150)], [1], [0])) == 1


def test_class_is_voted_across_track_with_confidence():
    counter = _center_counter()
    counter.update(0, [_box(50)], [1], [2], [0.9])
    counter.update(1, [_box(60)], [1], [2], [0.8])
    events = counter.update(2, [_box(150)], [1], [3], [0.95])
    assert events[0]["class_id"] == 2


def test_bottom_center_is_default_anchor():
    counter = Counter(lines=[Line("L", (0, 100), (200, 100))])
    # Box center remains above the line, while its bottom crosses it.
    counter.update(0, [[95, 50, 105, 80]], [1], [0])
    events = counter.update(1, [[95, 70, 105, 120]], [1], [0])
    assert len(events) == 1


def test_segment_intersection_includes_endpoints():
    assert _segments_cross((100, 50), (100, 100), (0, 100), (200, 100))


def test_custom_direction_labels():
    line = Line("gate", (0, 100), (200, 100), count_direction="in",
                positive_label="in", negative_label="out")
    counter = _center_counter(line)
    counter.update(0, [_box(50)], [1], [0])
    event = counter.update(1, [_box(150)], [1], [0])[0]
    assert event["direction"] == "in"


def test_load_lines_scales_to_actual_resolution(tmp_path):
    config = {
        "cam": {
            "video_resolution": [1000, 500],
            "lines": [{"name": "gate", "points": [[100, 50], [900, 50]]}],
        }
    }
    path = tmp_path / "lines.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    line = load_lines(str(path), "cam", video_size=(2000, 1000))[0]
    assert line.p1 == (200.0, 100.0)
    assert line.p2 == (1800.0, 100.0)


def test_invalid_line_and_input_lengths():
    with pytest.raises(ValueError):
        Line("bad", (1, 1), (1, 1))
    counter = _center_counter()
    with pytest.raises(ValueError):
        counter.update(0, [_box(50)], [1, 2], [0])
