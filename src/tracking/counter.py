"""Line-crossing counter backed by per-track state.

The counter waits until a track is observed on two stable sides of a line.
This is less sensitive to detector jitter than comparing two consecutive box
centres, and it avoids joining observations separated by a long tracking gap.

``ltr`` and ``rtl`` are retained as legacy names for positive/negative cross
product directions. They do *not* necessarily mean screen-left/screen-right.
Lines may provide domain labels such as ``in``/``out`` instead.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


_VALID_ANCHORS = {"center", "bottom_center"}


@dataclass(frozen=True)
class Line:
    name: str
    p1: tuple[float, float]
    p2: tuple[float, float]
    count_direction: str = "both"
    positive_label: str = "ltr"
    negative_label: str = "rtl"

    def __post_init__(self):
        if not self.name:
            raise ValueError("Line.name must not be empty")
        if len(self.p1) != 2 or len(self.p2) != 2:
            raise ValueError(f"Line '{self.name}' points must have two coordinates")
        if self.p1 == self.p2:
            raise ValueError(f"Line '{self.name}' must have non-zero length")
        if self.positive_label == self.negative_label:
            raise ValueError(f"Line '{self.name}' direction labels must be distinct")
        valid_directions = {"both", self.positive_label, self.negative_label}
        if self.count_direction not in valid_directions:
            raise ValueError(
                f"Line '{self.name}' count_direction must be one of "
                f"{sorted(valid_directions)}, got {self.count_direction!r}"
            )


def load_lines(config_path: str, video_key: str,
               video_size: tuple[int, int] | None = None) -> list[Line]:
    """Load, validate, and optionally scale counting-line coordinates."""
    path = Path(config_path)
    with path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    if video_key not in cfg:
        keys = sorted(k for k in cfg if not k.startswith("_"))
        raise KeyError(f"Unknown video key {video_key!r}; available keys: {keys}")

    video_cfg = cfg[video_key]
    raw_lines = video_cfg.get("lines")
    if not isinstance(raw_lines, list) or not raw_lines:
        raise ValueError(f"Video key {video_key!r} must contain a non-empty 'lines' list")

    scale_x = scale_y = 1.0
    configured_size = video_cfg.get("video_resolution")
    if video_size is not None and configured_size is not None:
        if len(configured_size) != 2 or min(configured_size) <= 0:
            raise ValueError(f"Invalid video_resolution for {video_key!r}")
        actual_w, actual_h = video_size
        if actual_w <= 0 or actual_h <= 0:
            raise ValueError(f"Invalid actual video size: {video_size}")
        scale_x = actual_w / configured_size[0]
        scale_y = actual_h / configured_size[1]

    lines = []
    seen_names = set()
    for raw in raw_lines:
        try:
            name = raw["name"]
            p1_raw, p2_raw = raw["points"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed line config for {video_key!r}: {raw!r}") from exc
        if name in seen_names:
            raise ValueError(f"Duplicate line name {name!r} for {video_key!r}")
        seen_names.add(name)

        labels = raw.get("direction_labels", {})
        if not isinstance(labels, dict):
            raise ValueError(f"direction_labels for line {name!r} must be an object")
        positive_label = labels.get("positive", "ltr")
        negative_label = labels.get("negative", "rtl")
        try:
            p1 = (float(p1_raw[0]) * scale_x, float(p1_raw[1]) * scale_y)
            p2 = (float(p2_raw[0]) * scale_x, float(p2_raw[1]) * scale_y)
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid points for line {name!r}") from exc
        lines.append(Line(
            name=name,
            p1=p1,
            p2=p2,
            count_direction=raw.get("count_direction", "both"),
            positive_label=positive_label,
            negative_label=negative_label,
        ))
    return lines


def _orientation(a, b, c, eps: float = 1e-9) -> int:
    value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(value) <= eps:
        return 0
    return 1 if value > 0 else -1


def _on_segment(a, p, b, eps: float = 1e-9) -> bool:
    return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps)


def _segments_cross(a1, a2, b1, b2) -> bool:
    """Return True for proper, endpoint, and collinear segment intersection."""
    o1 = _orientation(a1, a2, b1)
    o2 = _orientation(a1, a2, b2)
    o3 = _orientation(b1, b2, a1)
    o4 = _orientation(b1, b2, a2)
    if o1 != o2 and o3 != o4:
        return True
    return ((o1 == 0 and _on_segment(a1, b1, a2))
            or (o2 == 0 and _on_segment(a1, b2, a2))
            or (o3 == 0 and _on_segment(b1, a1, b2))
            or (o4 == 0 and _on_segment(b1, a2, b2)))


def _signed_distance(point, line: Line) -> float:
    """Signed perpendicular distance in pixels from point to infinite line."""
    vx = line.p2[0] - line.p1[0]
    vy = line.p2[1] - line.p1[1]
    px = point[0] - line.p1[0]
    py = point[1] - line.p1[1]
    return (vx * py - vy * px) / math.hypot(vx, vy)


def _crossing_direction(prev, curr, line: Line) -> str:
    """Map cross-product sign to the line's configured direction labels."""
    vx = line.p2[0] - line.p1[0]
    vy = line.p2[1] - line.p1[1]
    mx = curr[0] - prev[0]
    my = curr[1] - prev[1]
    cross = vx * my - vy * mx
    return line.positive_label if cross > 0 else line.negative_label


@dataclass
class _SideState:
    side: int
    point: tuple[float, float]


@dataclass
class Counter:
    lines: list[Line]
    anchor_mode: str = "bottom_center"
    hysteresis_px: float = 3.0
    max_gap_frames: int = 30
    state_ttl_frames: int = 300
    # counts[line_name][direction][class_id] = int
    counts: dict = field(default_factory=lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))))
    _counted: set = field(default_factory=set)
    _side_state: dict = field(default_factory=dict)
    _last_seen: dict = field(default_factory=dict)
    _class_votes: dict = field(default_factory=lambda: defaultdict(
        lambda: defaultdict(float)))
    _last_events: list = field(default_factory=list)

    def __post_init__(self):
        if self.anchor_mode not in _VALID_ANCHORS:
            raise ValueError(f"anchor_mode must be one of {sorted(_VALID_ANCHORS)}")
        if self.hysteresis_px < 0:
            raise ValueError("hysteresis_px must be >= 0")
        if self.max_gap_frames < 1:
            raise ValueError("max_gap_frames must be >= 1")
        if self.state_ttl_frames < self.max_gap_frames:
            raise ValueError("state_ttl_frames must be >= max_gap_frames")
        names = [line.name for line in self.lines]
        if len(names) != len(set(names)):
            raise ValueError("Line names must be unique")

    def _anchor(self, xyxy) -> tuple[float, float]:
        cx = (float(xyxy[0]) + float(xyxy[2])) / 2
        if self.anchor_mode == "bottom_center":
            return cx, float(xyxy[3])
        return cx, (float(xyxy[1]) + float(xyxy[3])) / 2

    def _expire_stale_tracks(self, frame_idx: int) -> None:
        stale = {tid for tid, last in self._last_seen.items()
                 if frame_idx - last > self.state_ttl_frames}
        if not stale:
            return
        for tid in stale:
            self._last_seen.pop(tid, None)
            self._class_votes.pop(tid, None)
        self._side_state = {
            key: value for key, value in self._side_state.items()
            if key[1] not in stale
        }
        self._counted = {key for key in self._counted if key[1] not in stale}

    def update(self, frame_idx: int, boxes_xyxy, ids, classes,
               confidences: Iterable[float] | None = None):
        """Update track states and return crossing events from this frame."""
        self._expire_stale_tracks(frame_idx)
        boxes = list(boxes_xyxy)
        track_ids = list(ids)
        class_ids = list(classes)
        if not (len(boxes) == len(track_ids) == len(class_ids)):
            raise ValueError("boxes_xyxy, ids, and classes must have equal lengths")
        confs = [1.0] * len(boxes) if confidences is None else list(confidences)
        if len(confs) != len(boxes):
            raise ValueError("confidences must have the same length as boxes_xyxy")

        events = []
        for xyxy, raw_tid, raw_cls, confidence in zip(
                boxes, track_ids, class_ids, confs):
            if raw_tid is None:
                continue
            tid = int(raw_tid)
            cls = int(raw_cls)
            point = self._anchor(xyxy)
            previous_seen = self._last_seen.get(tid)
            had_long_gap = (previous_seen is not None
                            and frame_idx - previous_seen > self.max_gap_frames)
            self._last_seen[tid] = frame_idx
            if had_long_gap:
                self._class_votes[tid].clear()
            self._class_votes[tid][cls] += max(float(confidence), 0.0)

            for line in self.lines:
                key = (line.name, tid)
                distance = _signed_distance(point, line)
                side = 1 if distance > self.hysteresis_px else (
                    -1 if distance < -self.hysteresis_px else 0)
                previous = self._side_state.get(key)

                if had_long_gap:
                    previous = None
                    self._side_state.pop(key, None)
                if side == 0:
                    continue
                if previous is None or previous.side == side:
                    self._side_state[key] = _SideState(side=side, point=point)
                    continue

                direction = _crossing_direction(previous.point, point, line)
                if (_segments_cross(previous.point, point, line.p1, line.p2)
                        and key not in self._counted
                        and line.count_direction in {"both", direction}):
                    stable_cls = max(self._class_votes[tid],
                                     key=self._class_votes[tid].get)
                    self.counts[line.name][direction][stable_cls] += 1
                    self._counted.add(key)
                    events.append({
                        "line": line.name,
                        "direction": direction,
                        "track_id": tid,
                        "class_id": stable_cls,
                    })
                # A disallowed-direction crossing is not marked as counted;
                # a later crossing in the configured direction remains valid.
                self._side_state[key] = _SideState(side=side, point=point)

        self._last_events = events
        return events

    def to_dict(self) -> dict:
        """Return ``{line: {direction: {class_id: count}}}``."""
        return {ln: {direction: dict(per_class)
                     for direction, per_class in directions.items()}
                for ln, directions in self.counts.items()}

    def totals_per_line(self) -> dict:
        return {line: sum(sum(per_class.values())
                          for per_class in directions.values())
                for line, directions in self.counts.items()}

    def total(self) -> int:
        return sum(self.totals_per_line().values())
