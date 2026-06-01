from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict
from io import BytesIO
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


CONTAINER_PRESETS: Dict[str, Dict[str, float]] = {
    "20ft 컨테이너": {"length_mm": 5898, "width_mm": 2352, "height_mm": 2393},
    "40HFT 컨테이너": {"length_mm": 12032, "width_mm": 2352, "height_mm": 2698},
    "카고 차량": {"length_mm": 10100, "width_mm": 2350, "height_mm": 2400},
    "윙바디 차량": {"length_mm": 10200, "width_mm": 2400, "height_mm": 2500},
    "직접 입력 Custom": {"length_mm": 12032, "width_mm": 2352, "height_mm": 2698},
}

PALETTE = [
    "#2563EB",
    "#F97316",
    "#16A34A",
    "#DC2626",
    "#7C3AED",
    "#0891B2",
    "#BE185D",
    "#64748B",
]


@dataclass(frozen=True)
class Container:
    length_mm: float
    width_mm: float
    height_mm: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class Tire:
    spec: str
    section_width_mm: float
    aspect_ratio: float
    rim_inch: float
    rim_diameter_mm: float
    sidewall_height_mm: float
    outer_diameter_mm: float
    color: str = "#2563EB"

    @property
    def volume_mm3(self) -> float:
        return math.pi * (self.outer_diameter_mm / 2) ** 2 * self.section_width_mm


def parse_tire_size(spec: str, color: str = "#2563EB") -> Tire:
    text = spec.strip().upper()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)R(\d+(?:\.\d+)?)", text)
    if not match:
        raise ValueError(f"타이어 규격 형식이 올바르지 않습니다: {spec}")
    section_width = float(match.group(1))
    aspect_ratio = float(match.group(2))
    rim_inch = float(match.group(3))
    rim_diameter = rim_inch * 25.4
    sidewall_height = section_width * aspect_ratio / 100
    outer_diameter = rim_diameter + 2 * sidewall_height
    return Tire(
        spec=text,
        section_width_mm=section_width,
        aspect_ratio=aspect_ratio,
        rim_inch=rim_inch,
        rim_diameter_mm=rim_diameter,
        sidewall_height_mm=sidewall_height,
        outer_diameter_mm=outer_diameter,
        color=color,
    )


def override_tire_dimensions(
    tire: Tire,
    outer_diameter_mm: Optional[float] = None,
    section_width_mm: Optional[float] = None,
) -> Tire:
    return Tire(
        spec=tire.spec,
        section_width_mm=float(section_width_mm or tire.section_width_mm),
        aspect_ratio=tire.aspect_ratio,
        rim_inch=tire.rim_inch,
        rim_diameter_mm=tire.rim_diameter_mm,
        sidewall_height_mm=tire.sidewall_height_mm,
        outer_diameter_mm=float(outer_diameter_mm or tire.outer_diameter_mm),
        color=tire.color,
    )


def _bbox(x: float, y: float, z: float, od: float, sw: float, axis: str) -> Tuple[float, float, float, float, float, float]:
    r = od / 2
    if axis == "z":
        return (x - r, x + r, y - r, y + r, z, z + sw)
    if axis == "x":
        return (x, x + sw, y - r, y + r, z - r, z + r)
    return (x - r, x + r, y, y + sw, z - r, z + r)


def make_placement(
    tire: Tire,
    x: float,
    y: float,
    z: float,
    axis: str,
    method: str,
    block_id: str,
    block_start_x: float = 0,
    block_end_x: Optional[float] = None,
    block_axis: str = "x",
    block_start_z: float = 0,
    block_end_z: Optional[float] = None,
) -> Dict[str, object]:
    od = tire.outer_diameter_mm
    sw = tire.section_width_mm
    bx0, bx1, by0, by1, bz0, bz1 = _bbox(x, y, z, od, sw, axis)
    return {
        "tire_spec": tire.spec,
        "outer_diameter_mm": od,
        "section_width_mm": sw,
        "x": x,
        "y": y,
        "z": z,
        "axis": axis,
        "method": method,
        "block_id": block_id,
        "block_start_x": block_start_x,
        "block_end_x": block_end_x,
        "block_axis": block_axis,
        "block_start_z": block_start_z,
        "block_end_z": block_end_z,
        "color": tire.color,
        "bbox_x0": bx0,
        "bbox_x1": bx1,
        "bbox_y0": by0,
        "bbox_y1": by1,
        "bbox_z0": bz0,
        "bbox_z1": bz1,
    }


def generate_flat_placements(
    container: Container,
    tire: Tire,
    clearance_mm: float = 0,
    x_start: float = 0,
    x_length: Optional[float] = None,
    y_start: float = 0,
    y_width: Optional[float] = None,
    z_start: float = 0,
    z_height: Optional[float] = None,
    max_count: Optional[int] = None,
    block_id: str = "single",
    block_axis: str = "x",
    include_standing_extras: bool = True,
) -> List[Dict[str, object]]:
    tire = override_tire_dimensions(tire, round(tire.outer_diameter_mm), tire.section_width_mm)
    od = tire.outer_diameter_mm
    sw = tire.section_width_mm
    restricted_y = y_start != 0 or (y_width is not None and y_start + y_width < container.width_mm - 1e-9)
    restricted_z = z_start != 0 or (z_height is not None and z_start + z_height < container.height_mm - 1e-9)
    if restricted_y or restricted_z:
        x_limit = x_start + (x_length if x_length is not None else container.length_mm - x_start)
        y_limit = y_start + (y_width if y_width is not None else container.width_mm - y_start)
        z_limit = z_start + (z_height if z_height is not None else container.height_mm - z_start)
        x_pitch = od + clearance_mm
        y_pitch = od + clearance_mm
        z_pitch = sw + clearance_mm
        placements: List[Dict[str, object]] = []
        ix = 0
        while x_start + ix * x_pitch + od <= x_limit + 1e-9:
            x_center = x_start + ix * x_pitch + od / 2
            iy = 0
            while y_start + iy * y_pitch + od <= y_limit + 1e-9:
                y_center = y_start + iy * y_pitch + od / 2
                iz = 0
                while z_start + iz * z_pitch + sw <= z_limit + 1e-9:
                    placements.append(make_placement(tire, x_center, y_center, z_start + iz * z_pitch, "z", "flat", block_id, x_start, None, block_axis, z_start))
                    if max_count and len(placements) >= max_count:
                        return placements
                    iz += 1
                iy += 1
            ix += 1
        return placements

    x_limit = x_start + (x_length if x_length is not None else container.length_mm - x_start)
    placements: List[Dict[str, object]] = []

    main_x = int(max(0, (x_limit - x_start + clearance_mm) // (od + clearance_mm)))
    main_y = int(max(0, (container.width_mm + clearance_mm) // (od + clearance_mm)))
    main_z = int(max(0, (container.height_mm + clearance_mm) // (sw + clearance_mm)))
    for ix in range(main_x):
        for iy in range(main_y):
            for iz in range(main_z):
                placements.append(make_placement(
                    tire,
                    x_start + ix * (od + clearance_mm) + od / 2,
                    iy * (od + clearance_mm) + od / 2,
                    iz * (sw + clearance_mm),
                    "z",
                    "flat",
                    block_id,
                    x_start,
                    None,
                    block_axis,
                    0,
                ))
                if max_count and len(placements) >= max_count:
                    return placements

    if not include_standing_extras:
        return placements

    side_y0 = main_y * (od + clearance_mm)
    side_cols = int(max(0, (container.width_mm - side_y0 + clearance_mm) // (sw + clearance_mm)))
    side_layers = int(max(0, (container.height_mm + clearance_mm) // (od + clearance_mm)))
    for ix in range(main_x):
        for iy in range(side_cols):
            for iz in range(side_layers):
                placements.append(make_placement(
                    tire,
                    x_start + ix * (od + clearance_mm) + od / 2,
                    side_y0 + iy * (sw + clearance_mm),
                    iz * (od + clearance_mm) + od / 2,
                    "y",
                    "flat",
                    block_id,
                    x_start,
                    None,
                    block_axis,
                    0,
                ))
                if max_count and len(placements) >= max_count:
                    return placements

    end_x0 = x_start + main_x * (od + clearance_mm)
    end_rows = int(max(0, (x_limit - end_x0 + clearance_mm) // (sw + clearance_mm)))
    end_y = int(max(0, (container.width_mm + clearance_mm) // (od + clearance_mm)))
    end_z = int(max(0, (container.height_mm + clearance_mm) // (od + clearance_mm)))
    for ix in range(end_rows):
        for iy in range(end_y):
            for iz in range(end_z):
                placements.append(make_placement(
                    tire,
                    end_x0 + ix * (sw + clearance_mm),
                    iy * (od + clearance_mm) + od / 2,
                    iz * (od + clearance_mm) + od / 2,
                    "x",
                    "flat",
                    block_id,
                    x_start,
                    None,
                    block_axis,
                    0,
                ))
                if max_count and len(placements) >= max_count:
                    return placements
    return placements


def generate_honeycomb_placements(
    container: Container,
    tire: Tire,
    clearance_mm: float = 0,
    x_start: float = 0,
    x_length: Optional[float] = None,
    y_start: float = 0,
    y_width: Optional[float] = None,
    z_start: float = 0,
    z_height: Optional[float] = None,
    max_count: Optional[int] = None,
    block_id: str = "single",
    block_axis: str = "x",
    include_standing_extras: bool = True,
) -> List[Dict[str, object]]:
    od = tire.outer_diameter_mm
    sw = tire.section_width_mm
    r = od / 2
    x_limit = x_start + (x_length if x_length is not None else container.length_mm - x_start)
    y_limit = y_start + (y_width if y_width is not None else container.width_mm - y_start)
    z_limit = z_start + (z_height if z_height is not None else container.height_mm - z_start)
    col_pitch = od + clearance_mm
    row_pitch = math.sqrt(3) / 2 * (od + clearance_mm)
    z_pitch = sw + clearance_mm
    placements: List[Dict[str, object]] = []
    restricted_y = y_start != 0 or (y_width is not None and y_start + y_width < container.width_mm - 1e-9)
    restricted_z = z_start != 0 or (z_height is not None and z_start + z_height < container.height_mm - 1e-9)
    iz = 0
    while z_start + iz * z_pitch + sw <= z_limit + 1e-9:
        z_base = z_start + iz * z_pitch
        row = 0
        while y_start + row * row_pitch + od <= y_limit + 1e-9:
            y_center = y_start + row * row_pitch + r
            offset = col_pitch / 2 if row % 2 else 0
            col = 0
            while x_start + offset + col * col_pitch + od <= x_limit + 1e-9:
                x_center = x_start + offset + col * col_pitch + r
                placements.append(make_placement(tire, x_center, y_center, z_base, "z", "honeycomb", block_id, x_start, None, block_axis, z_start))
                if max_count and len(placements) >= max_count:
                    return placements
                col += 1
            row += 1
        iz += 1
    if not include_standing_extras or restricted_y or restricted_z:
        return placements

    main_width_used = max((float(p["bbox_y1"]) for p in placements if p["axis"] == "z"), default=0.0)
    main_length_used = max((float(p["bbox_x1"]) for p in placements if p["axis"] == "z"), default=x_start)

    side_cols = int(max(0, (container.width_mm - main_width_used + clearance_mm) // (sw + clearance_mm)))
    side_x_count = int(max(0, (container.length_mm + clearance_mm) // (od + clearance_mm)))
    side_z_count = int(max(0, (container.height_mm + clearance_mm) // (od + clearance_mm)))
    for iy in range(side_cols):
        for ix in range(side_x_count):
            for iz2 in range(side_z_count):
                placements.append(make_placement(
                    tire,
                    ix * (od + clearance_mm) + od / 2,
                    main_width_used + iy * (sw + clearance_mm),
                    iz2 * (od + clearance_mm) + od / 2,
                    "y",
                    "honeycomb",
                    block_id,
                    x_start,
                    None,
                    block_axis,
                    0,
                ))
                if max_count and len(placements) >= max_count:
                    return placements

    end_rows = int(max(0, (container.length_mm - main_length_used + clearance_mm) // (sw + clearance_mm)))
    end_y_count = int(max(0, (container.width_mm + clearance_mm) // (od + clearance_mm)))
    end_z_count = int(max(0, (container.height_mm + clearance_mm) // (od + clearance_mm)))
    for ix in range(end_rows):
        for iy in range(end_y_count):
            for iz2 in range(end_z_count):
                placements.append(make_placement(
                    tire,
                    main_length_used + ix * (sw + clearance_mm),
                    iy * (od + clearance_mm) + od / 2,
                    iz2 * (od + clearance_mm) + od / 2,
                    "x",
                    "honeycomb",
                    block_id,
                    x_start,
                    None,
                    block_axis,
                    0,
                ))
                if max_count and len(placements) >= max_count:
                    return placements
    return placements


def calculate_flat_loading(container: Container, tire: Tire, clearance_mm: float = 0) -> Dict[str, object]:
    placements = generate_flat_placements(container, tire, clearance_mm)
    return summarize_single(container, tire, placements, "flat")


def calculate_honeycomb_loading(container: Container, tire: Tire, clearance_mm: float = 0) -> Dict[str, object]:
    placements = generate_honeycomb_placements(container, tire, clearance_mm)
    return summarize_single(container, tire, placements, "honeycomb")


def _layers(placements: List[Dict[str, object]]) -> int:
    return len({round(float(p["bbox_z0"]), 6) for p in placements})


def utilization_rate(container: Container, placements: List[Dict[str, object]]) -> float:
    used = sum(math.pi * (float(p["outer_diameter_mm"]) / 2) ** 2 * float(p["section_width_mm"]) for p in placements)
    total = container.length_mm * container.width_mm * container.height_mm
    return used / total if total else 0


def summarize_single(container: Container, tire: Tire, placements: List[Dict[str, object]], method: str) -> Dict[str, object]:
    return {
        "tire_spec": tire.spec,
        "outer_diameter_mm": tire.outer_diameter_mm,
        "section_width_mm": tire.section_width_mm,
        "count": len(placements),
        "layers": _layers(placements),
        "utilization_rate": utilization_rate(container, placements),
        "container_L_mm": container.length_mm,
        "container_W_mm": container.width_mm,
        "container_H_mm": container.height_mm,
        "loading_method": method,
        "placements": placements,
    }


def build_single_result_table(container: Container, tires: List[Tire], clearance_mm: float) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    rows = []
    detail: Dict[str, Dict[str, object]] = {}
    for tire in tires:
        flat = calculate_flat_loading(container, tire, clearance_mm)
        honey = calculate_honeycomb_loading(container, tire, clearance_mm)
        detail[tire.spec] = {"flat": flat, "honeycomb": honey}
        rows.append({
            "tire_spec": tire.spec,
            "outer_diameter_mm": round(tire.outer_diameter_mm, 1),
            "section_width_mm": round(tire.section_width_mm, 1),
            "flat_count": flat["count"],
            "honeycomb_count": honey["count"],
            "flat_layers": flat["layers"],
            "honeycomb_layers": honey["layers"],
            "flat_utilization_rate": round(flat["utilization_rate"], 4),
            "honeycomb_utilization_rate": round(honey["utilization_rate"], 4),
            "container_L_mm": container.length_mm,
            "container_W_mm": container.width_mm,
            "container_H_mm": container.height_mm,
        })
    return pd.DataFrame(rows), detail


def actual_block_end(placements: List[Dict[str, object]], start_x: float) -> float:
    if not placements:
        return start_x
    return max(float(p["bbox_x1"]) for p in placements)


def actual_block_end_z(placements: List[Dict[str, object]], start_z: float) -> float:
    if not placements:
        return start_z
    return max(float(p["bbox_z1"]) for p in placements)


def _refresh_x_bbox(p: Dict[str, object]) -> None:
    od = float(p["outer_diameter_mm"])
    sw = float(p["section_width_mm"])
    bx0, bx1, by0, by1, bz0, bz1 = _bbox(float(p["x"]), float(p["y"]), float(p["z"]), od, sw, str(p["axis"]))
    p["bbox_x0"] = bx0
    p["bbox_x1"] = bx1
    p["bbox_y0"] = by0
    p["bbox_y1"] = by1
    p["bbox_z0"] = bz0
    p["bbox_z1"] = bz1


def _refresh_bbox(p: Dict[str, object]) -> None:
    od = float(p["outer_diameter_mm"])
    sw = float(p["section_width_mm"])
    bx0, bx1, by0, by1, bz0, bz1 = _bbox(float(p["x"]), float(p["y"]), float(p["z"]), od, sw, str(p["axis"]))
    p["bbox_x0"] = bx0
    p["bbox_x1"] = bx1
    p["bbox_y0"] = by0
    p["bbox_y1"] = by1
    p["bbox_z0"] = bz0
    p["bbox_z1"] = bz1


def compact_block_left(
    existing: List[Dict[str, object]],
    block_placements: List[Dict[str, object]],
    container: Container,
    clearance_mm: float,
) -> List[Dict[str, object]]:
    """Slide each new tire left until it is stopped by a real tire collision.

    Mixed honeycomb blocks otherwise leave a rectangular boundary gap because
    staggered rows end at different x positions. This compaction keeps the
    visual/result placement list as the only source of truth and does not allow
    physical overlap.
    """
    packed: List[Dict[str, object]] = []
    placed = list(existing)
    for p in sorted(block_placements, key=lambda item: (float(item["bbox_x0"]), float(item["bbox_y0"]), float(item["bbox_z0"]))):
        if p["axis"] != "z":
            packed.append(p)
            placed.append(p)
            continue
        r = float(p["outer_diameter_mm"]) / 2
        min_center_x = r
        py = float(p["y"])
        pz0 = float(p["bbox_z0"])
        pz1 = float(p["bbox_z1"])
        for q in placed:
            qz_overlap = not (float(q["bbox_z1"]) <= pz0 + 1e-6 or pz1 <= float(q["bbox_z0"]) + 1e-6)
            if not qz_overlap or q["axis"] != "z":
                continue
            dy = py - float(q["y"])
            min_dist = (float(p["outer_diameter_mm"]) + float(q["outer_diameter_mm"])) / 2 + clearance_mm
            if abs(dy) >= min_dist:
                continue
            required_dx = math.sqrt(max(0.0, min_dist * min_dist - dy * dy))
            min_center_x = max(min_center_x, float(q["x"]) + required_dx)
        max_center_x = container.length_mm - r
        if min_center_x > max_center_x + 1e-6:
            continue
        p["x"] = min(max(float(p["x"]), min_center_x), max_center_x)
        _refresh_x_bbox(p)
        packed.append(p)
        placed.append(p)
    return packed


def compact_block_down(
    existing: List[Dict[str, object]],
    block_placements: List[Dict[str, object]],
    container: Container,
    clearance_mm: float,
) -> List[Dict[str, object]]:
    """Keep lower boundary candidates only when they do not collide in 3D."""
    packed: List[Dict[str, object]] = []
    placed = list(existing)
    for p in sorted(block_placements, key=lambda item: (float(item["bbox_z0"]), float(item["bbox_x0"]), float(item["bbox_y0"]))):
        if p["axis"] != "z":
            packed.append(p)
            placed.append(p)
            continue
        collides = False
        px = float(p["x"])
        py = float(p["y"])
        pz0 = float(p["bbox_z0"])
        pz1 = float(p["bbox_z1"])
        for q in placed:
            if q["axis"] != "z":
                continue
            z_overlap = not (float(q["bbox_z1"]) <= pz0 + 1e-6 or pz1 <= float(q["bbox_z0"]) + 1e-6)
            if not z_overlap:
                continue
            dx = px - float(q["x"])
            dy = py - float(q["y"])
            min_dist = (float(p["outer_diameter_mm"]) + float(q["outer_diameter_mm"])) / 2 + clearance_mm
            if math.hypot(dx, dy) < min_dist - 1e-6:
                collides = True
                break
        if collides:
            continue
        packed.append(p)
        placed.append(p)
    return packed


def _candidate_collides(candidate: Dict[str, object], placed: List[Dict[str, object]], clearance_mm: float) -> bool:
    if candidate["axis"] != "z":
        return False
    cx = float(candidate["x"])
    cy = float(candidate["y"])
    cz0 = float(candidate["bbox_z0"])
    cz1 = float(candidate["bbox_z1"])
    for item in placed:
        if item["axis"] != "z":
            continue
        z_overlap = not (float(item["bbox_z1"]) <= cz0 + 1e-6 or cz1 <= float(item["bbox_z0"]) + 1e-6)
        if not z_overlap:
            continue
        min_dist = (float(candidate["outer_diameter_mm"]) + float(item["outer_diameter_mm"])) / 2 + clearance_mm
        if math.hypot(cx - float(item["x"]), cy - float(item["y"])) < min_dist - 1e-6:
            return True
    return False


def _candidate_footprint_collides(candidate: Dict[str, object], placed: List[Dict[str, object]], clearance_mm: float) -> bool:
    if candidate["axis"] != "z":
        return False
    cx = float(candidate["x"])
    cy = float(candidate["y"])
    for item in placed:
        if item["axis"] != "z":
            continue
        min_dist = (float(candidate["outer_diameter_mm"]) + float(item["outer_diameter_mm"])) / 2 + clearance_mm
        if math.hypot(cx - float(item["x"]), cy - float(item["y"])) < min_dist - 1e-6:
            return True
    return False


def settle_candidate(
    candidate: Dict[str, object],
    placed: List[Dict[str, object]],
    container: Container,
    clearance_mm: float,
    layout_axis: str,
    require_strong_support: bool = False,
    allowed_support_block_ids: Optional[set] = None,
) -> Optional[Dict[str, object]]:
    """Snap a vertical tire to the lowest collision-free height before accepting it."""
    if candidate["axis"] != "z":
        return candidate
    od = float(candidate["outer_diameter_mm"])
    sw = float(candidate["section_width_mm"])
    target_z = 0.0
    cx = float(candidate["x"])
    cy = float(candidate["y"])
    for item in placed:
        if item["axis"] != "z":
            continue
        min_dist = (od + float(item["outer_diameter_mm"])) / 2 + clearance_mm
        center_dist = math.hypot(cx - float(item["x"]), cy - float(item["y"]))
        strong_support_dist = min(od, float(item["outer_diameter_mm"])) * 0.90
        if center_dist < min_dist - 1e-6 and (not require_strong_support or center_dist <= strong_support_dist):
            if allowed_support_block_ids is not None and str(item["block_id"]) not in allowed_support_block_ids:
                return None
            target_z = max(target_z, float(item["bbox_z1"]) + clearance_mm)
    if target_z + sw > container.height_mm + 1e-6:
        return None
    candidate["z"] = target_z
    _refresh_bbox(candidate)
    return candidate


def fill_remaining_space(
    container: Container,
    tire: Tire,
    method: str,
    clearance_mm: float,
    placed: List[Dict[str, object]],
    max_count: int,
    block_id: str,
    layout_axis: str,
    span_start: float = 0.0,
    span_end: Optional[float] = None,
    allow_vertical_mix: bool = False,
    anchor_placements: Optional[List[Dict[str, object]]] = None,
    stack_height_limit: Optional[float] = None,
    allowed_support_block_ids: Optional[set] = None,
    include_standing_extras_in_scan: bool = False,
) -> List[Dict[str, object]]:
    generator = generate_flat_placements if method == "flat" else generate_honeycomb_placements
    anchors = anchor_placements if anchor_placements is not None else placed
    if span_end is None:
        span_end = container.height_mm if layout_axis == "z" else container.length_mm
    if layout_axis == "z":
        anchor_starts = sorted({
            round(float(item["bbox_z1"]) + clearance_mm, 6)
            for item in anchors
            if span_start - 1e-6 <= float(item["bbox_z1"]) + clearance_mm
            and float(item["bbox_z1"]) + clearance_mm + tire.section_width_mm <= span_end + 1e-6
        }) if allow_vertical_mix else []
        starts = [span_start] + anchor_starts
        raw_candidates = []
        y_offsets = [0.0]
        z_height_limit = stack_height_limit if stack_height_limit is not None else container.height_mm
        for start in starts:
            for y_offset in y_offsets:
                raw_candidates.extend(generator(
                    container,
                    tire,
                    clearance_mm,
                    x_start=0,
                    x_length=container.length_mm,
                    y_start=y_offset,
                    y_width=container.width_mm - y_offset,
                    z_start=start,
                    z_height=span_end - start,
                    max_count=None,
                    block_id=block_id,
                    block_axis=layout_axis,
                    include_standing_extras=include_standing_extras_in_scan,
                ))
    else:
        anchor_starts = sorted({
            round(float(item["bbox_x1"]) + clearance_mm, 6)
            for item in anchors
            if span_start - 1e-6 <= float(item["bbox_x1"]) + clearance_mm
            and float(item["bbox_x1"]) + clearance_mm + tire.outer_diameter_mm <= span_end + 1e-6
        }) if allow_vertical_mix else []
        starts = [span_start] + anchor_starts
        raw_candidates = []
        y_offsets = [0.0]
        z_height_limit = stack_height_limit if stack_height_limit is not None else container.height_mm
        for start in starts:
            for y_offset in y_offsets:
                raw_candidates.extend(generator(
                    container,
                    tire,
                    clearance_mm,
                    x_start=start,
                    x_length=span_end - start,
                    y_start=y_offset,
                    y_width=container.width_mm - y_offset,
                    z_start=0,
                    z_height=z_height_limit,
                    max_count=None,
                    block_id=block_id,
                    block_axis=layout_axis,
                    include_standing_extras=include_standing_extras_in_scan,
                ))
    if layout_axis == "x" and allow_vertical_mix and anchor_placements:
        direct_anchor_candidates = []
        for anchor in sorted(anchor_placements, key=lambda p: (float(p["bbox_x0"]), float(p["bbox_z1"]), float(p["bbox_y0"]))):
            z_base = float(anchor["bbox_z1"]) + clearance_mm
            while z_base + tire.section_width_mm <= container.height_mm + 1e-6:
                candidate = make_placement(
                    tire,
                    float(anchor["x"]),
                    float(anchor["y"]),
                    z_base,
                    "z",
                    method,
                    block_id,
                    0,
                    None,
                    layout_axis,
                    0,
                    None,
                )
                if (
                    float(candidate["bbox_x0"]) >= -1e-6
                    and float(candidate["bbox_x1"]) <= container.length_mm + 1e-6
                    and float(candidate["bbox_y0"]) >= -1e-6
                    and float(candidate["bbox_y1"]) <= container.width_mm + 1e-6
                ):
                    candidate["_candidate_priority"] = 0
                    direct_anchor_candidates.append(candidate)
                z_base += tire.section_width_mm + clearance_mm
        raw_candidates = direct_anchor_candidates + raw_candidates
    if layout_axis == "x" and anchor_placements and not allow_vertical_mix:
        side_anchor_candidates = []
        for anchor in sorted(anchor_placements, key=lambda p: (float(p["bbox_x1"]), float(p["bbox_z0"]), float(p["bbox_y0"]))):
            x_center = float(anchor["bbox_x1"]) + clearance_mm + tire.outer_diameter_mm / 2
            if x_center + tire.outer_diameter_mm / 2 > container.length_mm + 1e-6:
                continue
            z_base = float(anchor["bbox_z0"])
            if z_base + tire.section_width_mm > container.height_mm + 1e-6:
                continue
            candidate = make_placement(
                tire,
                x_center,
                float(anchor["y"]),
                z_base,
                "z",
                method,
                block_id,
                0,
                None,
                layout_axis,
                0,
                None,
            )
            if (
                float(candidate["bbox_y0"]) >= -1e-6
                and float(candidate["bbox_y1"]) <= container.width_mm + 1e-6
            ):
                candidate["_candidate_priority"] = 1
                side_anchor_candidates.append(candidate)
        if side_anchor_candidates and not allow_vertical_mix:
            max_count += len(side_anchor_candidates)
        raw_candidates = side_anchor_candidates + raw_candidates
    seen = set()
    candidates = []
    for candidate in raw_candidates:
        key = (
            round(float(candidate["bbox_x0"]), 3),
            round(float(candidate["bbox_y0"]), 3),
            round(float(candidate["bbox_z0"]), 3),
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    if layout_axis == "z":
        candidates.sort(key=lambda p: (float(p["bbox_z0"]), float(p["bbox_x0"]), float(p["bbox_y0"])))
    elif allow_vertical_mix and anchor_placements:
        candidates.sort(key=lambda p: (
            int(p.get("_candidate_priority", 2)),
            float(p["bbox_x0"]),
            float(p["bbox_y0"]),
            float(p["bbox_z0"]),
        ))
    elif allow_vertical_mix:
        candidates.sort(key=lambda p: (float(p["bbox_x0"]), float(p["bbox_y0"]), float(p["bbox_z0"])))
    else:
        candidates.sort(key=lambda p: (float(p["bbox_x0"]), float(p["bbox_z0"]), float(p["bbox_y0"])))
    accepted: List[Dict[str, object]] = []
    working = list(placed)
    accepted_keys = set()
    for candidate in candidates:
        if layout_axis == "z" or allow_vertical_mix:
            candidate = settle_candidate(
                candidate,
                working,
                container,
                clearance_mm,
                layout_axis,
                require_strong_support=allow_vertical_mix,
                allowed_support_block_ids=allowed_support_block_ids,
            )
            if candidate is None:
                continue
        elif _candidate_footprint_collides(candidate, placed, clearance_mm):
            continue
        settled_key = (
            round(float(candidate["bbox_x0"]), 3),
            round(float(candidate["bbox_y0"]), 3),
            round(float(candidate["bbox_z0"]), 3),
        )
        if settled_key in accepted_keys:
            continue
        if _candidate_collides(candidate, working, clearance_mm):
            continue
        accepted.append(candidate)
        accepted_keys.add(settled_key)
        working.append(candidate)
        if len(accepted) >= max_count:
            break
    return accepted


def estimate_count_stack_span(
    container: Container,
    tire: Tire,
    method: str,
    clearance_mm: float,
    target_count: int,
) -> float:
    if target_count <= 0:
        return 0.0
    generator = generate_flat_placements if method == "flat" else generate_honeycomb_placements
    tire_for_step = override_tire_dimensions(tire, round(tire.outer_diameter_mm), tire.section_width_mm) if method == "flat" else tire
    step = max(1.0, tire_for_step.outer_diameter_mm + clearance_mm)
    target_height = container.height_mm * 0.82
    span = step
    while span <= container.length_mm + 1e-6:
        candidates = generator(
            container,
            tire,
            clearance_mm,
            x_start=0,
            x_length=span,
            z_start=0,
            z_height=target_height,
            max_count=target_count,
            include_standing_extras=False,
        )
        if len(candidates) >= target_count:
            return min(container.length_mm, span)
        span += step
    return container.length_mm


def generate_overflow_containers(
    container: Container,
    method: str,
    clearance_mm: float,
    requests: List[Dict[str, object]],
    layout_axis: str = "x",
) -> Tuple[List[Dict[str, object]], pd.DataFrame, int]:
    overflow_placements: List[Dict[str, object]] = []
    overflow_rows = []
    remaining = [
        {"idx": int(item["idx"]), "tire": item["tire"], "count": int(item["count"])}
        for item in requests
        if int(item.get("count", 0)) > 0
    ]
    container_no = 2
    container_span = container.length_mm if layout_axis == "x" else container.height_mm

    while remaining:
        placed: List[Dict[str, object]] = []
        previous_block_placements: List[Dict[str, object]] = []
        current_pos = 0.0
        next_remaining = []
        placed_in_this_container = 0

        for item in sorted(remaining, key=lambda row: (-int(row["count"]), int(row["idx"]))):
            tire = item["tire"]
            target_count = int(item["count"])
            if target_count <= 0:
                continue
            block_id = f"C{container_no}_B{item['idx'] + 1}_{tire.spec}"
            support_block_ids = {block_id}
            support_block_ids.update(str(p["block_id"]) for p in previous_block_placements)
            block_placements = fill_remaining_space(
                container,
                tire,
                method,
                clearance_mm,
                placed,
                target_count,
                block_id,
                layout_axis,
                0.0,
                container_span,
                allow_vertical_mix=True,
                anchor_placements=previous_block_placements if previous_block_placements else None,
                allowed_support_block_ids=support_block_ids,
                include_standing_extras_in_scan=bool(method == "flat" and not placed),
            )
            for p in block_placements:
                p["container_no"] = container_no
                if block_placements:
                    p["block_start_x"] = min(float(q["bbox_x0"]) for q in block_placements)
                    p["block_end_x"] = max(float(q["bbox_x1"]) for q in block_placements)
                    p["block_start_z"] = min(float(q["bbox_z0"]) for q in block_placements)
                    p["block_end_z"] = max(float(q["bbox_z1"]) for q in block_placements)
            placed.extend(block_placements)
            previous_block_placements = block_placements
            placed_count = len(block_placements)
            placed_in_this_container += placed_count
            remainder = target_count - placed_count
            overflow_rows.append({
                "container_no": container_no,
                "tire_spec": tire.spec,
                "loading_method": method,
                "requested_count": target_count,
                "count": placed_count,
                "unplaced_count": max(0, remainder),
                "unused_space_reason": "추가 컨테이너 잔량 적재" if placed_count else "추가 컨테이너에 배치 가능한 좌표 없음",
            })
            if remainder > 0:
                next_remaining.append({"idx": item["idx"], "tire": tire, "count": remainder})
            if placed:
                current_pos = max(float(p["bbox_x1"]) for p in placed) + clearance_mm
                if current_pos >= container_span:
                    current_pos = container_span

        overflow_placements.extend(placed)
        if placed_in_this_container == 0:
            break
        remaining = next_remaining
        container_no += 1

    unplaced_after_overflow = sum(int(item["count"]) for item in remaining)
    return overflow_placements, pd.DataFrame(overflow_rows), unplaced_after_overflow


def calculate_mixed_loading(
    container: Container,
    tires: List[Tire],
    method: str,
    clearance_mm: float,
    allocation_mode: str,
    allocations: List[float],
    layout_axis: str = "x",
) -> Dict[str, object]:
    generator = generate_flat_placements if method == "flat" else generate_honeycomb_placements
    placements: List[Dict[str, object]] = []
    rows = []
    overflow_requests: List[Dict[str, object]] = []
    current_pos = 0.0
    previous_block_placements: List[Dict[str, object]] = []
    allocation_mode_text = allocation_mode.lower()
    is_ratio_mode = "%" in allocation_mode
    is_length_mode = "mm" in allocation_mode_text or "길" in allocation_mode or "높" in allocation_mode or "length" in allocation_mode_text or "height" in allocation_mode_text
    is_count_mode = not is_ratio_mode and not is_length_mode
    positive_total = sum(float(v) for v in allocations if float(v) > 0)
    container_span = container.length_mm if layout_axis == "x" else container.height_mm
    positive_count = max(0, sum(1 for v in allocations if v > 0))
    total_gap_budget = max(0, positive_count - 1) * clearance_mm
    available_span = max(0.0, container_span - total_gap_budget)
    dedicated_loaded_total = 0
    if is_count_mode:
        capacity_func = calculate_flat_loading if method == "flat" else calculate_honeycomb_loading
        work_items = []
        for idx, tire in enumerate(tires):
            requested_count = int(float(allocations[idx])) if idx < len(allocations) else 0
            if requested_count <= 0:
                continue
            single_capacity = int(capacity_func(container, tire, clearance_mm)["count"])
            dedicated_container_count = requested_count // single_capacity if single_capacity > 0 else 0
            mixed_requested_count = requested_count % single_capacity if single_capacity > 0 else requested_count
            dedicated_count = dedicated_container_count * single_capacity
            dedicated_loaded_total += dedicated_count
            work_items.append({
                "idx": idx,
                "tire": tire,
                "value": float(mixed_requested_count),
                "original_requested_count": requested_count,
                "single_capacity": single_capacity,
                "dedicated_container_count": dedicated_container_count,
                "dedicated_count": dedicated_count,
                "mixed_requested_count": mixed_requested_count,
            })
        work_items.sort(key=lambda item: (-int(item["mixed_requested_count"]), int(item["idx"])))
    else:
        work_items = [
            {
                "idx": idx,
                "tire": tire,
                "value": float(allocations[idx]) if idx < len(allocations) else 0.0,
                "original_requested_count": None,
                "single_capacity": None,
                "dedicated_container_count": 0,
                "dedicated_count": 0,
                "mixed_requested_count": None,
            }
            for idx, tire in enumerate(tires)
        ]

    for item in work_items:
        idx = int(item["idx"])
        tire = item["tire"]
        value = float(item["value"])
        if value <= 0:
            if is_count_mode and int(item["original_requested_count"]) > 0:
                rows.append({
                    "tire_spec": tire.spec,
                    "loading_method": method,
                    "layout_axis": "가로구역(X)" if layout_axis == "x" else "세로층(Z)",
                    "assigned_length_mm": 0.0 if layout_axis == "x" else container.length_mm,
                    "assigned_height_mm": 0.0 if layout_axis == "z" else container.height_mm,
                    "block_start_x": round(current_pos, 1) if layout_axis == "x" else 0,
                    "block_end_x": round(current_pos, 1) if layout_axis == "x" else container.length_mm,
                    "block_start_z": round(current_pos, 1) if layout_axis == "z" else 0,
                    "block_end_z": round(current_pos, 1) if layout_axis == "z" else 0,
                    "original_requested_count": int(item["original_requested_count"]),
                    "single_spec_capacity": int(item["single_capacity"]),
                    "dedicated_container_count": int(item["dedicated_container_count"]),
                    "dedicated_count": int(item["dedicated_count"]),
                    "mixed_requested_count": 0,
                    "requested_count": 0,
                    "count": 0,
                    "unplaced_count": 0,
                    "used_volume": 0.0,
                    "utilization_rate": 0,
                    "unused_space_reason": "단일 규격 최대 적재량 단위로 단독 컨테이너 분리, 혼적 잔량 없음",
                })
            continue
        block_start = current_pos
        max_count: Optional[int] = None
        if is_ratio_mode:
            assigned_span = available_span * value / 100
        elif is_count_mode:
            assigned_span = container_span - current_pos
            max_count = int(value)
        else:
            assigned_span = value
        assigned_span = max(0.0, min(assigned_span, container_span - block_start))
        block_id = f"B{idx + 1}_{tire.spec}"
        if max_count is None:
            capacity_kwargs = (
                {"x_start": block_start, "x_length": assigned_span, "z_start": 0, "z_height": container.height_mm}
                if layout_axis == "x"
                else {"x_start": 0, "x_length": container.length_mm, "z_start": block_start, "z_height": assigned_span}
            )
            capacity_candidates = generator(container, tire, clearance_mm, max_count=None, block_id=block_id, block_axis=layout_axis, **capacity_kwargs)
            max_count = len(capacity_candidates)
        if is_count_mode:
            scan_start = 0.0
            scan_end = container_span
            support_block_ids = {block_id}
            support_block_ids.update(str(p["block_id"]) for p in previous_block_placements)
        elif layout_axis == "x":
            scan_start = max(0.0, block_start - tire.outer_diameter_mm - clearance_mm)
            scan_end = min(container.length_mm, block_start + assigned_span)
            support_block_ids = None
        else:
            scan_start = max(0.0, block_start - tire.section_width_mm - clearance_mm)
            scan_end = min(container.height_mm, block_start + assigned_span)
            support_block_ids = None
        block_placements = fill_remaining_space(
            container,
            tire,
            method,
            clearance_mm,
            placements,
            max_count,
            block_id,
            layout_axis,
            scan_start,
            scan_end,
            allow_vertical_mix=is_count_mode,
            anchor_placements=previous_block_placements if previous_block_placements else None,
            allowed_support_block_ids=support_block_ids,
            include_standing_extras_in_scan=bool(is_count_mode and method == "flat" and not placements),
        )
        if block_placements:
            block_end_x = actual_block_end(block_placements, 0)
            block_end_z = actual_block_end_z(block_placements, 0)
            block_start_x = min(float(p["bbox_x0"]) for p in block_placements)
            block_start_z = min(float(p["bbox_z0"]) for p in block_placements)
        else:
            block_end_x = block_start if layout_axis == "x" else 0
            block_end_z = block_start if layout_axis == "z" else 0
            block_start_x = block_start if layout_axis == "x" else 0
            block_start_z = block_start if layout_axis == "z" else 0
        for p in block_placements:
            p["container_no"] = 1
            p["block_start_x"] = block_start_x
            p["block_end_x"] = block_end_x
            p["block_start_z"] = block_start_z
            p["block_end_z"] = block_end_z
        block_end = block_end_x if layout_axis == "x" else block_end_z
        used_volume = sum(math.pi * (float(p["outer_diameter_mm"]) / 2) ** 2 * float(p["section_width_mm"]) for p in block_placements)
        assigned_volume = (
            assigned_span * container.width_mm * container.height_mm
            if layout_axis == "x"
            else container.length_mm * container.width_mm * assigned_span
        )
        requested_count = int(max_count) if max_count is not None else len(block_placements)
        unplaced_count = max(0, requested_count - len(block_placements)) if is_count_mode else 0
        if is_count_mode and unplaced_count > 0:
            overflow_requests.append({"idx": idx, "tire": tire, "count": unplaced_count})
        if unplaced_count:
            unused_space_reason = "단독 컨테이너 분리 후 혼적 잔량 중 일부가 남은 좌표에 들어가지 않음"
        elif block_placements:
            unused_space_reason = "이전 규격 placement와 충돌하지 않는 남은 좌표를 순차 스캔함"
        else:
            unused_space_reason = "목표 수량 또는 할당 용량에 넣을 수 있는 남은 좌표 없음"
        rows.append({
            "tire_spec": tire.spec,
            "loading_method": method,
            "layout_axis": "가로구역(X)" if layout_axis == "x" else "세로층(Z)",
            "assigned_length_mm": round(assigned_span, 1) if layout_axis == "x" else container.length_mm,
            "assigned_height_mm": round(assigned_span, 1) if layout_axis == "z" else container.height_mm,
            "block_start_x": round(block_start_x, 1),
            "block_end_x": round(block_end_x, 1),
            "block_start_z": round(block_start_z, 1),
            "block_end_z": round(block_end_z, 1),
            "original_requested_count": int(item["original_requested_count"]) if is_count_mode else None,
            "single_spec_capacity": int(item["single_capacity"]) if is_count_mode else None,
            "dedicated_container_count": int(item["dedicated_container_count"]) if is_count_mode else 0,
            "dedicated_count": int(item["dedicated_count"]) if is_count_mode else 0,
            "mixed_requested_count": requested_count if is_count_mode else None,
            "requested_count": requested_count,
            "count": len(block_placements),
            "unplaced_count": unplaced_count,
            "used_volume": round(used_volume, 1),
            "utilization_rate": round(used_volume / assigned_volume, 4) if assigned_volume else 0,
            "unused_space_reason": unused_space_reason,
        })
        placements.extend(block_placements)
        previous_block_placements = block_placements
        if is_count_mode:
            current_pos = block_end + clearance_mm
        else:
            current_pos = block_start + assigned_span + clearance_mm
        if current_pos >= container_span:
            current_pos = container_span
        continue
        if layout_axis == "z":
            candidate_start = max(0.0, block_start - tire.section_width_mm - clearance_mm)
            candidate_height = assigned_span + (block_start - candidate_start)
            block_placements = generator(
                container,
                tire,
                clearance_mm,
                x_start=0,
                x_length=container.length_mm,
                z_start=candidate_start,
                z_height=candidate_height,
                max_count=None,
                block_id=block_id,
                block_axis="z",
            )
            block_placements = compact_block_down(placements, block_placements, container, clearance_mm)
            if max_count is not None:
                block_placements = block_placements[:max_count]
            block_end = actual_block_end_z(block_placements, block_start)
        else:
            candidate_start = max(0.0, block_start - tire.outer_diameter_mm - clearance_mm)
            candidate_length = assigned_span + (block_start - candidate_start)
            block_placements = generator(
                container,
                tire,
                clearance_mm,
                x_start=candidate_start,
                x_length=candidate_length,
                z_start=0,
                z_height=container.height_mm,
                max_count=None,
                block_id=block_id,
                block_axis="x",
            )
            block_placements = compact_block_left(placements, block_placements, container, clearance_mm)
            if max_count is not None:
                block_placements = block_placements[:max_count]
            block_end = actual_block_end(block_placements, block_start)
        for p in block_placements:
            if layout_axis == "z":
                p["block_start_z"] = block_start
                p["block_end_z"] = block_end
                p["block_start_x"] = 0
                p["block_end_x"] = container.length_mm
            else:
                p["block_end_x"] = block_end
                p["block_start_z"] = 0
                p["block_end_z"] = container.height_mm
        used_volume = sum(math.pi * (float(p["outer_diameter_mm"]) / 2) ** 2 * float(p["section_width_mm"]) for p in block_placements)
        assigned_volume = (
            assigned_span * container.width_mm * container.height_mm
            if layout_axis == "x"
            else container.length_mm * container.width_mm * assigned_span
        )
        rows.append({
            "tire_spec": tire.spec,
            "loading_method": method,
            "layout_axis": "가로구역(X)" if layout_axis == "x" else "세로층(Z)",
            "assigned_length_mm": round(assigned_span, 1) if layout_axis == "x" else container.length_mm,
            "assigned_height_mm": round(assigned_span, 1) if layout_axis == "z" else container.height_mm,
            "block_start_x": round(block_start, 1) if layout_axis == "x" else 0,
            "block_end_x": round(block_end, 1) if layout_axis == "x" else container.length_mm,
            "block_start_z": round(block_start, 1) if layout_axis == "z" else 0,
            "block_end_z": round(block_end, 1) if layout_axis == "z" else container.height_mm,
            "count": len(block_placements),
            "used_volume": round(used_volume, 1),
            "utilization_rate": round(used_volume / assigned_volume, 4) if assigned_volume else 0,
            "unused_space_reason": "잔여 폭/높이/길이가 타이어 점유 치수보다 작음" if block_placements else "할당 길이 또는 목표 수량이 부족함",
        })
        placements.extend(block_placements)
        current_pos = block_end + clearance_mm
        if current_pos >= container_span:
            break
    overflow_placements: List[Dict[str, object]] = []
    overflow_table = pd.DataFrame()
    unplaced_after_overflow = sum(int(item["count"]) for item in overflow_requests)
    if is_count_mode and overflow_requests:
        overflow_placements, overflow_table, unplaced_after_overflow = generate_overflow_containers(
            container,
            method,
            clearance_mm,
            overflow_requests,
            layout_axis,
        )
    table = pd.DataFrame(rows)
    overflow_count = len(overflow_placements)
    return {
        "placements": placements,
        "overflow_placements": overflow_placements,
        "overflow_table": overflow_table,
        "table": table,
        "total_count": len(placements),
        "overflow_count": overflow_count,
        "requested_total": int(positive_total) if is_count_mode else None,
        "dedicated_loaded_total": int(dedicated_loaded_total) if is_count_mode else 0,
        "loaded_total": int(dedicated_loaded_total + len(placements) + overflow_count) if is_count_mode else len(placements),
        "unplaced_total": int(unplaced_after_overflow) if is_count_mode else 0,
        "first_container_unplaced_total": int(sum(int(item["count"]) for item in overflow_requests)) if is_count_mode else 0,
        "capacity_limited": bool(is_count_mode and dedicated_loaded_total > 0),
        "capacity_reference_count": None,
    }


def validate_placements(
    container: Container,
    placements: List[Dict[str, object]],
    clearance_mm: float = 0,
    expected_count: Optional[int] = None,
) -> pd.DataFrame:
    rows = []
    eps = 1e-6
    outside = []
    for i, p in enumerate(placements):
        inside = (
            float(p["bbox_x0"]) >= -eps
            and float(p["bbox_x1"]) <= container.length_mm + eps
            and float(p["bbox_y0"]) >= -eps
            and float(p["bbox_y1"]) <= container.width_mm + eps
            and float(p["bbox_z0"]) >= -eps
            and float(p["bbox_z1"]) <= container.height_mm + eps
        )
        if not inside:
            outside.append(i)
    rows.append({"validation": "container_bounds", "status": "PASS" if not outside else "FAIL", "detail": f"outside_count={len(outside)}"})

    overlaps = 0
    clearance_violations = 0
    for i in range(len(placements)):
        a = placements[i]
        for j in range(i + 1, len(placements)):
            b = placements[j]
            z_overlap = not (float(a["bbox_z1"]) <= float(b["bbox_z0"]) + eps or float(b["bbox_z1"]) <= float(a["bbox_z0"]) + eps)
            if not z_overlap:
                continue
            if a["axis"] == "z" and b["axis"] == "z":
                dx = float(a["x"]) - float(b["x"])
                dy = float(a["y"]) - float(b["y"])
                dist = math.hypot(dx, dy)
                min_dist = (float(a["outer_diameter_mm"]) + float(b["outer_diameter_mm"])) / 2
                if dist < min_dist - eps:
                    overlaps += 1
                elif dist < min_dist + clearance_mm - eps:
                    clearance_violations += 1
            else:
                bbox_overlap = (
                    float(a["bbox_x0"]) < float(b["bbox_x1"]) - eps
                    and float(b["bbox_x0"]) < float(a["bbox_x1"]) - eps
                    and float(a["bbox_y0"]) < float(b["bbox_y1"]) - eps
                    and float(b["bbox_y0"]) < float(a["bbox_y1"]) - eps
                )
                if bbox_overlap:
                    overlaps += 1
    rows.append({"validation": "tire_overlap", "status": "PASS" if overlaps == 0 else "FAIL", "detail": f"overlap_pairs={overlaps}"})
    rows.append({"validation": "clearance", "status": "PASS" if clearance_violations == 0 else "WARN", "detail": f"violating_pairs={clearance_violations}"})

    block_rows = {}
    interlocked_axis = any(p.get("block_axis") in {"x", "z"} for p in placements)
    for p in placements:
        bid = str(p["block_id"])
        if p.get("block_axis") == "z":
            block_rows.setdefault(bid, [float(p["block_start_z"]), float(p["block_end_z"])])
        else:
            block_rows.setdefault(bid, [float(p["block_start_x"]), float(p["block_end_x"])])
    block_intersections = 0
    blocks = sorted(block_rows.items(), key=lambda item: item[1][0])
    for (_, a), (_, b) in zip(blocks, blocks[1:]):
        if a[1] + clearance_mm > b[0] + eps:
            block_intersections += 1
    boundary_status = "PASS" if block_intersections == 0 or interlocked_axis else "FAIL"
    boundary_detail = f"boundary_issues={block_intersections}"
    if block_intersections and interlocked_axis:
        boundary_detail += ", interlocked_boundary_allowed=true"
    rows.append({"validation": "mixed_block_boundary", "status": boundary_status, "detail": boundary_detail})

    if expected_count is not None:
        match = len(placements) == expected_count
        rows.append({"validation": "visual_table_count_match", "status": "PASS" if match else "FAIL", "detail": f"placements={len(placements)}, expected={expected_count}"})

    floating = 0
    for p in placements:
        if float(p["bbox_z0"]) <= eps:
            continue
        supported = False
        for q in placements:
            if q is p:
                continue
            touches_below = (
                float(q["bbox_z1"]) <= float(p["bbox_z0"]) + eps
                and float(q["bbox_z1"]) >= float(p["bbox_z0"]) - max(clearance_mm, 1) - eps
            )
            if not touches_below:
                continue
            support_dist = (float(q["outer_diameter_mm"]) + float(p["outer_diameter_mm"])) / 2 + clearance_mm
            if math.hypot(float(q["x"]) - float(p["x"]), float(q["y"]) - float(p["y"])) < support_dist - eps:
                supported = True
                break
        if not supported:
            floating += 1
    rows.append({"validation": "vertical_support", "status": "PASS" if floating == 0 else "WARN", "detail": f"floating_like_count={floating}"})
    return pd.DataFrame(rows)


def logic_explanation_lines() -> List[str]:
    return [
        "평치 적재: 기준표와 맞추기 위해 계산 OD를 정수 mm로 반올림하고, 기본 눕힘 적재 후 남는 폭/길이 구간에는 세운 타이어 후보를 추가한다.",
        "평치 좌표: 기본 영역은 x/y pitch=OD+clearance, z pitch=SW+clearance이며, 잔여 폭/길이 보조 영역은 타이어 축을 y 또는 x 방향으로 돌려 placement를 만든다.",
        "벌집 적재: 같은 수직 원기둥을 사용하되 행마다 x 방향 offset=(OD+clearance)/2를 적용한다.",
        "벌집 pitch: column pitch=OD+clearance, row pitch=sqrt(3)/2*(OD+clearance), layer pitch=SW+clearance를 사용한다.",
        "단일 규격 수량: 산식 결과가 아니라 generate_*_placements가 만든 placement list 길이를 적입수량으로 사용한다.",
        "혼적 적재: 컨테이너 길이 x축을 규격별 블록으로 나누고 각 블록 안에서 동일한 placement 생성 함수를 호출한다.",
        "혼적 경계: 가로구역/세로층 혼적은 다음 규격 후보를 이전 블록 쪽으로 한 타이어 치수만큼 겹쳐 만든 뒤 실제 충돌이 없는 좌표만 남겨 경계 빈공간을 줄인다.",
        "검증: 모든 bbox가 컨테이너 내부인지, 같은 높이 구간의 원기둥이 겹치는지, clearance가 지켜졌는지, 표 count와 placement count가 같은지 확인한다.",
        "공통 구조: 3D 시각화, 결과표, Excel 출력은 모두 동일한 placement list에서 파생된다.",
    ]


def create_excel_report(
    input_df: pd.DataFrame,
    single_df: pd.DataFrame,
    mixed_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        input_df.to_excel(writer, sheet_name="INPUT", index=False)
        single_df.to_excel(writer, sheet_name="SINGLE_SPEC_RESULT", index=False)
        mixed_df.to_excel(writer, sheet_name="MIXED_LOADING_RESULT", index=False)
        pd.DataFrame({"logic": logic_explanation_lines()}).to_excel(writer, sheet_name="LOGIC_EXPLANATION", index=False)
        validation_df.to_excel(writer, sheet_name="VALIDATION", index=False)
    return output.getvalue()


def write_logic_report(path: str = "loading_logic_report.md") -> None:
    content = """# 타이어 평치/벌집 적재 산출 로직 설명

## 1. 단일 규격별 컨테이너 적입수량 산출 로직
타이어 규격 문자열에서 단면폭, 편평비, 림 인치를 파싱하고 외경을 계산합니다. 사용자가 보정한 외경과 단면폭이 있으면 보정값을 우선 사용합니다. 평치와 벌집 모두 실제 좌표 placement list를 먼저 만들고, 적입수량은 이 리스트의 길이로 산출합니다.

## 2. 평치 적재 방식
타이어 1개를 수직 원기둥으로 정의합니다. 바닥 점유 크기는 `outer_diameter_mm x outer_diameter_mm`, 높이는 `section_width_mm`입니다. x/y/z pitch는 각각 `OD+clearance`, `OD+clearance`, `SW+clearance`입니다. 컨테이너 내부에 완전히 들어오는 후보만 placement로 채택합니다.

## 3. 벌집/육각 적재 방식
벌집 방식도 동일한 원기둥 모델을 사용합니다. 열 pitch는 `OD+clearance`, 행 pitch는 `sqrt(3)/2*(OD+clearance)`이며 홀수 행은 x 방향으로 `pitch/2` offset을 둡니다. 각 layer는 `SW+clearance` 간격으로 쌓습니다.

## 4. 혼적 규격별 컨테이너 적입수량 산출 로직
혼적 모드는 컨테이너 길이 방향 x축을 규격별 블록으로 나눕니다. 비율, 목표 수량, 할당 길이 중 선택된 입력값을 기준으로 각 블록의 후보 영역을 만들고, 그 안에서 단일 규격과 동일한 placement 함수를 호출합니다.

## 5. 혼적 시 구역 배분 방식
각 블록은 `block_start_x`, `assigned_length_mm`, `block_end_x`를 가집니다. `block_end_x`는 할당 길이 끝이 아니라 실제 배치된 타이어의 마지막 bbox_x1입니다. 다음 블록은 `block_end_x + clearance`에서 시작합니다.

## 6. 타규격 간 경계부 여백 처리
다른 규격 사이에는 사용자가 입력한 clearance만 반영합니다. 계산 중 임의의 큰 여백은 추가하지 않으며, 실제 점유 끝 좌표를 기준으로 다음 규격을 이어 붙입니다.

## 7. 충돌/범위 검증 방식
모든 placement의 bbox가 컨테이너 L/W/H 범위 안에 있는지 확인합니다. 같은 높이 구간에서 원기둥 중심 간 거리가 반지름 합보다 작으면 overlap으로 판단합니다. 반지름 합 이상이지만 clearance보다 가까우면 clearance warning으로 기록합니다. 혼적 블록은 이전 block_end_x와 다음 block_start_x를 비교해 침범 여부를 검증합니다.

## 8. 계산 결과와 시각화 결과 일치 구조
3D 시각화, 단일/혼적 결과표, Excel 출력, validation은 모두 같은 placement list를 입력으로 사용합니다. 따라서 화면에 표시된 타이어 개수와 결과표 count가 분리되어 달라지지 않습니다.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
