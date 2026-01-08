from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import xml.etree.ElementTree as ET
import yaml


@dataclass(frozen=True)
class WorldSpec:
    width: float
    height: float


@dataclass(frozen=True)
class WorldRectPx:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class WorldInfo:
    size: WorldSpec
    rect_px: WorldRectPx


@dataclass(frozen=True)
class WallSegment:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class BoxObstacle:
    xmin: float
    xmax: float
    ymin: float
    ymax: float


def _load_xml(path: Path) -> ET.Element:
    tree = ET.parse(path)
    return tree.getroot()


def _find_cell(root: ET.Element, cell_id: str) -> Optional[ET.Element]:
    for cell in root.iter("mxCell"):
        if cell.attrib.get("id") == cell_id:
            return cell
    return None


def _cell_value_as_float(cell: ET.Element, *, name: str) -> float:
    raw = cell.attrib.get("value", "")
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} value must be a float, got '{raw}'.") from exc


def _cell_geometry(cell: ET.Element) -> WorldRectPx:
    geom = cell.find("mxGeometry")
    if geom is None:
        raise ValueError(f"Cell id={cell.attrib.get('id')} missing mxGeometry.")
    x = float(geom.attrib.get("x", "0"))
    y = float(geom.attrib.get("y", "0"))
    width = float(geom.attrib["width"])
    height = float(geom.attrib["height"])
    return WorldRectPx(x=x, y=y, width=width, height=height)


def _parse_world_info(
    root: ET.Element,
    *,
    world_cell_id: str,
    world_height_id: str,
    world_width_id: str,
) -> WorldInfo:
    world_cell = _find_cell(root, world_cell_id)
    if world_cell is None:
        raise ValueError(f"World cell id={world_cell_id} not found.")
    world_rect = _cell_geometry(world_cell)

    height_cell = _find_cell(root, world_height_id)
    if height_cell is None:
        raise ValueError(f"World height cell id={world_height_id} not found.")
    width_cell = _find_cell(root, world_width_id)
    if width_cell is None:
        raise ValueError(f"World width cell id={world_width_id} not found.")

    world_height = _cell_value_as_float(height_cell, name="world height")
    world_width = _cell_value_as_float(width_cell, name="world width")
    return WorldInfo(
        size=WorldSpec(width=world_width, height=world_height),
        rect_px=world_rect,
    )


def parse_world_info(
    drawio_path: Path,
    *,
    world_cell_id: str = "4",
    world_height_id: str = "5",
    world_width_id: str = "6",
) -> WorldInfo:
    root = _load_xml(drawio_path)
    return _parse_world_info(
        root,
        world_cell_id=world_cell_id,
        world_height_id=world_height_id,
        world_width_id=world_width_id,
    )


def _point_in_world_px(x: float, y: float, rect: WorldRectPx) -> bool:
    return rect.x <= x <= rect.x + rect.width and rect.y <= y <= rect.y + rect.height


def _rect_in_world_px(geom: WorldRectPx, rect: WorldRectPx) -> bool:
    return (
        rect.x <= geom.x
        and rect.y <= geom.y
        and geom.x + geom.width <= rect.x + rect.width
        and geom.y + geom.height <= rect.y + rect.height
    )


def _px_to_world(x: float, y: float, world: WorldInfo) -> tuple[float, float]:
    ux = (x - world.rect_px.x) / world.rect_px.width
    uy = (y - world.rect_px.y) / world.rect_px.height
    wx = ux * world.size.width
    wy = (1.0 - uy) * world.size.height
    return wx, wy


def _cell_center_world(cell: ET.Element, world: WorldInfo, *, name: str) -> tuple[float, float]:
    geom = cell.find("mxGeometry")
    if geom is None:
        raise ValueError(f"{name} cell missing mxGeometry.")
    x = float(geom.attrib.get("x", "0"))
    y = float(geom.attrib.get("y", "0"))
    width = float(geom.attrib.get("width", "0"))
    height = float(geom.attrib.get("height", "0"))
    cx = x + width / 2.0
    cy = y + height / 2.0
    if not _point_in_world_px(cx, cy, world.rect_px):
        raise ValueError(f"{name} must be inside world rectangle.")
    return _px_to_world(cx, cy, world)


def _round3(value: float) -> float:
    return round(value, 3)


def parse_start_goal(
    root: ET.Element,
    world: WorldInfo,
    *,
    start_id: str,
    goal_id: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    start_cell = _find_cell(root, start_id)
    if start_cell is None:
        raise ValueError(f"Start cell id={start_id} not found.")
    goal_cell = _find_cell(root, goal_id)
    if goal_cell is None:
        raise ValueError(f"Goal cell id={goal_id} not found.")
    start = _cell_center_world(start_cell, world, name="start")
    goal = _cell_center_world(goal_cell, world, name="goal")
    return start, goal


def parse_obstacles(
    root: ET.Element,
    world: WorldInfo,
    *,
    exclude_ids: Iterable[str] = (),
) -> tuple[list[WallSegment], list[BoxObstacle]]:
    exclude = set(exclude_ids)
    walls: list[WallSegment] = []
    boxes: list[BoxObstacle] = []

    for cell in root.iter("mxCell"):
        cell_id = cell.attrib.get("id")
        if cell_id in exclude:
            continue

        if cell.attrib.get("edge") == "1":
            geom = cell.find("mxGeometry")
            if geom is None:
                continue
            src = geom.find("mxPoint[@as='sourcePoint']")
            tgt = geom.find("mxPoint[@as='targetPoint']")
            if src is None or tgt is None:
                continue
            x0 = float(src.attrib["x"])
            y0 = float(src.attrib["y"])
            x1 = float(tgt.attrib["x"])
            y1 = float(tgt.attrib["y"])
            if not (
                _point_in_world_px(x0, y0, world.rect_px)
                and _point_in_world_px(x1, y1, world.rect_px)
            ):
                continue
            wx0, wy0 = _px_to_world(x0, y0, world)
            wx1, wy1 = _px_to_world(x1, y1, world)
            walls.append(WallSegment(x0=wx0, y0=wy0, x1=wx1, y1=wy1))
            continue

        if cell.attrib.get("vertex") == "1":
            geom_px = _cell_geometry(cell)
            if not _rect_in_world_px(geom_px, world.rect_px):
                continue
            x0, y0 = _px_to_world(geom_px.x, geom_px.y + geom_px.height, world)
            x1, y1 = _px_to_world(geom_px.x + geom_px.width, geom_px.y, world)
            xmin, xmax = min(x0, x1), max(x0, x1)
            ymin, ymax = min(y0, y1), max(y0, y1)
            boxes.append(BoxObstacle(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax))

    return walls, boxes


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse drawio and emit layout.yaml."
    )
    default_path = Path(__file__).resolve().parent / "layout.drawio"
    parser.add_argument(
        "drawio",
        nargs="?",
        default=str(default_path),
        help="Path to layout.drawio",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "layout.yaml"),
        help="Path to output layout.yaml",
    )
    parser.add_argument("--nx", type=int, default=22, help="Grid cells in x.")
    parser.add_argument("--ny", type=int, default=22, help="Grid cells in y.")
    parser.add_argument(
        "--goal-radius",
        type=float,
        default=0.05,
        help="Goal radius in world units.",
    )
    parser.add_argument(
        "--wall-thickness",
        type=float,
        default=0.05,
        help="Common wall thickness.",
    )
    parser.add_argument(
        "--world-id",
        default="4",
        help="Cell id for the world rectangle.",
    )
    parser.add_argument(
        "--world-height-id",
        default="5",
        help="Cell id for the world height label.",
    )
    parser.add_argument(
        "--world-width-id",
        default="6",
        help="Cell id for the world width label.",
    )
    parser.add_argument(
        "--start-id",
        default="17",
        help="Cell id for the start marker.",
    )
    parser.add_argument(
        "--goal-id",
        default="16",
        help="Cell id for the goal marker.",
    )
    args = parser.parse_args()

    root = _load_xml(Path(args.drawio))
    info = _parse_world_info(
        root,
        world_cell_id=args.world_id,
        world_height_id=args.world_height_id,
        world_width_id=args.world_width_id,
    )
    walls, boxes = parse_obstacles(
        root,
        info,
        exclude_ids=(
            args.world_id,
            args.world_height_id,
            args.world_width_id,
            args.start_id,
            args.goal_id,
        ),
    )
    start, goal = parse_start_goal(
        root,
        info,
        start_id=args.start_id,
        goal_id=args.goal_id,
    )

    layout: dict[str, object] = {
        "world": {
            "xlim": [0.0, _round3(info.size.width)],
            "ylim": [0.0, _round3(info.size.height)],
        },
        "start": [_round3(start[0]), _round3(start[1])],
        "goal": [_round3(goal[0]), _round3(goal[1])],
        "obstacles": [],
    }
    obstacles: list[dict[str, object]] = []
    for idx, wall in enumerate(walls, start=1):
        obstacles.append(
            {
                "name": f"wall{idx}",
                "type": "wall",
                "x0": _round3(wall.x0),
                "y0": _round3(wall.y0),
                "x1": _round3(wall.x1),
                "y1": _round3(wall.y1),
            }
        )
    for idx, box in enumerate(boxes, start=1):
        obstacles.append(
            {
                "name": f"box{idx}",
                "type": "box",
                "xmin": _round3(box.xmin),
                "xmax": _round3(box.xmax),
                "ymin": _round3(box.ymin),
                "ymax": _round3(box.ymax),
            }
        )
    layout["obstacles"] = obstacles

    output_path = Path(args.output)
    output_path.write_text(
        yaml.safe_dump(layout, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
