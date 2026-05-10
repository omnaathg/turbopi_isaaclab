"""Procedural mountain cliff road scene for TurboPi visual experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

import isaaclab.sim as sim_utils


SKY_TEXTURE_PATH = Path(__file__).resolve().parents[1] / "assets" / "generated" / "mountain_sky_latlong.png"


@dataclass(frozen=True)
class MountainCliffSceneCfg:
    """Configuration for a narrow mountain shelf road with a cliff drop."""

    map_name: str = "figure8"
    road_width: float = 0.48
    road_thickness: float = 0.055
    road_z: float = 0.82
    start_height: float = 0.040
    lower_terrain_z: float = -0.42
    shoulder_width: float = 0.10
    rail_height: float = 0.12
    scene_extent: float = 5.0
    start_offset: float = 0.32


ORIGINAL_ROAD_CENTERLINE: tuple[tuple[float, float], ...] = (
    (-1.70, -1.18),
    (-1.14, -0.92),
    (-0.65, -0.54),
    (-0.28, -0.05),
    (0.28, 0.30),
    (0.94, 0.66),
    (1.46, 1.15),
    (1.78, 1.72),
    (1.64, 2.34),
    (1.08, 2.88),
    (0.36, 3.24),
    (-0.34, 3.70),
)

ORIGINAL_RIGHT_BRANCH_CENTERLINE: tuple[tuple[float, float], ...] = (
    (1.42, 1.10),
    (2.10, 1.20),
    (2.72, 0.92),
    (3.24, 0.46),
    (3.66, -0.16),
    (3.86, -0.88),
)

FIGURE8_COMMON_ARM_CENTERLINE: tuple[tuple[float, float], ...] = (
    (0.00, -1.45),
    (0.00, -0.68),
    (0.00, 0.10),
    (0.00, 0.88),
)

FIGURE8_LEFT_LOOP_CENTERLINE: tuple[tuple[float, float], ...] = (
    (0.00, 0.88),
    (-0.91, 0.72),
    (-1.58, 0.30),
    (-1.82, -0.28),
    (-1.58, -0.87),
    (-0.91, -1.29),
    (0.00, -1.45),
)

FIGURE8_RIGHT_LOOP_CENTERLINE: tuple[tuple[float, float], ...] = (
    (0.00, 0.88),
    (0.91, 0.72),
    (1.58, 0.30),
    (1.82, -0.28),
    (1.58, -0.87),
    (0.91, -1.29),
    (0.00, -1.45),
)

FIGURE8_LEFT_VISUAL_CENTERLINE: tuple[tuple[float, float], ...] = (
    *FIGURE8_COMMON_ARM_CENTERLINE,
    *FIGURE8_LEFT_LOOP_CENTERLINE[1:],
)

FIGURE8_RIGHT_VISUAL_CENTERLINE: tuple[tuple[float, float], ...] = (
    *FIGURE8_RIGHT_LOOP_CENTERLINE,
)

FIGURE8_ROAD_CENTERLINE: tuple[tuple[float, float], ...] = (
    *FIGURE8_COMMON_ARM_CENTERLINE,
    *FIGURE8_LEFT_LOOP_CENTERLINE[1:],
)

FIGURE8_RIGHT_BRANCH_CENTERLINE: tuple[tuple[float, float], ...] = (
    *FIGURE8_COMMON_ARM_CENTERLINE,
    *FIGURE8_RIGHT_LOOP_CENTERLINE[1:],
)


ROAD_CENTERLINE = ORIGINAL_ROAD_CENTERLINE
RIGHT_BRANCH_CENTERLINE = ORIGINAL_RIGHT_BRANCH_CENTERLINE
VALID_MAP_NAMES = ("original", "figure8")


@dataclass(frozen=True)
class RoadMap:
    road_centerline: tuple[tuple[float, float], ...]
    right_branch_centerline: tuple[tuple[float, float], ...]
    left_visual_centerline: tuple[tuple[float, float], ...]
    right_visual_centerline: tuple[tuple[float, float], ...]
    common_arm_centerline: tuple[tuple[float, float], ...]


def _normalize_map_name(map_name: str) -> str:
    if map_name not in VALID_MAP_NAMES:
        raise ValueError(f"Unknown map {map_name!r}. Expected one of: {', '.join(VALID_MAP_NAMES)}")
    return map_name


def road_map(scene_cfg: MountainCliffSceneCfg) -> RoadMap:
    if _normalize_map_name(scene_cfg.map_name) == "figure8":
        return RoadMap(
            road_centerline=FIGURE8_ROAD_CENTERLINE,
            right_branch_centerline=FIGURE8_RIGHT_BRANCH_CENTERLINE,
            left_visual_centerline=FIGURE8_LEFT_VISUAL_CENTERLINE,
            right_visual_centerline=FIGURE8_RIGHT_VISUAL_CENTERLINE,
            common_arm_centerline=FIGURE8_COMMON_ARM_CENTERLINE,
        )
    return RoadMap(
        road_centerline=ORIGINAL_ROAD_CENTERLINE,
        right_branch_centerline=ORIGINAL_RIGHT_BRANCH_CENTERLINE,
        left_visual_centerline=ORIGINAL_ROAD_CENTERLINE,
        right_visual_centerline=ORIGINAL_RIGHT_BRANCH_CENTERLINE,
        common_arm_centerline=ORIGINAL_ROAD_CENTERLINE[:7],
    )


def route_waypoints(scene_cfg: MountainCliffSceneCfg, task_name: str) -> tuple[tuple[float, float], ...]:
    paths = road_map(scene_cfg)
    if scene_cfg.map_name == "figure8":
        return paths.road_centerline if task_name == "go_left" else paths.right_branch_centerline

    start_position, _ = start_pose(scene_cfg)
    start_xy = (float(start_position[0]), float(start_position[1]))
    if task_name == "go_left":
        return (start_xy, *paths.road_centerline[1:])
    return (start_xy, *paths.road_centerline[1:7], *paths.right_branch_centerline[1:])


def _preview(color: tuple[float, float, float], roughness: float = 0.85) -> sim_utils.PreviewSurfaceCfg:
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=roughness)


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def _cuboid(
    prim_path: str,
    *,
    size: tuple[float, float, float],
    translation: tuple[float, float, float],
    color: tuple[float, float, float],
    collision: bool = False,
    roughness: float = 0.85,
    yaw: float = 0.0,
) -> None:
    cfg = sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg() if collision else None,
        physics_material=(
            sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.15,
                dynamic_friction=0.95,
                restitution=0.0,
            )
            if collision
            else None
        ),
        visual_material=_preview(color, roughness),
    )
    cfg.func(prim_path, cfg, translation=translation, orientation=_yaw_to_quat(yaw))


def _cylinder(
    prim_path: str,
    *,
    radius: float,
    height: float,
    translation: tuple[float, float, float],
    color: tuple[float, float, float],
    collision: bool = False,
    roughness: float = 0.85,
) -> None:
    cfg = sim_utils.CylinderCfg(
        radius=radius,
        height=height,
        collision_props=sim_utils.CollisionPropertiesCfg() if collision else None,
        visual_material=_preview(color, roughness),
    )
    cfg.func(prim_path, cfg, translation=translation)


def _cone(
    prim_path: str,
    *,
    radius: float,
    height: float,
    translation: tuple[float, float, float],
    color: tuple[float, float, float],
    collision: bool = False,
    roughness: float = 0.85,
) -> None:
    cfg = sim_utils.ConeCfg(
        radius=radius,
        height=height,
        collision_props=sim_utils.CollisionPropertiesCfg() if collision else None,
        visual_material=_preview(color, roughness),
    )
    cfg.func(prim_path, cfg, translation=translation)


def _sphere(
    prim_path: str,
    *,
    radius: float,
    translation: tuple[float, float, float],
    color: tuple[float, float, float],
    collision: bool = False,
    roughness: float = 0.85,
) -> None:
    cfg = sim_utils.SphereCfg(
        radius=radius,
        collision_props=sim_utils.CollisionPropertiesCfg() if collision else None,
        visual_material=_preview(color, roughness),
    )
    cfg.func(prim_path, cfg, translation=translation)


def _mesh_grid(
    prim_path: str,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    nx: int,
    ny: int,
    height_fn: Callable[[float, float], float],
    color: tuple[float, float, float],
    roughness: float = 0.92,
    collision: bool = False,
) -> None:
    """Create a static terrain mesh from a height function."""
    import isaacsim.core.utils.prims as prim_utils

    xs = np.linspace(x_range[0], x_range[1], nx, dtype=np.float32)
    ys = np.linspace(y_range[0], y_range[1], ny, dtype=np.float32)
    points: list[tuple[float, float, float]] = []
    for y in ys:
        for x in xs:
            points.append((float(x), float(y), float(height_fn(float(x), float(y)))))

    faces: list[int] = []
    counts: list[int] = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            a = iy * nx + ix
            b = a + 1
            c = a + nx
            d = c + 1
            faces.extend((a, c, b, b, c, d))
            counts.extend((3, 3))

    prim_utils.create_prim(prim_path, "Xform")
    mesh_prim = prim_utils.create_prim(
        f"{prim_path}/mesh",
        "Mesh",
        attributes={
            "points": points,
            "faceVertexIndices": np.asarray(faces, dtype=np.int32),
            "faceVertexCounts": np.asarray(counts, dtype=np.int32),
            "subdivisionScheme": "bilinear",
        },
    )
    material_path = f"{prim_path}/visualMaterial"
    material = _preview(color, roughness)
    material.func(material_path, material)
    sim_utils.bind_visual_material(mesh_prim.GetPrimPath(), material_path)
    if collision:
        mesh_path = mesh_prim.GetPrimPath()
        sim_utils.define_collision_properties(mesh_path, sim_utils.CollisionPropertiesCfg(collision_enabled=True))
        sim_utils.define_mesh_collision_properties(mesh_path, sim_utils.TriangleMeshPropertiesCfg())


def _segment_geometry(
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float, float, float, float]:
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length = math.hypot(dx, dy)
    yaw = math.atan2(-dx, dy)
    return 0.5 * (sx + ex), 0.5 * (sy + ey), dx / length, dy / length, yaw


def _offset_point(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    t: float,
    side_sign: float,
    offset: float,
) -> tuple[float, float]:
    sx, sy = start
    ex, ey = end
    _, _, ux, uy, _ = _segment_geometry(start, end)
    x = sx + t * (ex - sx)
    y = sy + t * (ey - sy)
    return x - uy * side_sign * offset, y + ux * side_sign * offset


def _spawn_boulder_cluster(
    prim_prefix: str,
    *,
    x: float,
    y: float,
    z: float,
    scale: float,
    color: tuple[float, float, float],
    collision: bool = True,
) -> None:
    offsets = (
        (0.00, 0.00, 0.00, 1.00),
        (0.08, -0.04, 0.02, 0.72),
        (-0.06, 0.05, 0.01, 0.58),
        (0.02, 0.09, 0.04, 0.46),
    )
    for idx, (ox, oy, oz, radius_scale) in enumerate(offsets):
        shade = tuple(max(0.02, min(1.0, c + 0.035 * ((idx % 2) - 0.5))) for c in color)
        _sphere(
            f"{prim_prefix}Lump{idx:02d}",
            radius=scale * radius_scale,
            translation=(x + scale * ox, y + scale * oy, z + scale * (0.74 * radius_scale + oz)),
            color=shade,
            collision=collision,
            roughness=0.98,
        )
    _cuboid(
        f"{prim_prefix}Slab",
        size=(scale * 1.15, scale * 0.42, scale * 0.28),
        translation=(x - scale * 0.02, y + scale * 0.04, z + scale * 0.20),
        color=tuple(max(0.02, c - 0.045) for c in color),
        collision=collision,
        roughness=0.98,
        yaw=0.62,
    )


def _spawn_road(scene_cfg: MountainCliffSceneCfg) -> None:
    paths = road_map(scene_cfg)
    road_color = (0.18, 0.13, 0.085)
    dust_color = (0.32, 0.24, 0.16)
    edge_color = (0.39, 0.34, 0.27)
    gravel_color = (0.40, 0.36, 0.29)
    deck_z = scene_cfg.road_z - 0.5 * scene_cfg.road_thickness
    surface_z = scene_cfg.road_z + 0.008
    mark_z = scene_cfg.road_z + 0.015

    for idx, (start, end) in enumerate(zip(paths.left_visual_centerline[:-1], paths.left_visual_centerline[1:], strict=False)):
        cx, cy, ux, uy, yaw = _segment_geometry(start, end)
        length = math.dist(start, end)
        total_width = scene_cfg.road_width + 2.0 * scene_cfg.shoulder_width
        _cuboid(
            f"/World/MountainCliffRoad/RoadRockShelf{idx:02d}",
            size=(total_width + 0.22, length + 0.12, 0.16),
            translation=(cx, cy, scene_cfg.road_z - 0.145),
            color=(0.29, 0.25, 0.20),
            collision=True,
            roughness=0.99,
            yaw=yaw,
        )
        _cuboid(
            f"/World/MountainCliffRoad/RoadSegment{idx:02d}",
            size=(total_width, length + 0.06, scene_cfg.road_thickness),
            translation=(cx, cy, deck_z),
            color=dust_color,
            collision=True,
            roughness=0.96,
            yaw=yaw,
        )
        _cuboid(
            f"/World/MountainCliffRoad/TravelSurface{idx:02d}",
            size=(scene_cfg.road_width, length + 0.05, 0.006),
            translation=(cx, cy, surface_z),
            color=road_color,
            collision=False,
            roughness=0.98,
            yaw=yaw,
        )
        if idx not in {4, 5, 6}:
            for side, side_sign in (("Left", -1.0), ("Right", 1.0)):
                ox = -uy * side_sign * (0.5 * scene_cfg.road_width - 0.025)
                oy = ux * side_sign * (0.5 * scene_cfg.road_width - 0.025)
                _cuboid(
                    f"/World/MountainCliffRoad/{side}GravelEdge{idx:02d}",
                    size=(0.030, length - 0.030, 0.007),
                    translation=(cx + ox, cy + oy, mark_z),
                    color=edge_color,
                    collision=False,
                    roughness=0.95,
                    yaw=yaw,
                )
        for dash_idx in range(max(1, int(length / 0.22))):
            t = (dash_idx + 0.35) / max(1, int(length / 0.22))
            sx, sy = _offset_point(start, end, t=t, side_sign=(-1.0 if dash_idx % 2 else 1.0), offset=0.18)
            _cuboid(
                f"/World/MountainCliffRoad/ShoulderGravel{idx:02d}_{dash_idx:02d}",
                size=(0.032 + 0.008 * (dash_idx % 2), 0.052 + 0.014 * (dash_idx % 3), 0.006),
                translation=(sx, sy, mark_z + 0.001),
                color=gravel_color,
                collision=False,
                roughness=0.98,
                yaw=yaw + 0.35 * ((dash_idx % 3) - 1),
            )

    for idx, point in enumerate(paths.left_visual_centerline[1:-1], start=1):
        _cylinder(
            f"/World/MountainCliffRoad/CurvePatch{idx:02d}",
            radius=0.5 * (scene_cfg.road_width + 2.0 * scene_cfg.shoulder_width),
            height=scene_cfg.road_thickness,
            translation=(point[0], point[1], deck_z),
            color=dust_color,
            collision=True,
            roughness=0.96,
        )
        _cylinder(
            f"/World/MountainCliffRoad/CurveSurface{idx:02d}",
            radius=0.5 * scene_cfg.road_width,
            height=0.006,
            translation=(point[0], point[1], surface_z),
            color=road_color,
            collision=False,
            roughness=0.98,
        )


def _spawn_right_branch(scene_cfg: MountainCliffSceneCfg) -> None:
    """Spawn the right-hand loop from the shared arm junction."""
    paths = road_map(scene_cfg)
    road_color = (0.18, 0.13, 0.085)
    dust_color = (0.31, 0.23, 0.15)
    edge_color = (0.38, 0.33, 0.26)
    deck_z = scene_cfg.road_z - 0.5 * scene_cfg.road_thickness
    surface_z = scene_cfg.road_z + 0.008
    mark_z = scene_cfg.road_z + 0.015

    for idx, (start, end) in enumerate(zip(paths.right_visual_centerline[:-1], paths.right_visual_centerline[1:], strict=False)):
        cx, cy, ux, uy, yaw = _segment_geometry(start, end)
        length = math.dist(start, end)
        total_width = scene_cfg.road_width + 2.0 * scene_cfg.shoulder_width
        _cuboid(
            f"/World/MountainCliffRoad/RightBranchRockShelf{idx:02d}",
            size=(total_width + 0.18, length + 0.10, 0.18),
            translation=(cx, cy, scene_cfg.road_z - 0.15),
            color=(0.28, 0.24, 0.19),
            collision=True,
            roughness=0.99,
            yaw=yaw,
        )
        _cuboid(
            f"/World/MountainCliffRoad/RightBranchRoadSegment{idx:02d}",
            size=(total_width, length + 0.05, scene_cfg.road_thickness),
            translation=(cx, cy, deck_z),
            color=dust_color,
            collision=True,
            roughness=0.97,
            yaw=yaw,
        )
        _cuboid(
            f"/World/MountainCliffRoad/RightBranchSurface{idx:02d}",
            size=(scene_cfg.road_width, length + 0.04, 0.006),
            translation=(cx, cy, surface_z),
            color=road_color,
            collision=False,
            roughness=0.98,
            yaw=yaw,
        )
        if idx > 1:
            for side, side_sign in (("Left", -1.0), ("Right", 1.0)):
                ox = -uy * side_sign * (0.5 * scene_cfg.road_width - 0.025)
                oy = ux * side_sign * (0.5 * scene_cfg.road_width - 0.025)
                _cuboid(
                    f"/World/MountainCliffRoad/RightBranch{side}Edge{idx:02d}",
                    size=(0.026, max(0.10, length - 0.04), 0.007),
                    translation=(cx + ox, cy + oy, mark_z),
                    color=edge_color,
                    collision=False,
                    roughness=0.96,
                    yaw=yaw,
                )
    for idx, point in enumerate(paths.right_visual_centerline[:-1]):
        _cylinder(
            f"/World/MountainCliffRoad/RightBranchCurvePatch{idx:02d}",
            radius=0.5 * (scene_cfg.road_width + 2.0 * scene_cfg.shoulder_width),
            height=scene_cfg.road_thickness,
            translation=(point[0], point[1], deck_z),
            color=dust_color,
            collision=True,
            roughness=0.97,
        )
        _cylinder(
            f"/World/MountainCliffRoad/RightBranchCurveSurface{idx:02d}",
            radius=0.5 * scene_cfg.road_width,
            height=0.006,
            translation=(point[0], point[1], surface_z),
            color=road_color,
            collision=False,
            roughness=0.98,
        )


def _spawn_guard_rails(scene_cfg: MountainCliffSceneCfg) -> None:
    """Legacy shelf-road rails are disabled for the figure-8 track.

    The old one-sided rail followed ROAD_CENTERLINE directly. On the closed
    loop geometry, that side alternates between outer and inner edges, which
    can place posts across the drivable path at bends.
    """
    if scene_cfg.map_name == "figure8":
        return

    paths = road_map(scene_cfg)
    post_color = (0.36, 0.28, 0.20)
    rail_color = (0.47, 0.45, 0.40)
    for idx, (start, end) in enumerate(zip(paths.road_centerline[:-1], paths.road_centerline[1:], strict=False)):
        cx, cy, ux, uy, yaw = _segment_geometry(start, end)
        length = math.dist(start, end)
        # Put rails on the valley side of the shelf road, with gaps at turns.
        side_sign = 1.0
        offset = 0.5 * scene_cfg.road_width + scene_cfg.shoulder_width + 0.035
        ox = -uy * side_sign * offset
        oy = ux * side_sign * offset
        _cuboid(
            f"/World/MountainCliffRoad/GuardRail{idx:02d}",
            size=(0.030, max(0.10, length - 0.16), 0.024),
            translation=(cx + ox, cy + oy, scene_cfg.road_z + scene_cfg.rail_height),
            color=rail_color,
            collision=True,
            roughness=0.75,
            yaw=yaw,
        )
        _cuboid(
            f"/World/MountainCliffRoad/LowerGuardRail{idx:02d}",
            size=(0.024, max(0.08, length - 0.22), 0.018),
            translation=(cx + ox, cy + oy, scene_cfg.road_z + 0.060),
            color=(0.32, 0.31, 0.28),
            collision=True,
            roughness=0.85,
            yaw=yaw,
        )
        post_count = max(2, int(length / 0.24))
        for post_idx in range(post_count):
            t = (post_idx + 0.5) / post_count
            px = start[0] + t * (end[0] - start[0]) + ox
            py = start[1] + t * (end[1] - start[1]) + oy
            _cuboid(
                f"/World/MountainCliffRoad/GuardPost{idx:02d}_{post_idx:02d}",
                size=(0.030, 0.030, scene_cfg.rail_height),
                translation=(px, py, scene_cfg.road_z + 0.5 * scene_cfg.rail_height),
                color=tuple(max(0.02, c + 0.025 * ((post_idx % 3) - 1)) for c in post_color),
                collision=True,
                roughness=0.85,
                yaw=yaw,
            )
            if post_idx % 4 == 0:
                _cuboid(
                    f"/World/MountainCliffRoad/GuardReflector{idx:02d}_{post_idx:02d}",
                    size=(0.034, 0.006, 0.018),
                    translation=(px, py, scene_cfg.road_z + scene_cfg.rail_height + 0.020),
                    color=(0.80, 0.58, 0.12),
                    collision=False,
                    roughness=0.45,
                    yaw=yaw,
                )


def _spawn_right_side_barricades(scene_cfg: MountainCliffSceneCfg) -> None:
    """Guard rails are disabled so the figure-8 remains fully drivable."""
    if scene_cfg.map_name == "figure8":
        return

    paths = road_map(scene_cfg)
    post_color = (0.34, 0.29, 0.22)
    rail_color = (0.60, 0.56, 0.48)
    barrier_specs = (
        ("ApproachRight", paths.road_centerline[:6], -1.0, 0.5 * scene_cfg.road_width + scene_cfg.shoulder_width + 0.030),
        ("MainRight", paths.road_centerline[7:], -1.0, 0.5 * scene_cfg.road_width + scene_cfg.shoulder_width + 0.030),
        ("BranchRight", paths.right_branch_centerline[2:], 1.0, 0.5 * scene_cfg.road_width + scene_cfg.shoulder_width + 0.025),
    )
    for name, centerline, side_sign, offset in barrier_specs:
        for idx, (start, end) in enumerate(zip(centerline[:-1], centerline[1:], strict=False)):
            cx, cy, ux, uy, yaw = _segment_geometry(start, end)
            length = math.dist(start, end)
            ox = -uy * side_sign * offset
            oy = ux * side_sign * offset
            _cuboid(
                f"/World/MountainCliffRoad/{name}BarricadeRail{idx:02d}",
                size=(0.026, max(0.12, length - 0.18), 0.020),
                translation=(cx + ox, cy + oy, scene_cfg.road_z + 0.095),
                color=rail_color,
                collision=True,
                roughness=0.82,
                yaw=yaw,
            )
            post_count = max(2, int(length / 0.34))
            for post_idx in range(post_count):
                if post_idx % 3 == 2:
                    continue
                t = (post_idx + 0.5) / post_count
                px = start[0] + t * (end[0] - start[0]) + ox
                py = start[1] + t * (end[1] - start[1]) + oy
                _cuboid(
                    f"/World/MountainCliffRoad/{name}BarricadePost{idx:02d}_{post_idx:02d}",
                    size=(0.030, 0.030, 0.095),
                    translation=(px, py, scene_cfg.road_z + 0.045),
                    color=post_color,
                    collision=True,
                    roughness=0.90,
                    yaw=yaw,
                )


def _spawn_road_end_caps(scene_cfg: MountainCliffSceneCfg) -> None:
    """No end caps: both fork choices are closed loops back to the shared start."""
    if scene_cfg.map_name == "figure8":
        return

    paths = road_map(scene_cfg)
    cap_color = (0.55, 0.51, 0.43)
    post_color = (0.34, 0.29, 0.22)
    for name, centerline in (("LeftFork", paths.road_centerline), ("RightFork", paths.right_branch_centerline)):
        prev_point = centerline[-2]
        end_point = centerline[-1]
        _, _, _, _, yaw = _segment_geometry(prev_point, end_point)
        _cuboid(
            f"/World/MountainCliffRoad/{name}EndRail",
            size=(scene_cfg.road_width + 2.0 * scene_cfg.shoulder_width + 0.12, 0.040, 0.070),
            translation=(end_point[0], end_point[1], scene_cfg.road_z + 0.075),
            color=cap_color,
            collision=True,
            roughness=0.84,
            yaw=yaw,
        )
        for label, side_sign in (("Left", -1.0), ("Right", 1.0)):
            x, y = _offset_point(
                prev_point,
                end_point,
                t=1.0,
                side_sign=side_sign,
                offset=0.5 * scene_cfg.road_width + scene_cfg.shoulder_width,
            )
            _cuboid(
                f"/World/MountainCliffRoad/{name}EndPost{label}",
                size=(0.045, 0.045, 0.140),
                translation=(x, y, scene_cfg.road_z + 0.070),
                color=post_color,
                collision=True,
                roughness=0.90,
                yaw=yaw,
            )


def _spawn_terrain(scene_cfg: MountainCliffSceneCfg) -> None:
    paths = road_map(scene_cfg)

    def valley_height(x: float, y: float) -> float:
        ripple = 0.055 * math.sin(2.7 * x + 0.4) + 0.045 * math.cos(2.2 * y - 0.6)
        slope = -0.055 * x + 0.025 * y
        return scene_cfg.lower_terrain_z + slope + ripple

    def far_mountain_height(x: float, y: float) -> float:
        distance = max(0.0, y - 3.0)
        ridge = 0.18 + 0.10 * distance
        ridge += 0.28 * math.sin(0.75 * y + 0.55 * x)
        ridge += 0.16 * math.cos(1.35 * x - 0.25 * y)
        return scene_cfg.lower_terrain_z + ridge + 0.08 * abs(x)

    def high_mountain_height(x: float, y: float) -> float:
        distance = max(0.0, y - 8.0)
        ridge = 0.42 + 0.12 * distance + 0.09 * abs(x)
        ridge += 0.22 * math.sin(0.54 * y + 0.30 * x)
        ridge += 0.14 * math.cos(1.00 * x - 0.16 * y)
        return scene_cfg.lower_terrain_z + ridge

    def right_mountain_height(x: float, y: float) -> float:
        distance = max(0.0, x - 3.1)
        ridge = 0.22 + 0.18 * distance + 0.04 * max(0.0, y + 2.0)
        ridge += 0.18 * math.sin(0.85 * y + 0.55 * x)
        ridge += 0.12 * math.cos(1.20 * x - 0.30 * y)
        return scene_cfg.lower_terrain_z + ridge

    _mesh_grid(
        "/World/MountainCliffRoad/ValleyGround",
        x_range=(-5.0, 5.0),
        y_range=(-4.2, 9.4),
        nx=54,
        ny=78,
        height_fn=valley_height,
        color=(0.22, 0.20, 0.16),
        collision=True,
    )
    _mesh_grid(
        "/World/MountainCliffRoad/FarMountainRidges",
        x_range=(-6.8, 6.8),
        y_range=(3.2, 10.6),
        nx=56,
        ny=40,
        height_fn=far_mountain_height,
        color=(0.24, 0.22, 0.19),
        collision=True,
    )
    _mesh_grid(
        "/World/MountainCliffRoad/DistantHighMountains",
        x_range=(-8.6, 8.6),
        y_range=(8.4, 15.0),
        nx=60,
        ny=34,
        height_fn=high_mountain_height,
        color=(0.27, 0.26, 0.23),
        collision=True,
    )
    _mesh_grid(
        "/World/MountainCliffRoad/RightFarMountains",
        x_range=(3.1, 8.2),
        y_range=(-3.6, 9.6),
        nx=30,
        ny=64,
        height_fn=right_mountain_height,
        color=(0.25, 0.23, 0.20),
        collision=True,
    )
    _mesh_grid(
        "/World/MountainCliffRoad/LeftRockSlope",
        x_range=(-3.00, -1.58),
        y_range=(-3.4, 8.6),
        nx=12,
        ny=72,
        height_fn=lambda x, y: scene_cfg.lower_terrain_z + 0.30 + 0.78 * (-1.58 - x) + 0.06 * math.sin(4.0 * y),
        color=(0.27, 0.23, 0.19),
        collision=True,
    )
    _mesh_grid(
        "/World/MountainCliffRoad/RightDropSlope",
        x_range=(1.35, 3.55),
        y_range=(-3.4, 8.8),
        nx=12,
        ny=72,
        height_fn=lambda x, y: scene_cfg.lower_terrain_z + 0.18 + 0.22 * (x - 1.35) + 0.05 * math.cos(3.5 * y),
        color=(0.20, 0.19, 0.17),
        collision=True,
    )
    _mesh_grid(
        "/World/MountainCliffRoad/ForwardRollingHills",
        x_range=(-3.2, 4.2),
        y_range=(1.6, 9.2),
        nx=42,
        ny=52,
        height_fn=lambda x, y: scene_cfg.lower_terrain_z
        + 0.12
        + 0.095 * (y - 1.6)
        + 0.09 * math.sin(2.1 * x + 0.7 * y)
        + 0.05 * math.cos(3.0 * y),
        color=(0.23, 0.22, 0.18),
        collision=True,
    )
    _mesh_grid(
        "/World/MountainCliffRoad/ForegroundTalusField",
        x_range=(-2.80, 2.65),
        y_range=(-2.95, -1.55),
        nx=34,
        ny=12,
        height_fn=lambda x, y: scene_cfg.lower_terrain_z + 0.10 + 0.04 * math.sin(3.4 * x) + 0.05 * math.cos(2.6 * y),
        color=(0.19, 0.18, 0.15),
        collision=True,
    )
    _cuboid(
        "/World/MountainCliffRoad/LowerSolidGround",
        size=(11.0, 14.5, 0.060),
        translation=(0.0, 2.4, scene_cfg.lower_terrain_z - 0.10),
        color=(0.17, 0.16, 0.13),
        collision=True,
        roughness=1.0,
    )
    _cuboid(
        "/World/MountainCliffRoad/FallCatchFloor",
        size=(9.4, 12.8, 0.060),
        translation=(0.30, 1.85, scene_cfg.lower_terrain_z - 0.045),
        color=(0.20, 0.19, 0.16),
        collision=True,
        roughness=1.0,
    )

    _cuboid(
        "/World/MountainCliffRoad/River",
        size=(0.20, 12.4, 0.006),
        translation=(2.42, 2.10, scene_cfg.lower_terrain_z - 0.025),
        color=(0.03, 0.22, 0.34),
        collision=False,
        roughness=0.25,
        yaw=-0.28,
    )
    for idx, x_offset in enumerate((-0.05, 0.04, 0.095)):
        _cuboid(
            f"/World/MountainCliffRoad/RiverHighlight{idx:02d}",
            size=(0.024, 10.8 - 0.40 * idx, 0.004),
            translation=(2.42 + x_offset, 2.00 + 0.10 * idx, scene_cfg.lower_terrain_z - 0.019),
            color=(0.12, 0.42, 0.58),
            collision=False,
            roughness=0.22,
            yaw=-0.28,
        )
    for idx, (x, y, sx, sy) in enumerate(
        (
            (1.66, -1.20, 0.26, 0.70),
            (2.22, 0.42, 0.24, 0.86),
            (1.72, 1.20, 0.22, 0.52),
            (2.30, 2.00, 0.20, 0.66),
            (1.88, 3.20, 0.24, 0.82),
            (2.34, 4.40, 0.22, 0.70),
            (2.70, 5.70, 0.26, 0.76),
            (2.22, 7.10, 0.22, 0.62),
        )
    ):
        _cuboid(
            f"/World/MountainCliffRoad/GravelBar{idx:02d}",
            size=(sx, sy, 0.006),
            translation=(x, y, scene_cfg.lower_terrain_z - 0.014),
            color=(0.42, 0.36, 0.25),
            collision=False,
            roughness=0.95,
            yaw=-0.24,
        )
    _cuboid(
        "/World/MountainCliffRoad/StartShelf",
        size=(1.10, 0.82, 0.10),
        translation=(paths.road_centerline[0][0], paths.road_centerline[0][1], scene_cfg.road_z - 0.085),
        color=(0.29, 0.24, 0.19),
        collision=True,
        roughness=0.98,
        yaw=0.42,
    )


def _spawn_rocks_and_plants(scene_cfg: MountainCliffSceneCfg) -> None:
    rock_specs = (
        (-1.52, -1.36, 0.12, (0.30, 0.26, 0.22)),
        (-1.46, -0.58, 0.16, (0.25, 0.23, 0.21)),
        (-1.07, 0.07, 0.11, (0.36, 0.30, 0.24)),
        (-0.72, 0.62, 0.15, (0.29, 0.25, 0.22)),
        (1.70, 0.94, 0.14, (0.32, 0.28, 0.24)),
        (1.80, -0.35, 0.10, (0.27, 0.25, 0.22)),
        (2.10, -1.20, 0.18, (0.33, 0.28, 0.23)),
        (1.34, 1.78, 0.15, (0.31, 0.27, 0.23)),
        (0.88, 2.36, 0.18, (0.28, 0.25, 0.22)),
        (0.14, 3.04, 0.13, (0.35, 0.30, 0.24)),
        (-0.56, 3.72, 0.20, (0.30, 0.27, 0.23)),
        (2.42, 2.78, 0.22, (0.27, 0.25, 0.22)),
        (-1.32, 4.48, 0.16, (0.31, 0.27, 0.23)),
        (-1.64, 5.22, 0.20, (0.28, 0.25, 0.22)),
        (-0.42, 5.84, 0.14, (0.35, 0.30, 0.24)),
        (0.94, 5.92, 0.17, (0.30, 0.27, 0.23)),
        (3.22, -2.18, 0.18, (0.30, 0.26, 0.22)),
        (2.12, -2.78, 0.15, (0.34, 0.29, 0.24)),
    )
    for idx, (x, y, radius, color) in enumerate(rock_specs):
        _spawn_boulder_cluster(
            f"/World/MountainCliffRoad/Boulder{idx:02d}",
            x=x,
            y=y,
            z=scene_cfg.lower_terrain_z,
            scale=radius,
            color=color,
        )

    talus_specs = [
        (-1.34, -1.00, 0.052),
        (-1.22, -0.78, 0.044),
        (-1.02, -0.46, 0.060),
        (-0.78, -0.18, 0.040),
        (-0.44, 0.12, 0.050),
        (0.02, 0.42, 0.042),
        (0.50, 0.72, 0.056),
        (1.26, -1.12, 0.062),
        (1.44, -0.72, 0.046),
        (1.62, -0.18, 0.052),
        (1.78, 0.38, 0.042),
        (1.86, 0.94, 0.058),
        (-0.92, 3.82, 0.050),
        (-1.22, 4.58, 0.060),
        (-0.98, 5.12, 0.046),
        (-0.34, 5.76, 0.052),
        (0.52, 6.08, 0.056),
        (3.56, -1.80, 0.054),
        (3.12, -2.42, 0.060),
    ]
    for idx, (x, y, radius) in enumerate(talus_specs):
        _sphere(
            f"/World/MountainCliffRoad/TalusStone{idx:02d}",
            radius=radius,
            translation=(x, y, scene_cfg.lower_terrain_z + 0.06 + 0.01 * (idx % 3)),
            color=(0.24 + 0.02 * (idx % 2), 0.22, 0.19),
            collision=False,
            roughness=0.98,
        )

    left_slope_specs = (
        (-1.88, -1.06, 0.11, (0.31, 0.27, 0.23)),
        (-2.12, -0.62, 0.14, (0.27, 0.25, 0.22)),
        (-1.96, -0.18, 0.10, (0.34, 0.29, 0.24)),
        (-2.28, 0.36, 0.12, (0.29, 0.26, 0.23)),
        (-2.38, 1.08, 0.16, (0.30, 0.27, 0.23)),
        (-1.92, 1.32, 0.13, (0.35, 0.30, 0.25)),
    )
    for idx, (x, y, radius, color) in enumerate(left_slope_specs):
        z = scene_cfg.lower_terrain_z + 0.30 + 0.78 * (-1.58 - x) + 0.06 * math.sin(4.0 * y)
        _spawn_boulder_cluster(
            f"/World/MountainCliffRoad/LeftChaseBoulder{idx:02d}",
            x=x,
            y=y,
            z=z,
            scale=radius,
            color=color,
            collision=True,
        )

    strata_specs = (
        (-2.02, -0.88, 0.42, 0.12, 0.035, 0.35),
        (-2.22, -0.34, 0.36, 0.10, 0.030, -0.18),
        (-2.34, 0.62, 0.46, 0.13, 0.035, 0.22),
        (-1.84, 0.18, 0.30, 0.09, 0.028, 0.52),
        (-2.08, 1.02, 0.38, 0.11, 0.032, -0.35),
    )
    for idx, (x, y, sx, sy, sz, yaw) in enumerate(strata_specs):
        z = scene_cfg.lower_terrain_z + 0.30 + 0.78 * (-1.58 - x) + 0.06 * math.sin(4.0 * y)
        _cuboid(
            f"/World/MountainCliffRoad/LeftChaseStrataSlab{idx:02d}",
            size=(sx, sy, sz),
            translation=(x, y, z + 0.5 * sz),
            color=(0.26, 0.23, 0.19),
            collision=True,
            roughness=0.99,
            yaw=yaw,
        )


def _spawn_sky_and_lights(scene_cfg: MountainCliffSceneCfg) -> None:
    sky_kwargs: dict[str, object] = {}
    if SKY_TEXTURE_PATH.exists():
        sky_kwargs = {
            "texture_file": str(SKY_TEXTURE_PATH),
            "texture_format": "latlong",
            "visible_in_primary_ray": True,
        }
    sky_cfg = sim_utils.DomeLightCfg(intensity=900.0, color=(1.0, 1.0, 1.0), **sky_kwargs)
    sky_cfg.func("/World/MountainCliffRoad/SkyLight", sky_cfg)
    sun_cfg = sim_utils.DistantLightCfg(intensity=1700.0, color=(1.0, 0.86, 0.66), angle=0.24)
    sun_cfg.func("/World/MountainCliffRoad/SunLight", sun_cfg)

    # No sky wall here: chase-view teleop should see open background, not a flat panel.



def design_mountain_cliff_scene(scene_cfg: MountainCliffSceneCfg) -> None:
    """Spawn a realistic-looking mountain shelf road with a canyon drop."""
    _spawn_sky_and_lights(scene_cfg)
    _spawn_terrain(scene_cfg)
    _spawn_road(scene_cfg)
    _spawn_right_branch(scene_cfg)
    _spawn_guard_rails(scene_cfg)
    _spawn_right_side_barricades(scene_cfg)
    _spawn_road_end_caps(scene_cfg)
    _spawn_rocks_and_plants(scene_cfg)


def start_pose(scene_cfg: MountainCliffSceneCfg | None = None) -> tuple[tuple[float, float, float], float]:
    """Start at the lower end of the shared straight segment, facing the decision point."""
    cfg = scene_cfg or MountainCliffSceneCfg()
    paths = road_map(cfg)
    start = paths.road_centerline[0]
    next_point = paths.road_centerline[1]
    yaw = math.atan2(next_point[1] - start[1], next_point[0] - start[0])
    if cfg.map_name == "figure8":
        return (start[0], start[1], cfg.road_z + cfg.start_height), yaw

    dx = next_point[0] - start[0]
    dy = next_point[1] - start[1]
    segment_length = max(1e-6, math.hypot(dx, dy))
    t = min(0.55, cfg.start_offset / segment_length)
    x = start[0] + t * dx
    y = start[1] + t * dy
    return (x, y, cfg.road_z + cfg.start_height), yaw
