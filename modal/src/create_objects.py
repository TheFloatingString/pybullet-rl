from math import pi
from typing import Any, List


def _create_visual_cylinder(
    p: Any, radius: float, length: float, rgba: List[float]
) -> int:
    """Create a cylinder visual shape."""
    return p.createVisualShape(
        p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=rgba
    )


def _create_arrow_body(
    p: Any, shape_index: int, position: List[float], euler: List[float]
) -> None:
    """Create a multi-body with the given visual shape."""
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=shape_index,
        basePosition=position,
        baseOrientation=p.getQuaternionFromEuler(euler),
    )


def _create_axis_arrow(
    p: Any,
    origin: List[float],
    axis_idx: int,
    color: List[float],
    axis_len: float,
    shaft_r: float,
    head_r: float,
    head_len: float,
    euler: List[float],
) -> None:
    """Create arrow for single axis with shaft and head."""
    shaft = _create_visual_cylinder(p, shaft_r, axis_len - head_len, color)
    head = _create_visual_cylinder(p, head_r, head_len, color)

    shaft_pos = origin.copy()
    shaft_pos[axis_idx] += (axis_len - head_len) / 2
    _create_arrow_body(p, shaft, shaft_pos, euler)

    head_pos = origin.copy()
    head_pos[axis_idx] += axis_len - head_len / 2
    _create_arrow_body(p, head, head_pos, euler)


def create_rgb_axes(p: Any) -> None:
    """Create XYZ (RGB) axis arrows at origin."""
    origin = [0.0, 0.0, 0.5]
    axis_length = 1.0
    arrow_radius = 0.02
    arrow_head_radius = 0.04
    arrow_head_length = 0.15

    _create_axis_arrow(
        p,
        origin,
        0,
        [1, 0, 0, 1],
        axis_length,
        arrow_radius,
        arrow_head_radius,
        arrow_head_length,
        [0, pi / 2, 0],
    )
    _create_axis_arrow(
        p,
        origin,
        1,
        [0, 1, 0, 1],
        axis_length,
        arrow_radius,
        arrow_head_radius,
        arrow_head_length,
        [pi / 2, 0, 0],
    )
    _create_axis_arrow(
        p,
        origin,
        2,
        [0, 0, 1, 1],
        axis_length,
        arrow_radius,
        arrow_head_radius,
        arrow_head_length,
        [0, 0, 0],
    )
