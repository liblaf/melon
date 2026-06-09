"""Triangular surface geometry helpers."""

from ._contains import contains
from ._edge import edge_length
from ._fill_point import fill_point
from ._geodestic import geodesic_path
from ._group import extract_cells, extract_groups, select_groups
from ._implicit_distance import implicit_distance
from ._query_ray import query_ray
from ._repair import fix_normals

__all__ = [
    "contains",
    "edge_length",
    "extract_cells",
    "extract_groups",
    "fill_point",
    "fix_normals",
    "geodesic_path",
    "implicit_distance",
    "query_ray",
    "select_groups",
]
