"""Triangular surface geometry helpers."""

from ._contains import contains
from ._edge import edge_length
from ._fill_point import fill_point
from ._geodestic import geodesic_path
from ._group import extract_cells, extract_groups, select_groups
from ._implicit_distance import implicit_distance
from ._mesh_query_point import MeshQueryPointResult, mesh_query_point
from ._query_ray import query_ray
from ._repair import fix_normals

__all__ = [
    "MeshQueryPointResult",
    "contains",
    "edge_length",
    "extract_cells",
    "extract_groups",
    "fill_point",
    "fix_normals",
    "geodesic_path",
    "implicit_distance",
    "mesh_query_point",
    "query_ray",
    "select_groups",
]
