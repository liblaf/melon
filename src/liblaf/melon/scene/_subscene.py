from collections.abc import Iterable

import trimesh as tm


def extract_geometries(
    scene: tm.Scene, geometry_names: Iterable[str]
) -> list[tm.Geometry]:
    geometries: list[tm.Geometry] = []
    for geometry_name in geometry_names:
        for node_name in scene.graph.geometry_nodes[geometry_name]:
            transform, geometry_name_ = scene.graph[node_name]
            assert geometry_name == geometry_name_
            geometry: tm.Geometry = scene.geometry[geometry_name]
            geometry: tm.Geometry = geometry.copy()
            geometry: tm.Geometry = geometry.apply_transform(transform)
            geometries.append(geometry)
    return geometries
