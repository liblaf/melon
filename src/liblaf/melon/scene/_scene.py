from collections.abc import Iterable

import trimesh as tm


def dump(
    scene: tm.Scene,
    geometry_names: Iterable[str] | None = None,
    *,
    include_visual: bool = True,
) -> list[tm.Trimesh]:
    geometries: list[tm.Trimesh] = []
    if geometry_names is None:
        geometry_names: list[str] = [
            name
            for name, geometry in scene.geometry.items()
            if isinstance(geometry, tm.Trimesh)
        ]
    for geometry_name in geometry_names:
        for node_name in scene.graph.geometry_nodes[geometry_name]:
            transform, geometry_name_ = scene.graph[node_name]
            assert geometry_name == geometry_name_
            geometry: tm.Trimesh = scene.geometry[geometry_name]
            geometry: tm.Trimesh = geometry.copy(include_visual=include_visual)
            geometry: tm.Trimesh = geometry.apply_transform(transform)
            geometry.metadata["name"] = geometry_name
            geometry.metadata["node"] = node_name
            geometries.append(geometry)
    return geometries


def subscene(scene: tm.Scene, node_name: str) -> tm.Scene:
    subscene: tm.Scene = scene.subscene(node_name)
    transform, _geometry_name = scene.graph[node_name]
    subscene.apply_transform(transform)
    return subscene
