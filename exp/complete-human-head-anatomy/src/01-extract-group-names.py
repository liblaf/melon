import collections
from pathlib import Path

import pygltflib
import pyvista as pv
import rich
from loguru import logger
from rich.tree import Tree

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    glb: Path = (
        grapes.find_project_dir() / "data/01-raw/complete_human_head_anatomy.glb"
    )
    obj: Path = grapes.find_project_dir() / "data/01-raw/Full human head anatomy.obj"
    output: Path = grapes.find_project_dir() / "data/02-intermediate/groups.toml"


def traverse_names(gltf: pygltflib.GLTF2, node_id: int) -> set[str]:
    node: pygltflib.Node = gltf.nodes[node_id]
    names: set[str] = {node.name}  # pyright: ignore[reportAssignmentType]
    if node.mesh is not None:
        mesh: pygltflib.Mesh = gltf.meshes[node.mesh]
        names.add(mesh.name)  # pyright: ignore[reportArgumentType]
    if node.children is not None:
        for child in node.children:
            names |= traverse_names(gltf, child)
    return names


def traverse_gltf(gltf: pygltflib.GLTF2, node_id: int, tree: Tree) -> None:
    node: pygltflib.Node = gltf.nodes[node_id]
    name: str = node.name  # pyright: ignore[reportAssignmentType]
    if node.mesh is not None:
        name += " (Mesh)"
    subtree: Tree = tree.add(name)
    if node.children is not None:
        for child_id in node.children:
            traverse_gltf(gltf, child_id, subtree)


def pretty_tree(gltf: pygltflib.GLTF2) -> Tree:
    scene: pygltflib.Scene = gltf.scenes[gltf.scene]
    assert len(scene.nodes) == 1  # pyright: ignore[reportArgumentType]
    root_id: int = scene.nodes[0]  # pyright: ignore[reportOptionalSubscript]
    root: pygltflib.Node = gltf.nodes[root_id]
    tree = Tree(root.name)  # pyright: ignore[reportArgumentType]
    traverse_gltf(gltf, root_id, tree)
    return tree


def main(cfg: Config) -> None:
    glb: pygltflib.GLTF2 = pygltflib.GLTF2.load(cfg.glb)  # pyright: ignore[reportAssignmentType]
    obj: pv.PolyData = melon.load_poly_data(cfg.obj)
    rich.print(pretty_tree(glb))
    group_names: set[str] = set(obj.field_data["GroupNames"])
    scene: pygltflib.Scene = glb.scenes[glb.scene]
    assert len(scene.nodes) == 1  # pyright: ignore[reportArgumentType]
    root: int = scene.nodes[0]  # pyright: ignore[reportOptionalSubscript]
    while len(glb.nodes[root].children) == 1:  # pyright: ignore[reportArgumentType]
        root = glb.nodes[root].children[0]  # pyright: ignore[reportOptionalSubscript]
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for subgroup_id in glb.nodes[root].children:  # pyright: ignore[reportOptionalIterable]
        subgroup: pygltflib.Node = glb.nodes[subgroup_id]
        subgroup_name: str = subgroup.name.split()[0]  # pyright: ignore[reportOptionalMemberAccess]
        names: set[str] = traverse_names(glb, subgroup_id)
        for name_glb in names:
            name: str = name_glb.replace(" ", "_")
            if name in group_names:
                groups[subgroup_name].append(name)  # pyright: ignore[reportArgumentType]
    components: set[str] = {name for values in groups.values() for name in values}
    for name in components - group_names:
        logger.warning("Not found in Group Names: `{}`.", name)
    for name in group_names - components:
        logger.warning("Not found in GLTF2: `{}`.", name)
    groups = {key: value for key, value in groups.items() if len(value) > 1}
    grapes.save(cfg.output, groups)


if __name__ == "__main__":
    cherries.run(main)
