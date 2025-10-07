import collections
from collections.abc import Generator
from pathlib import Path
from typing import Self, cast

import attrs
import pygltflib
import pyvista as pv
import rich
import rich.tree
from loguru import logger

from liblaf import cherries, grapes
from liblaf.melon import io


class Config(cherries.BaseConfig):
    glb: Path = cherries.input("01-raw/complete_human_head_anatomy.glb")
    obj: Path = cherries.input("01-raw/Full human head anatomy.obj")

    output: Path = cherries.output("01-raw/groups.toml")


@attrs.define
class Tree:
    gltf: pygltflib.GLTF2 = attrs.field(repr=False)
    node_id: int = attrs.field()
    children: list[Self] = attrs.field(factory=list)
    parent: Self | None = attrs.field(default=None, repr=False)

    @classmethod
    def from_node(cls, gltf: pygltflib.GLTF2, node_id: int) -> Self:
        self: Self = cls(gltf=gltf, node_id=node_id)
        node: pygltflib.Node = gltf.nodes[node_id]
        for child_id in node.children or []:
            child: Self = cls.from_node(gltf, child_id)
            child.parent = self
            self.children.append(child)
        return self

    @property
    def name(self) -> str:
        assert self.node.name is not None
        return self.node.name

    @property
    def node(self) -> pygltflib.Node:
        return self.gltf.nodes[self.node_id]

    @property
    def rich_tree(self) -> rich.tree.Tree:
        tree: rich.tree.Tree = rich.tree.Tree(self.name)
        for child in self.children:
            tree.children.append(child.rich_tree)
        return tree

    @property
    def subtrees(self) -> Generator[Self]:
        yield self
        for child in self.children:
            yield from child.subtrees

    def find_subgroup(self) -> Self | None:
        if "subgroup" in self.name.lower():
            return self
        if self.parent is None:
            return None
        return self.parent.find_subgroup()


def main(cfg: Config) -> None:
    gltf: pygltflib.GLTF2 = cast("pygltflib.GLTF2", pygltflib.GLTF2.load(cfg.glb))
    scene: pygltflib.Scene = gltf.scenes[gltf.scene]
    assert scene.nodes is not None
    assert len(scene.nodes) == 1
    tree: Tree = Tree.from_node(gltf, scene.nodes[0])
    rich.print(tree.rich_tree)

    name_to_tree: dict[str, Tree] = {}
    for subtree in tree.subtrees:
        assert subtree.node.name is not None
        name: str = subtree.node.name.replace(" ", "_")
        name_to_tree[name] = subtree

    obj: pv.PolyData = io.load_polydata(cfg.obj)
    obj.clean(inplace=True)
    subgroups: dict[str, list[str]] = collections.defaultdict(list)
    for group_name_np in obj.field_data["GroupNames"]:
        group_name = str(group_name_np)
        if group_name not in name_to_tree:
            logger.warning("Node not found in glTF: {}", group_name)
            continue
        subtree: Tree = name_to_tree[group_name]
        subgroup: Tree | None = subtree.find_subgroup()
        if subgroup is None:
            logger.warning("Subgroup not found for: {}", group_name)
            continue
        subgroup_name: str = subgroup.name.split(maxsplit=1)[0]
        subgroups[subgroup_name].append(group_name)
        logger.success("{} > {}", subgroup_name, group_name)
    grapes.save(cfg.output, subgroups)


if __name__ == "__main__":
    cherries.run(main)
