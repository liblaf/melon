from pathlib import Path

import attrs
import rich
import trimesh as tm
from rich.text import Text
from rich.tree import Tree

from liblaf import cherries


class Config(cherries.BaseConfig):
    input: Path = cherries.input("00-complete_human_head_anatomy.glb")


@attrs.define
class SceneTree:
    data: dict[str, Tree] = attrs.field(factory=dict)

    def get(self, node: str, geometry: str | None = None) -> Tree:
        if node not in self.data:
            if geometry is None:
                self.data[node] = Tree(node)
            elif node == geometry:
                self.data[node] = Tree(Text.assemble((f"[{geometry}]", "green")))
            else:
                self.data[node] = Tree(
                    Text.assemble(node, " ", (f"[{geometry}]", "green"))
                )
        return self.data[node]


def main(cfg: Config) -> None:
    scene: tm.Scene = tm.load_scene(cfg.input)
    tree: SceneTree = SceneTree()
    for parent, child, metadata in scene.graph.to_edgelist():
        parent_node: Tree = tree.get(parent)
        child_node: Tree = tree.get(child, metadata.get("geometry"))
        parent_node.add(child_node)
    rich.print(tree.get(scene.graph.base_frame))


if __name__ == "__main__":
    cherries.main(main, profile="debug")
