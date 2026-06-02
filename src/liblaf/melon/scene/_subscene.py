import trimesh as tm


def subscene(scene: tm.Scene, node: str) -> tm.Scene:
    subscene: tm.Scene = scene.subscene(node)
    transform, _ = scene.graph[node]
    subscene.apply_transform(transform)
    return subscene
