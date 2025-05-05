from pathlib import Path

import networkx as nx
import numpy as np
import pyvista as pv
from jaxtyping import Integer

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    tetmesh: Path = (
        grapes.find_project_dir() / "data/03-primary/skin-with-mouth-socket.ply"
    )
    output: Path = grapes.find_project_dir() / "data/03-primary/skin-distance.vtp"


def main(cfg: Config) -> None:
    surface: pv.PolyData = melon.load(cfg.tetmesh)
    surface.point_data["point-id"] = np.arange(surface.n_points)
    edges: pv.PolyData = surface.extract_all_edges()
    edges = edges.compute_cell_sizes()
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        np.concatenate(
            (
                edges.lines.reshape(-1, 3)[:, 1:],
                edges.cell_data["Length"].reshape(-1, 1),
            ),
            axis=1,
        )
    )
    ic(graph.number_of_nodes(), graph.number_of_edges())

    nearest_result: melon.NearestPointResult = melon.nearest(
        edges,
        pv.wrap(
            np.asarray(
                [[0.724907, 25.935352, 9.666245], [0.689024, 26.452383, 9.960980]]
            )
        ),
        algo=melon.NearestPoint(normal_threshold=-np.inf),
    )  # pyright: ignore[reportAssignmentType]
    source: int
    target: int
    source, target = nearest_result.vertex_id
    ic(source, target)
    path: list[int] = nx.shortest_path(graph, source, target, weight="weight")  # pyright: ignore[reportAssignmentType]
    ic(nx.path_weight(graph, path, weight="weight"))
    path: Integer[np.ndarray, " P"] = np.asarray(path, dtype=int)
    surface.point_data["in-path"] = np.zeros((surface.n_points,), dtype=bool)
    surface.point_data["in-path"][edges.point_data["point-id"][path]] = True
    melon.save(cfg.output, surface)


if __name__ == "__main__":
    cherries.run(main)
