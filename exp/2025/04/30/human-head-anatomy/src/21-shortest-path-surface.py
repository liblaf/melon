from pathlib import Path

import networkx as nx
import numpy as np
import pyvista as pv
from jaxtyping import Integer

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    surface: Path = cherries.data("02-intermediate/skin-with-mouth-socket.ply")
    output: Path = cherries.data("02-intermediate/skin-distance.vtp")


def main(cfg: Config) -> None:
    cherries.log_input(cfg.surface)
    surface: pv.PolyData = melon.load_poly_data(cfg.surface)

    graph: nx.Graph = melon.mesh.graph(surface)
    nearest_result: melon.NearestResult = melon.nearest(
        surface,
        np.asarray(
            [
                [0.724907, 25.935352, 9.666245],
                [0.689024, 26.452383, 9.960980],
            ]
        ),
    )
    source: int
    target: int
    source, target = nearest_result["vertex_id"]
    path: list[int] = nx.shortest_path(graph, source, target, weight="length")  # pyright: ignore[reportAssignmentType]
    ic(nx.path_weight(graph, path, weight="weight"))
    path: Integer[np.ndarray, " P"] = np.asarray(path, dtype=int)
    surface.point_data["in-path"] = np.zeros((surface.n_points,), dtype=bool)
    surface.point_data["in-path"][path] = True

    melon.save(cfg.output, surface)
    cherries.log_output(cfg.output)


if __name__ == "__main__":
    cherries.run(main)
