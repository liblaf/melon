from pathlib import Path

import networkx as nx
import numpy as np
import pyvista as pv
from jaxtyping import Integer

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    tetgen: Path = cherries.input("02-intermediate/20-tetgen.vtu")
    output: Path = cherries.output("02-intermediate/22-tetgen-distance.vtp")


def main(cfg: Config) -> None:
    cherries.log_input(cfg.tetgen)
    tetgen: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetgen)
    surface: pv.PolyData = tetgen.extract_surface()
    surface = surface.compute_normals()
    graph: nx.Graph = melon.mesh.graph(surface)
    nearest_result: melon.NearestResult = melon.nearest(
        surface,
        np.asarray(
            [
                [0.724907, 25.935352, 9.666245],  # midpoint below lips
                [0.689024, 26.452383, 9.960980],  # midpoint above lips
            ]
        ),
        algo=melon.NearestPoint(normal_threshold=-np.inf),
    )
    source: int
    target: int
    source, target = nearest_result["vertex_id"]
    path: list[int] = nx.shortest_path(graph, source, target, weight="length")  # pyright: ignore[reportAssignmentType]
    cherries.log_metric(
        "shortest_path_length", nx.path_weight(graph, path, weight="length")
    )
    path: Integer[np.ndarray, " P"] = np.asarray(path, dtype=int)
    surface.point_data["in-path"] = np.zeros((surface.n_points,), dtype=bool)
    surface.point_data["in-path"][path] = True

    distance: dict[str, float] = nx.shortest_path_length(graph, source, weight="length")
    surface.point_data["distance"] = np.zeros((surface.n_points,), dtype=float)
    surface.point_data["distance"][list(distance.keys())] = list(distance.values())

    melon.save(cfg.output, surface)
    cherries.log_output(cfg.output)


if __name__ == "__main__":
    cherries.run(main, profile="playground")
