from pathlib import Path

import numpy as np
import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    full: Path = cherries.input("00-Full human head anatomy.obj")
    groups: Path = cherries.input("00-groups.toml")

    output: Path = cherries.output("10-muscles.vtp")


def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_polydata(cfg.full)
    full.clean(inplace=True)
    groups: dict[str, list[str]] = grapes.load(cfg.groups)

    muscles: list[pv.PolyData] = []
    edges_list: list[pv.PolyData] = []
    n_pipe_like: int = 0
    for group_name in groups["Muscles"]:
        muscle: pv.PolyData = melon.tri.extract_groups(full, group_name)
        blocks: pv.MultiBlock = muscle.split_bodies().as_polydata_blocks()
        for i, block in enumerate(blocks):
            block: pv.PolyData
            fixed: pv.PolyData = melon.mesh_fix(block)
            muscle_name: str = f"{group_name}.{i}"
            ic(muscle_name)
            fixed.user_dict["name"] = muscle_name
            fixed.cell_data["muscle-id"] = np.full(
                (fixed.n_cells,), len(muscles), dtype=np.int32
            )
            fixed.cell_data["is-volume"] = np.full(
                (fixed.n_cells,), melon.tri.is_volume(fixed), dtype=np.bool
            )
            edges: pv.PolyData = block.extract_feature_edges(
                boundary_edges=True,
                non_manifold_edges=False,
                feature_edges=False,
                manifold_edges=False,
            )  # pyright: ignore[reportAssignmentType]
            edges_list.append(edges)
            if edges.is_empty:
                fixed.cell_data["n-holes"] = np.zeros((fixed.n_cells,), dtype=np.int32)
            else:
                edges = edges.connectivity()  # pyright: ignore[reportAssignmentType]
                n_holes: int = np.max(edges.point_data["RegionId"]) + 1
                if n_holes == 2:
                    n_pipe_like += 1
                    ic(f"{muscle_name} is pipe-like")
                fixed.cell_data["n-holes"] = np.full(
                    (fixed.n_cells,), n_holes, dtype=np.int32
                )
            muscles.append(fixed)
    ic(len(muscles))
    ic(n_pipe_like)
    melon.save(cfg.output, pv.merge(muscles))
    melon.save("tmp/edges.vtp", pv.merge(edges_list))


if __name__ == "__main__":
    cherries.main(main)
