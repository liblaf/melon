from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Bool

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    suffix: str = "3191k"
    source: Path = cherries.input("50-expression.vtp")


def main(config: Config) -> None:
    source: pv.PolyData = melon.io.load_polydata(config.source)
    source.extract_largest(inplace=True)
    target: pv.UnstructuredGrid = melon.io.load_unstructured_grid(
        cherries.input(f"42-tetmesh-{config.suffix}.vtu")
    )

    target_surface: pv.PolyData = target.extract_surface(algorithm=None)
    source: pv.PolyData = melon.ext.fast_wrapping(source, target_surface)
    melon.save(source, cherries.temp(f"50-{config.suffix}-source-wrapped.vtp"))

    names: list[str] = [
        name for name in source.point_data if name.startswith("Expression")
    ]
    target_surface: pv.PolyData = melon.xfer.tri_point_to_tri_point(
        source=source, target=target_surface, names=names
    )
    melon.save(target_surface, cherries.temp(f"50-{config.suffix}-target-surface.vtp"))
    mask: Bool[np.ndarray, " T"] = ~np.all(
        np.isfinite(target_surface.point_data[names[0]]), axis=-1
    )
    target_surface: pv.PolyData = melon.tri.fill_point(
        target_surface, mask, names=names
    )
    for name in names:
        target_surface.point_data[name] = np.nan_to_num(
            target_surface.point_data[name], nan=0.0
        )
    melon.save(
        target_surface, cherries.temp(f"50-{config.suffix}-target-surface-fill.vtp")
    )
    target: pv.UnstructuredGrid = melon.xfer.tri_point_to_tet_point(
        target_surface, target, fill_values=dict.fromkeys(names, 0.0)
    )
    melon.save(target, cherries.output(f"50-tetmesh-{config.suffix}.vtu"))


if __name__ == "__main__":
    cherries.main(main)
