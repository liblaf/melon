from pathlib import Path

import numpy as np
import polars as pl
import pyvista as pv
from jaxtyping import Bool

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    groups: Path = Path("data/01_raw/groups.toml")
    input: Path = Path("data/01_raw/human-head-anatomy.obj")
    output: Path = Path("data/01_raw/")


@cherries.main()
def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_poly_data(cfg.input)

    df: pl.DataFrame = pl.from_dict(
        {
            "GroupIds": range(len(full.field_data["GroupNames"])),
            "GroupNames": full.field_data["GroupNames"],  # pyright: ignore[reportArgumentType]
        }
    )
    df.write_csv(cfg.output / "groups.en.csv")

    for group_name in full.field_data["GroupNames"]:
        selection: Bool[np.ndarray, " C"] = melon.select_cells_by_group(
            full, group_name
        )
        mesh: pv.PolyData = melon.extract_cells(full, selection)
        melon.save(cfg.output / "objects" / f"{group_name}.ply", mesh)

    groups: dict[str, list[str]] = grapes.load(cfg.groups)
    for key, value in groups.items():
        selection: Bool[np.ndarray, " C"] = melon.select_cells_by_group(full, value)
        subgroup: pv.PolyData = melon.extract_cells(full, selection)
        melon.save(cfg.output / f"{key}.ply", subgroup)
        for group_name in value:
            selection: Bool[np.ndarray, " C"] = melon.select_cells_by_group(
                full, group_name
            )
            mesh: pv.PolyData = melon.extract_cells(full, selection)
            melon.save(cfg.output / key / f"{group_name}.ply", mesh)


if __name__ == "__main__":
    main(Config())
