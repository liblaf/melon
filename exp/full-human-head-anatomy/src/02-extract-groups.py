from pathlib import Path

import numpy as np
import polars as pl
import pyvista as pv
from jaxtyping import Integer

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    groups: Path = Path("data/01-raw/groups.toml")
    input: Path = Path("data/01-raw/complete-human-head-anatomy.obj")
    output: Path = Path("data/01-raw/")


@cherries.main()
def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_poly_data(cfg.input)
    ic(full.field_data["GroupNames"])
    df: pl.DataFrame = pl.from_dict(
        {
            "Group ID": range(len(full.field_data["GroupNames"])),
            "Group Name": full.field_data["GroupNames"],  # pyright: ignore[reportArgumentType]
        }
    )
    df.write_csv(cfg.output / "groups.en.csv")

    for group_name in full.field_data["GroupNames"]:
        selection: Integer[np.ndarray, " S"] = melon.triangle.select_groups(
            full, group_name
        )
        mesh: pv.PolyData = melon.triangle.extract_cells(full, selection)
        melon.save(cfg.output / "objects" / f"{group_name}.ply", mesh)

    groups: dict[str, list[str]] = grapes.load(cfg.groups)
    for key, value in groups.items():
        selection: Integer[np.ndarray, " S"] = melon.triangle.select_groups(full, value)
        subgroup: pv.PolyData = melon.triangle.extract_cells(full, selection)
        melon.save(cfg.output / f"{key}.ply", subgroup)
        for group_name in value:
            selection: Integer[np.ndarray, " S"] = melon.triangle.select_groups(
                full, group_name
            )
            mesh: pv.PolyData = melon.triangle.extract_cells(full, selection)
            melon.save(cfg.output / key / f"{group_name}.ply", mesh)


if __name__ == "__main__":
    main(Config())
