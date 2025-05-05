from pathlib import Path

import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    full: Path = Path("data/01-raw/Full human head anatomy.obj")
    groups: Path = Path("data/02-intermediate/groups.toml")
    output_dir: Path = Path("data/04-registration/")


def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_poly_data(cfg.full)
    groups: dict[str, list[str]] = grapes.load(cfg.groups)
    groups: dict[str, set[str]] = {k: set(v) for k, v in groups.items()}

    cranium: pv.PolyData = melon.triangle.extract_groups(
        full, groups["cranium"] | groups["upper-teeth"]
    )
    melon.save(cfg.output_dir / "cranium.ply", cranium)

    mandible: pv.PolyData = melon.triangle.extract_groups(
        full, groups["mandible"] | groups["lower-teeth"]
    )
    melon.save(cfg.output_dir / "mandible.ply", mandible)


if __name__ == "__main__":
    cherries.run(main)
