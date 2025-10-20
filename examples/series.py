import sys
from pathlib import Path

import pyvista as pv
import pyvista.examples
import rich
from icecream import ic

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import grapes


def main() -> None:
    grapes.logging.init()
    file: Path = Path(sys.argv[0]).parent / "data" / "animation.vts.series"
    pvd = pv.PVDReader(pyvista.examples.download_wavy(load=False))
    rich.inspect(pvd)
    with melon.SeriesWriter(file, clear=True) as writer:
        for time in pvd.time_values:
            pvd.set_active_time_value(time)
            block: pv.MultiBlock = pvd.read()
            obj: pv.StructuredGrid = block[0]  # pyright: ignore[reportAssignmentType]
            writer.append(obj, time=time)
    reader: melon.SeriesReader[pv.StructuredGrid] = melon.SeriesReader(
        file, melon.load_structured_grid
    )
    for obj in reader:
        ic(obj)


if __name__ == "__main__":
    main()
