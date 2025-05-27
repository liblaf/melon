from pathlib import Path

import pyvista as pv
import pyvista.examples
import rich

from liblaf import melon


def main() -> None:
    path = Path(pyvista.examples.download_wavy(load=False))
    reader = pv.PVDReader(path)
    rich.inspect(reader)
    writer = melon.SeriesWriter("data/animation.vtu.series", clear=True)
    for time in reader.time_values:
        reader.set_active_time_value(time)
        block: pv.MultiBlock = reader.read()
        structured_grid: pv.StructuredGrid = block[0]
        writer.append(structured_grid, time=time)
    writer.end()


if __name__ == "__main__":
    main()
