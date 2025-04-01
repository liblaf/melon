from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import attrs

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import grapes


@attrs.frozen(kw_only=True)
class PVDDataSet:
    timestep: float
    group: str
    part: int
    file: Path


@attrs.frozen
class PVDWriter:
    """...

    References:
        - [ParaView/Data formats - KitwarePublic](https://www.paraview.org/Wiki/ParaView/Data_formats#PVD_File_Format)
    """

    def _default_frame_dir(self) -> Path:
        return self.fpath.parent / "frames"

    datasets: list[PVDDataSet] = attrs.field(init=False, factory=list)
    fpath: Path = attrs.field(default=Path("animation.pvd"), converter=grapes.as_path)
    frame_dir: Path = attrs.field(
        default=attrs.Factory(_default_frame_dir, takes_self=True)
    )
    fps: float = attrs.field(default=30.0, kw_only=True)

    def end(self) -> None:
        root = ElementTree.Element(
            "VTKFile", type="Collection", version="0.1", byte_order="LittleEndian"
        )
        collection: ElementTree.Element = ElementTree.SubElement(root, "Collection")
        root_dir: Path = self.fpath.absolute().parent
        for dataset in self.datasets:
            elem: ElementTree.Element = ElementTree.SubElement(collection, "DataSet")
            elem.set("timestep", str(dataset.timestep))
            elem.set("group", dataset.group)
            elem.set("part", str(dataset.part))
            elem.set("file", dataset.file.absolute().relative_to(root_dir).as_posix())
        tree = ElementTree.ElementTree(root)
        ElementTree.indent(tree, space="  ")
        self.fpath.parent.mkdir(parents=True, exist_ok=True)
        tree.write(self.fpath, xml_declaration=True)

    def append(
        self,
        dataset: Any,
        timestep: float | None = None,
        *,
        group: str = "",
        part: int = 0,
        ext: str | None = None,
    ) -> None:
        if timestep is None:
            timestep = (
                self.datasets[-1].timestep + (1 / self.fps) if self.datasets else 0
            )
        frame_id: int = len(self.datasets)
        filename: str = f"F{frame_id:06d}"
        if ext is None:
            ext = melon.io.identify_data_format(dataset)
        filename += ext
        filepath: Path = self.frame_dir / filename
        melon.save(filepath, dataset)
        self.datasets.append(
            PVDDataSet(timestep=timestep, group=group, part=part, file=filepath)
        )
