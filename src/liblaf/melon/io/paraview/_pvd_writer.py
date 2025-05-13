import shutil
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import attrs

from liblaf.melon.io._save import save


@attrs.frozen(kw_only=True)
class PVDDataSet:
    timestep: float
    part: int
    file: Path


@attrs.frozen
class PVDWriter:
    """.

    References:
        [1]: [ParaView/Data formats - KitwarePublic](https://www.paraview.org/Wiki/ParaView/Data_formats#PVD_File_Format)
    """

    @property
    def folder(self) -> Path:
        return self.file.with_suffix("")

    @property
    def name(self) -> str:
        return self.file.stem

    clear: bool = attrs.field(default=False, kw_only=True)
    datasets: list[PVDDataSet] = attrs.field(init=False, factory=list)
    file: Path = attrs.field(default=Path("animation.pvd"), converter=Path)
    fps: float = attrs.field(default=30.0, kw_only=True)

    def __attrs_post_init__(self) -> None:
        if self.clear:
            shutil.rmtree(self.folder, ignore_errors=True)

    def end(self) -> None:
        root = ElementTree.Element(
            "VTKFile", type="Collection", version="0.1", byte_order="LittleEndian"
        )
        collection: ElementTree.Element = ElementTree.SubElement(root, "Collection")
        root_dir: Path = self.file.absolute().parent
        for dataset in self.datasets:
            elem: ElementTree.Element = ElementTree.SubElement(collection, "DataSet")
            elem.set("timestep", str(dataset.timestep))
            elem.set("part", str(dataset.part))
            elem.set("file", dataset.file.absolute().relative_to(root_dir).as_posix())
        tree = ElementTree.ElementTree(root)
        ElementTree.indent(tree, space="  ")
        self.file.parent.mkdir(parents=True, exist_ok=True)
        tree.write(self.file, xml_declaration=True)

    def append(
        self, dataset: Any, timestep: float | None = None, *, ext: str, part: int = 0
    ) -> None:
        if timestep is None:
            timestep = (
                self.datasets[-1].timestep + (1 / self.fps) if self.datasets else 0
            )
        frame_id: int = len(self.datasets)
        filename: str = f"{self.name}_{frame_id:06d}"
        filename += ext
        filepath: Path = self.folder / filename
        save(filepath, dataset)
        self.datasets.append(PVDDataSet(timestep=timestep, part=part, file=filepath))
        self.end()
