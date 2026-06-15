from pathlib import Path

import pyvista as pv

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    folder: Path = cherries.input("~/.local/opt/Wrap/Gallery/Blendshapes/")
    output: Path = cherries.output("60-blendshapes.vtp")


def main(cfg: Config) -> None:
    mesh: pv.PolyData = melon.io.load_polydata(cfg.folder / "Basemesh.obj")
    expression_name: list[str] = []
    for file in cfg.folder.glob("*.obj"):
        if file.name == "Basemesh.obj":
            continue
        blendshape: pv.PolyData = melon.io.load_polydata(file)
        expression_name.append(file.stem)
        mesh.point_data[file.stem] = blendshape.points - mesh.points
    mesh.clean(inplace=True)
    mesh.flip_faces(inplace=True)
    mesh.field_data["ExpressionName"] = expression_name
    melon.save(mesh, cfg.output)


if __name__ == "__main__":
    cherries.main(main)
