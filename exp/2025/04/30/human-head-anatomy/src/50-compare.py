import pyvista as pv

from liblaf import melon


def extract(mesh: pv.UnstructuredGrid) -> pv.PolyData:
    surface: pv.PolyData = mesh.extract_surface()
    surface = surface.extract_points(surface.point_data["is-face"]).extract_surface()
    return surface


def main() -> None:
    old: pv.UnstructuredGrid = melon.io.load_unstructured_grid(
        "/home/liblaf/github/liblaf/apple/exp/2025/10/22/inverse-flame/data/10-target.vtu"
    )
    new: pv.UnstructuredGrid = melon.io.load_unstructured_grid(
        "/home/liblaf/github/liblaf/melon/exp/2025/04/30/human-head-anatomy/data/40-expression.vtu"
    )
    old_surface: pv.PolyData = extract(old)
    new_surface: pv.PolyData = extract(new)
    melon.io.save("old-surface.vtp", old_surface)
    melon.io.save("new-surface.vtp", new_surface)


if __name__ == "__main__":
    main()
