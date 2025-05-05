import pyvista as pv

from liblaf import melon


def main() -> None:
    mesh: pv.UnstructuredGrid = pv.read("./tetgen.msh")
    ic(mesh)
    ic(mesh.cell_data)
    melon.save("tetgen.vtu", mesh)


if __name__ == "__main__":
    main()
