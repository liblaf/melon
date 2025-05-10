from typing import Any


def is_instance(obj: Any, prefix: str, name: str) -> bool:
    return is_subclass(type(obj), prefix, name)


def is_subclass(cls: type, prefix: str, name: str) -> bool:
    return cls.__module__.startswith(prefix) and cls.__name__ == name


# ----------------------------------- numpy ---------------------------------- #
def is_numpy(obj: Any) -> bool:
    return is_subclass(obj, "numpy", "ndarray")


# ---------------------------------- pyvista --------------------------------- #
def is_image_data(obj: Any) -> bool:
    return is_instance(obj, "pyvista", "ImageData")


def is_point_set(obj: Any) -> bool:
    return is_instance(obj, "pyvista", "PointSet")


def is_poly_data(obj: Any) -> bool:
    return is_instance(obj, "pyvista", "PolyData")


def is_unstructured_grid(obj: Any) -> bool:
    return is_instance(obj, "pyvista", "UnstructuredGrid")


# ---------------------------------- trimesh --------------------------------- #
def is_trimesh(obj: Any) -> bool:
    return is_instance(obj, "trimesh", "Trimesh")
