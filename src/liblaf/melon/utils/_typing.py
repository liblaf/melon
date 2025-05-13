from collections.abc import Sequence
from typing import Any


def is_instance(obj: Any, prefix: str, name: str) -> bool:
    return is_subclass(type(obj), prefix, name)


def is_subclass(cls: type, prefix: str, name: str) -> bool:
    return cls.__module__.startswith(prefix) and cls.__name__ == name


def is_array_like(obj: Any) -> bool:
    # TODO: Implement a more robust check for array-like objects.
    return (
        is_jax(obj)
        or is_numpy(obj)
        or is_torch(obj)
        or is_warp(obj)
        or isinstance(obj, Sequence)
    )


# ----------------------------------- numpy ---------------------------------- #


def is_jax(obj: Any) -> bool:
    return is_subclass(obj, "jax", "Array")


def is_numpy(obj: Any) -> bool:
    return is_subclass(obj, "numpy", "ndarray")


def is_torch(obj: Any) -> bool:
    return is_subclass(obj, "torch", "Tensor")


def is_warp(obj: Any) -> bool:
    return is_subclass(obj, "warp", "array")


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
