I am developing a Python library for 3D data processing. I want to design a modular and extensible I/O system that can handle various 3D data formats and libraries. The system should allow easy addition of new formats and libraries in the future.

The I/O system should include the following features:

- convenient conversion between different 3D data classes (e.g., `pyvista.PolyData`, `trimesh.Trimesh`, etc.), handle inheritance correctly
- load and save different formats using appropriate libraries
- use of type hints for better code clarity and maintainability

### Conversion

I want the following API: `as_polydata(obj: Any, **kwargs) -> pv.PolyData`, `as_pointset(obj: Any, **kwargs) -> pv.PointSet`, etc. The `obj` parameter can be of various types, and the function should convert it to the desired type if possible.

### Loading

I want the following API: `load_polydata(path: Path, **kwargs) -> pv.PolyData`, `load_trimesh(path: Path, **kwargs) -> tm.Trimesh`, etc. The `path` parameter is the file path to load the data from (`.ply`, `.obj`, etc.), and the function should use the appropriate library to read the file and return the data in the desired format. Specially, DICOM files can be a folder with a `DIRFILE` file.

### Saving

I want the following API: `save(path: Path, obj: Any, **kwargs) -> None`. The `path` parameter is the file path to save the data to, and the `obj` parameter is the data to be saved. The function should determine the appropriate library and format based on the file extension and the type of `obj` and save the data accordingly. For example, to save a `pv.PolyData` object to a `.ply` file, the function should use `pyvista` to write the file; to save a `tm.Trimesh` object to a `.ply` file, the function should use `trimesh` to write the file.

### Smart

If `save()` is implemented for (`.ply`, `pv.PolyData`) but not for (`.ply`, `tm.Trimesh`), and `as_polydata` is implemented for `tm.Trimesh`, then `save(path: Path, obj: tm.Trimesh, **kwargs)` should automatically convert `obj` to `pv.PolyData` and save it as a `.ply` file. If `as_pytorch3d` is implemented for `pv.PolyData` but not for `tm.Trimesh`, and `as_polydata` is implemented for `tm.Trimesh`, then `as_pytorch3d(obj: tm.Trimesh, **kwargs)` should automatically convert `obj` to `pv.PolyData` and then to `pytorch3d`. Similarly for `load()`. This should be done in a way that avoids infinite loops and is efficient.

Inheritance should be handled correctly in the conversion functions. For example, if `CustomPolyData` is a subclass of `pv.PolyData` and saving and conversion is implemented for `pv.PolyData`, then `save()` and conversion to `tm.Trimesh` should be able to handle `CustomPolyData` objects without requiring a separate registration. Something like `functools.singledispatch`?

### Extensibility

The I/O system should be designed in a way that allows easy addition of new formats and libraries in the future. This could involve using a registry pattern or a plugin system to manage different loaders and savers.

How to design this I/O system in Python?
