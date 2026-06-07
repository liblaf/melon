# Melon

Melon is a small mesh-processing toolbox for Python workflows that move between
PyVista, Trimesh, MeshIO, Warp, TetWild, PyMeshFix, and Faceform Wrap. The
library keeps most behavior behind dispatchers: readers dispatch by file suffix,
writers dispatch by suffix and object type, and converters dispatch by source
runtime type.

## Core Workflows

- Use `liblaf.melon.io` to load PyVista datasets, save registered mesh objects,
  and convert meshes between PyVista, Trimesh, MeshIO, and Warp.
- Use `liblaf.melon.io.wrap` for Wrap-compatible landmark and polygon-selection
  JSON sidecars.
- Use `liblaf.melon.tri` for triangular surface selection, edge lengths, normal
  repair, and geodesic paths.
- Use `liblaf.melon.tet` for tetrahedral winding repair and sampled volume
  fractions.
- Use `liblaf.melon.xfer` to move arrays from triangle cells to triangle points
  and from triangle points to tetrahedral points.
- Use `liblaf.melon.ext` when workflows need PyMeshFix, TetWild, or Faceform
  Wrap command-line tools.

## Example

```python
import numpy as np
import pyvista as pv

from liblaf.melon import io, tet, tri

surface = pv.Sphere(theta_resolution=16, phi_resolution=16)
io.save(surface, "surface.vtp")

mesh = io.load_polydata("surface.vtp")
lengths = tri.edge_length(mesh)

tetmesh = pv.UnstructuredGrid(
    np.array([4, 0, 1, 2, 3]),
    np.array([pv.CellType.TETRA]),
    np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
)
tetmesh = tet.fix_winding(tetmesh)
```

## Validation

The maintained package surface is covered by doctests and focused pytest tests:

```bash
UV_FROZEN=1 uv run pytest -q
UV_FROZEN=1 mise run lint
UV_FROZEN=1 mise run docs:build
```
