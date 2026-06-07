<div align="center" markdown>
<a name="readme-top"></a>

![Melon](https://socialify.git.ci/liblaf/melon/image?description=1&forks=1&issues=1&language=1&logo=https%3A%2F%2Fraw.githubusercontent.com%2Fmicrosoft%2Ffluentui-emoji%2Frefs%2Fheads%2Fmain%2Fassets%2FWatermelon%2F3D%2Fwatermelon_3d.png&name=1&owner=1&pattern=Transparent&pulls=1&stargazers=1&theme=Auto)

**[Explore the docs »](https://liblaf.github.io/melon/)**

[![codecov](https://codecov.io/gh/liblaf/melon/graph/badge.svg)](https://codecov.io/gh/liblaf/melon)
[![MegaLinter](https://github.com/liblaf/melon/actions/workflows/shared-mega-linter.yaml/badge.svg)](https://github.com/liblaf/melon/actions/workflows/mega-linter.yaml)
[![Test](https://github.com/liblaf/melon/actions/workflows/python-test.yaml/badge.svg)](https://github.com/liblaf/melon/actions/workflows/test.yaml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/liblaf/melon/main.svg)](https://results.pre-commit.ci/latest/github/liblaf/melon/main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/liblaf-melon?logo=PyPI&label=Downloads)](https://pypi.org/project/liblaf-melon)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/liblaf-melon?logo=Python&label=Python)](https://pypi.org/project/liblaf-melon)
[![PyPI - Version](https://img.shields.io/pypi/v/liblaf-melon?logo=PyPI&label=PyPI)](https://pypi.org/project/liblaf-melon)

[Changelog](https://github.com/liblaf/melon/blob/main/CHANGELOG.md) · [Report Bug](https://github.com/liblaf/melon/issues) · [Request Feature](https://github.com/liblaf/melon/issues)

![Rule](https://cdn.jsdelivr.net/gh/andreasbm/readme/assets/lines/rainbow.png)

</div>

## ✨ Features

- 🧭 **Dispatch-based mesh I/O**: Read PyVista datasets, convert between PyVista, Trimesh, MeshIO, and Warp meshes, and write registered objects by file suffix.
- 🗂️ **Wrap sidecars**: Load and save Faceform Wrap landmark and polygon-selection JSON files with predictable sidecar naming.
- 📐 **Surface helpers**: Select named OBJ groups, extract surfaces, compute edge lengths, orient normals, and compute geodesic paths.
- 🧱 **Tetrahedral helpers**: Repair tetra winding and estimate per-cell volume fractions against a closed triangular surface.
- 🔁 **Data transfer**: Move arrays from triangle cells to triangle points, then from triangle points to tetrahedral points.
- 🛠️ **External integrations**: Wrap PyMeshFix, TetWild, and Faceform Wrap project templates while keeping the core APIs small.

## 📦 Installation

```bash
uv add liblaf-melon
```

Melon requires Python 3.12 or newer and pulls in mesh-processing libraries such as PyVista, Trimesh, MeshIO, Warp, and PyTetWild.

## 🚀 Quick Start

```python
import numpy as np
import pyvista as pv

from liblaf.melon import io, tet, tri, xfer

surface = pv.Sphere(theta_resolution=16, phi_resolution=16)
io.save(surface, "surface.vtp")

loaded = io.load_polydata("surface.vtp")
edge_lengths = tri.edge_length(loaded)

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

loaded.point_data["distance"] = np.linalg.norm(loaded.points, axis=1)
xfer.tri_point_to_tet_point(loaded, tetmesh, {"distance": -1.0})
```

## ⌨️ Local Development

You can use GitHub Codespaces for online development:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/liblaf/melon)

Or clone it for local development:

```bash
gh repo clone liblaf/melon
cd melon
mise run install
UV_FROZEN=1 uv run pytest -q
UV_FROZEN=1 mise run lint
UV_FROZEN=1 mise run docs:build
```

## 🤝 Contributing

Contributions of all types are welcome. For bugs, API ideas, or mesh workflow improvements, open an issue on GitHub.

[![PR Welcome](https://img.shields.io/badge/%F0%9F%A4%AF%20PR%20WELCOME-%E2%86%92-ffcb47?labelColor=black&style=for-the-badge)](https://github.com/liblaf/melon/pulls)

[![Contributors](https://contrib.nn.ci/api?repo=liblaf/melon)](https://github.com/liblaf/melon/graphs/contributors)

## 🔗 More Projects

- **[🍇 Grapes](https://github.com/liblaf/grapes)** - Supercharge your Python with rich logging, precise timing, and seamless serialization.
- **[🍊 Tangerine](https://github.com/liblaf/tangerine)** - Squeeze dynamic content into your files with Tangerine's template magic.
- **[🍎 Apple](https://github.com/liblaf/apple)** - Differentiable physics simulation with elastic energy models and finite elements.
- **[🍒 Cherries](https://github.com/liblaf/cherries)** - Sweet experiment tracking with Comet, DVC, and Git integration.

---

#### 📝 License

Copyright © 2025 [liblaf](https://github.com/liblaf). <br />
This project is [MIT](https://github.com/liblaf/melon/blob/main/LICENSE) licensed.
