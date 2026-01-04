import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, Float

from liblaf.melon.tet import compute_volume_fraction


def test_compute_volume_fraction() -> None:
    tetmesh: pv.UnstructuredGrid = pv.examples.download_tetrahedron()  # pyright: ignore[reportAssignmentType]
    surface: pv.PolyData = pv.Icosphere()
    volume_fraction: Float[Array, " cells"] = compute_volume_fraction(tetmesh, surface)
    assert volume_fraction.shape == (tetmesh.n_cells,)
    assert jnp.all((volume_fraction >= 0.0) & (volume_fraction <= 1.0))
