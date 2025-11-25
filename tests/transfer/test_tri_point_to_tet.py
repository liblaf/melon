import numpy as np
import pyvista as pv
from jaxtyping import Integer

from liblaf.melon import transfer


def test_tri_point_to_tet() -> None:
    rng: np.random.Generator = np.random.default_rng()
    target: pv.UnstructuredGrid = pv.examples.download_letter_a()  # pyright: ignore[reportAssignmentType]
    target.point_data["OrderedPointId"] = np.arange(target.n_points)
    target.point_data["UnorderedPointId"] = rng.permutation(np.arange(target.n_points))
    source: pv.PolyData = target.extract_surface()  # pyright: ignore[reportAssignmentType]
    result: pv.UnstructuredGrid = transfer.transfer_tri_point_to_tet(
        source, target, data=["OrderedPointId"], fill=0, point_id="UnorderedPointId"
    )
    surface_indices: Integer[np.ndarray, " S"] = source.point_data["OrderedPointId"]
    np.testing.assert_array_equal(
        result.point_data["OrderedPointId"][surface_indices],
        target.point_data["OrderedPointId"][surface_indices],
    )
