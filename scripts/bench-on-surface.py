from collections.abc import Generator
from typing import no_type_check

import attrs
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import trimesh as tm
import warp as wp
from jaxtyping import Float, Integer
from liblaf.grapes.bench import Bencher, BenchResults

from liblaf import melon


@attrs.define
class OnSurfaceResults:
    barycentric: Float[np.ndarray, "N 3"]
    closest: Float[np.ndarray, "N 3"]
    distance: Float[np.ndarray, " N"]
    triangle_id: Integer[np.ndarray, " N"]


def on_surface_trimesh(
    mesh: pv.PolyData, points: Float[np.ndarray, "N 3"]
) -> OnSurfaceResults:
    mesh_tm: tm.Trimesh = melon.as_trimesh(mesh)
    closest: Float[np.ndarray, "N 3"]
    distance: Float[np.ndarray, " N"]
    triangle_id: Integer[np.ndarray, " N"]
    closest, distance, triangle_id = mesh_tm.nearest.on_surface(points)
    barycentric: Float[np.ndarray, "N 3"] = tm.triangles.points_to_barycentric(
        mesh_tm.vertices[mesh_tm.faces[triangle_id]], closest
    )
    return OnSurfaceResults(
        barycentric=barycentric,
        closest=closest,
        distance=distance,
        triangle_id=triangle_id,
    )


@wp.kernel
@no_type_check
def contains_warp_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    max_dist: wp.float32,
    # outputs
    barycentric: wp.array(dtype=wp.vec3),
    closest: wp.array(dtype=wp.vec3),
    distance: wp.array(dtype=wp.float32),
    triangle_id: wp.array(dtype=wp.int32),
) -> None:
    tid = wp.tid()
    point = points[tid]
    query = wp.mesh_query_point_no_sign(mesh_id, point, max_dist)
    barycentric[tid] = wp.vector(
        query.u, query.v, type(query.u)(1.0) - query.u - query.v
    )
    closest[tid] = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    distance[tid] = wp.length(closest[tid] - point)
    triangle_id[tid] = query.face


def contains_warp(
    mesh: pv.PolyData, points: Float[np.ndarray, "N 3"]
) -> OnSurfaceResults:
    mesh_wp: wp.Mesh = wp.Mesh(
        wp.from_numpy(mesh.points, wp.vec3),
        wp.from_numpy(mesh.regular_faces, wp.int32).flatten(),
    )
    barycentric_wp: wp.array = wp.zeros(points.shape[0], dtype=wp.vec3)
    closest_wp: wp.array = wp.zeros(points.shape[0], dtype=wp.vec3)
    distance_wp: wp.array = wp.zeros(points.shape[0], dtype=wp.float32)
    triangle_id_wp: wp.array = wp.zeros(points.shape[0], dtype=wp.int32)
    wp.launch(
        contains_warp_kernel,
        dim=(points.shape[0],),
        inputs=[mesh_wp.id, wp.from_numpy(points, wp.vec3), mesh.length],
        outputs=[barycentric_wp, closest_wp, distance_wp, triangle_id_wp],
    )
    return OnSurfaceResults(
        barycentric=barycentric_wp.numpy(),
        closest=closest_wp.numpy(),
        distance=distance_wp.numpy(),
        triangle_id=triangle_id_wp.numpy(),
    )


def main() -> None:
    mesh: pv.PolyData = pv.examples.download_bunny()  # pyright: ignore[reportAssignmentType]
    mesh.clean(inplace=True)
    mesh = melon.mesh_fix(mesh)

    bencher = Bencher(timeout=10.0, warmup=1)

    @bencher.setup
    def _() -> Generator[tuple[tuple[Float[np.ndarray, "N 3"]], dict]]:
        generator: np.random.Generator = np.random.default_rng()
        for n in np.logspace(1, 8, num=10, dtype=int):
            points: Float[np.ndarray, "N 3"] = generator.uniform(
                low=mesh.bounds[::2], high=mesh.bounds[1::2], size=(n, 3)
            )
            yield (points,), {}

    @bencher.size
    def _(points: Float[np.ndarray, "N 3"]) -> int:
        return points.shape[0]

    @bencher.bench(label="trimesh")
    def _(points: Float[np.ndarray, "N 3"]) -> OnSurfaceResults:
        return on_surface_trimesh(mesh, points)

    @bencher.bench(label="warp")
    def _(points: Float[np.ndarray, "N 3"]) -> OnSurfaceResults:
        return contains_warp(mesh, points)

    results: BenchResults = bencher.run()

    for label, outputs in results.outputs.items():
        if label == "trimesh":
            continue
        for size, actual, expected in zip(
            results.sizes, outputs, results.outputs["trimesh"], strict=True
        ):
            if actual is None or expected is None:
                continue
            actual: OnSurfaceResults
            expected: OnSurfaceResults
            ic(
                size,
                np.count_nonzero(actual.triangle_id != expected.triangle_id) / size,
            )
            np.testing.assert_allclose(actual.barycentric, expected.barycentric)
            np.testing.assert_allclose(actual.closest, expected.closest)
            np.testing.assert_allclose(actual.distance, expected.distance)

    results.plot()
    plt.show()


if __name__ == "__main__":
    main()
