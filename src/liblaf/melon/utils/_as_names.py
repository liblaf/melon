from collections.abc import Collection, Iterable

import pyvista as pv


def as_names(
    names: str | Iterable[str] | None, attributes: pv.DataSetAttributes | None = None
) -> Collection[str]:
    if names is None:
        if attributes is None:
            return []
        return attributes.keys()
    if isinstance(names, str):
        return [names]
    if isinstance(names, Collection):
        return names
    return list(names)
