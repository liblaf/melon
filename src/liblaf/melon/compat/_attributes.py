import numpy as np
import pyvista as pv
from jaxtyping import Integer

from liblaf import grapes


def get_group_id(obj: pv.DataSet) -> Integer[np.ndarray, " cells"]:
    __warnings_hide = True
    return grapes.getitem(
        obj.cell_data,
        "GroupId",
        deprecated_keys=[
            "group_id",
            "group_ids",
            "group-id",
            "group-ids",
            "GroupID",
            "GroupIds",
            "GroupIDs",
        ],
    )


def get_group_name(obj: pv.DataSet) -> list[str]:
    __warnings_hide = True
    return grapes.getitem(
        obj.field_data,
        "GroupName",
        deprecated_keys=[
            "group_name",
            "group_names",
            "group-name",
            "group-names",
            "GroupNames",
        ],
    ).tolist()
