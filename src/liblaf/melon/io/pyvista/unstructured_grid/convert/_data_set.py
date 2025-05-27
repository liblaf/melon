from typing import Any, override

from liblaf.melon import utils
from liblaf.melon.io import abc as _abc


class DataSetToUnstructuredGrid(_abc.AbstractConverter):
    @override
    def match_from(self, data: Any) -> bool:
        return utils.is_data_set(data)

    @override
    def convert(self, data: Any, /, **kwargs) -> Any:
        return data.cast_to_unstructured_grid(**kwargs)
