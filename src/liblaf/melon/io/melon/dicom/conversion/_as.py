from typing import Any

from liblaf.melon import struct
from liblaf.melon.io import abc


def as_dicom(obj: Any) -> struct.DICOM:
    return abc.convert(obj, struct.DICOM)
