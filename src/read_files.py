# Script to read in halo photonics file 

# import modules
import logging
import pkgutil
import re
from datetime import datetime, timezone
from io import BufferedReader, BytesIO
from pathlib import Path
from typing import Sequence

import lark
import numpy as np
import numpy.typing as npt
from lark.exceptions import UnexpectedInput

from haloreader.background_reader import read_background
from haloreader.data_reader import read_data
from haloreader.exceptions import BackgroundReadError
from haloreader.halo import Halo, HaloBg
from haloreader.metadata import Metadata
from haloreader.utils import UNIX_TIME_UNIT
from haloreader.variable import Variable

from haloreader.exceptions import (
    FileEmpty,
    HeaderNotFound,
    InconsistentRangeError,
    UnexpectedDataTokens,
)
from haloreader.transformer import HeaderTransformer

log = logging.getLogger(__name__)
import haloreader as hr
# from haloreader.common import Channel

# Load the Halo file
file_path = r"../data/Tauern/Stare_156_20210504_23.hpl"
halo_file = hr.data_reader.read_data(file_path, header_end=1)
src = file_path
_read_single(src)

def _range_consistent(range_var: Variable) -> bool:
    if not isinstance(range_var.dimensions, tuple):
        raise TypeError
    if not isinstance(range_var.data, np.ndarray):
        raise TypeError
    if len(range_var.dimensions) != 2 or range_var.dimensions[1] != "range":
        return False
    expected_range = np.arange(range_var.data.shape[1], dtype=range_var.data.dtype)
    if all(np.allclose(expected_range, r) for r in range_var.data):
        return True
    return False

def _decimaltime2timestamp(time: Variable, metadata: Metadata) -> Variable:
    if time.long_name != "decimal time" or time.units != "hours":
        raise NotImplementedError
    if metadata.start_time.units != UNIX_TIME_UNIT:
        raise NotImplementedError
    day_in_seconds = 86400
    hour_in_seconds = 3600
    if (
        not isinstance(metadata.start_time.data, np.ndarray)
        or np.ndim(metadata.start_time.data) != 1
        or metadata.start_time.data.size != 1
    ):
        raise TypeError
    t_start = np.floor(metadata.start_time.data / day_in_seconds) * day_in_seconds
    if not isinstance(time.data, np.ndarray):
        raise TypeError
    time_ = t_start + hour_in_seconds * time.data
    i_day_changed = _find_change_of_day(0, time_)
    while i_day_changed >= 0:
        time_[i_day_changed:] += day_in_seconds
        i_day_changed = _find_change_of_day(i_day_changed, time_)
    return Variable(
        name="time",
        long_name="time",
        calendar="standard",
        data=time_,
        dimensions=("time",),
        units=UNIX_TIME_UNIT,
    )
def _read_data(src: Path | BytesIO, header_end: int) -> bytes:
    if isinstance(src, Path):
        with src.open("rb") as src_buf:
            src_buf.seek(header_end)
            return src_buf.read()
    else:
        src.seek(header_end)
        return src.read()

def _find_change_of_day(start: int, time: npt.NDArray) -> int:
    half_day = 43200
    for i, (time_current, time_next) in enumerate(
        zip(time[start:-1], time[start + 1 :])
    ):
        if time_current - time_next > half_day:
            return i + 1
    return -1

def _read_header(src: Path | BytesIO) -> tuple[int, bytes]:
    if isinstance(src, Path):
        with src.open("rb") as src_buf:
            return _read_header_from_bytes(src_buf)
    else:
        return _read_header_from_bytes(src)


def _read_single(src: Path | BytesIO) -> Halo:
    header_end, header_bytes = _read_header(src)
    metadata, time_vars, time_range_vars, range_func = header_parser.parse(
        header_bytes.decode()
    )
    log.info("Reading data from %s", metadata.filename.value)
    data_bytes = _read_data(src, header_end)
    if not isinstance(metadata.ngates.data, int):
        raise TypeError
    read_data(data_bytes, metadata.ngates.data, time_vars, time_range_vars)
    vars_ = {var.name: var for var in time_vars + time_range_vars}
    if not _range_consistent(vars_["range"]):
        raise InconsistentRangeError
    vars_["time"] = _decimaltime2timestamp(vars_["time"], metadata)
    vars_["range"] = range_func(vars_["range"], metadata.gate_range)
    return Halo(metadata=metadata, **vars_)

def _read_header_from_bytes(
    src: BufferedReader | BytesIO,
) -> tuple[int, bytes]:
    header_end = _find_header_end(src)
    if header_end < 0:
        src.seek(0)
        if len(src.read()) == 0:
            raise FileEmpty
        raise HeaderNotFound
    header_bytes = src.read(header_end)
    return header_end, header_bytes


