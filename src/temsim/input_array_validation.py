"""Validate embedded NPY/NPZ lengths before any array allocation."""
from io import BytesIO
import math
from pathlib import PurePosixPath
import stat
from zipfile import ZipFile

import numpy as np


def _header(stream, length, maximum):
    version = np.lib.format.read_magic(stream)
    reader = {(1, 0): np.lib.format.read_array_header_1_0,
              (2, 0): np.lib.format.read_array_header_2_0}.get(version)
    if reader is None:
        raise ValueError("Unsupported archived input array header")
    shape, _, dtype = reader(stream, max_header_size=16384)
    if (len(shape) > 32 or any(type(n) is not int or n < 0 for n in shape)
            or dtype.kind not in "biufcSU" or dtype.fields is not None or dtype.itemsize <= 0):
        raise ValueError("Archived input arrays require bounded plain numeric/text values")
    count = math.prod(shape)
    size = count * dtype.itemsize
    if count > np.iinfo(np.intp).max or size > maximum or stream.tell() + size != length:
        raise ValueError("Archived input array length exceeds its payload or budget")


def validate_input_arrays(path, content, maximum):
    suffix = path.suffix.lower()
    if suffix == ".npy":
        _header(BytesIO(content), len(content), maximum)
    elif suffix == ".npz":
        with ZipFile(BytesIO(content)) as archive:
            entries = archive.infolist()
            names = {entry.filename for entry in entries}
            if (len(names) != len(entries) or len(entries) > 10000
                    or sum(entry.file_size for entry in entries) > maximum):
                raise ValueError("Archived input NPZ inventory exceeds its budget or repeats entries")
            for entry in entries:
                name = PurePosixPath(entry.filename)
                if (name.is_absolute() or ".." in name.parts or "\\" in entry.filename
                        or ":" in entry.filename or name.as_posix() != entry.filename
                        or name.suffix != ".npy" or entry.is_dir()
                        or stat.S_ISLNK(entry.external_attr >> 16)):
                    raise ValueError("Unsafe archived input NPZ entry")
                with archive.open(entry) as stream:
                    _header(stream, entry.file_size, maximum)
