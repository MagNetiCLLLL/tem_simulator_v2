"""Private temporary transport, not a persistent cache or external importer.

Windows named-pipe writes can fail with WinError 1450 for large serialized
operators even with available RAM. Workers exchange small file descriptors;
only their own execution's cloudpickle streams are read. Files are owned by
one TemporaryDirectory and removed after all workers have completed.
"""
from pathlib import Path

import cloudpickle


def write_packet(path, value):
    path = Path(path)
    with path.open("xb") as stream:
        cloudpickle.dump(value, stream, protocol=5)
    return str(path), path.stat().st_size


def read_packet(descriptor):
    path, size = descriptor
    path = Path(path)
    if type(size) is not int or size < 0 or path.stat().st_size != size:
        raise ValueError("Incomplete private wave-process packet")
    with path.open("rb") as stream:
        result = cloudpickle.load(stream)
        if stream.read(1):
            raise ValueError("Unexpected trailing private wave-process data")
    return result


def read_worker_result(descriptor):
    success, value, traceback = read_packet(descriptor)
    if not success:
        value.worker_traceback = traceback
        raise value
    return value
