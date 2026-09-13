"""Retain untruncated subprocess diagnostics for long reference calculations."""
import subprocess
import sys
from pathlib import Path


def run_logged_driver(command, *, cwd, env, log_path):
    """Stream a trusted driver and retain raw stdout/stderr without overwrite."""
    with Path(log_path).open("xb") as log:
        with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as process:
            for line in iter(process.stdout.readline, b""):
                log.write(line)
                log.flush()
                sys.stdout.write(line.decode("utf-8", errors="replace"))
                sys.stdout.flush()
            return process.wait()
