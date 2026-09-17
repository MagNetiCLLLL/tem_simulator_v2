"""Bounded owned validation processes, including Windows venv launchers."""
import os
import signal
import subprocess


def run_bounded(command, *, timeout, **kwargs):
    """Return CompletedProcess; 124 denotes an owned process-tree timeout.

    Windows venv python.exe can launch another interpreter. Killing just the
    launcher leaves pytest running, so terminate that exact owned PID tree.
    No process-name search, broad kill or user-application control is used.
    """
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with subprocess.Popen(command, **options, **kwargs) as child:
        try:
            child.wait(timeout=timeout)
            code = child.returncode
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               check=False, timeout=15)
            else:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            child.wait(timeout=15)
            code = 124
        return subprocess.CompletedProcess(command, code)
