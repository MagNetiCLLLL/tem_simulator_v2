"""Bounded owned validation processes, including Windows venv launchers."""
from dataclasses import dataclass
import math
import os
import signal
import subprocess
from threading import RLock
from time import monotonic


_CLEANUP_SECONDS = 15.
_FAILED_CLEANUPS = []
_CLEANUP_LOCK = RLock()


class ValidationCleanupError(subprocess.SubprocessError):
    """An owned process or its tree could not be confirmed stopped."""

    def __init__(self, pid, detail, *, tree_termination_confirmed=False):
        self.pid = pid
        self.tree_termination_confirmed = bool(tree_termination_confirmed)
        self.detail = detail
        super().__init__(f"Owned validation PID {pid}: cleanup could not be confirmed; {detail}. "
                         "Further validation is blocked. Verify and clear the owned process tree "
                         "before restarting the validation driver.")


@dataclass
class _FailedCleanup:
    child: object
    tree_termination_confirmed: bool
    detail: str


def _remember_failure(child, tree_confirmed, detail):
    with _CLEANUP_LOCK:
        if not any(item.child is child for item in _FAILED_CLEANUPS):
            _FAILED_CLEANUPS.append(_FailedCleanup(child, tree_confirmed, detail))


def _check_failed_cleanups():
    with _CLEANUP_LOCK:
        # poll() reaps a now-finished direct child without an unbounded wait.
        # Root exit alone does not prove that an unsuccessfully killed tree is
        # gone. Never retry taskkill against a potentially recycled dead PID.
        _FAILED_CLEANUPS[:] = [item for item in _FAILED_CLEANUPS
                              if item.child.poll() is None or not item.tree_termination_confirmed]
        if _FAILED_CLEANUPS:
            item = _FAILED_CLEANUPS[0]
            raise ValidationCleanupError(item.child.pid, item.detail,
                                         tree_termination_confirmed=item.tree_termination_confirmed)


def _remaining(deadline):
    return max(0., deadline - monotonic())


def _stop_direct(child, deadline, errors):
    """Bounded last resort for the exact Popen-owned handle, never a PID search."""
    for name in ("terminate", "kill"):
        if child.poll() is not None:
            return
        try:
            getattr(child, name)()
        except OSError as error:
            errors.append(f"{name} failed: {error}")
        try:
            child.wait(timeout=min(.5, _remaining(deadline)))
        except (OSError, subprocess.TimeoutExpired) as error:
            errors.append(f"bounded wait after {name} failed: {error}")


def _taskkill_tree(pid, deadline):
    # subprocess.run's context-manager exit can itself wait without a timeout
    # if killing its helper fails. Own this short-lived utility explicitly too.
    utility = subprocess.Popen(["taskkill", "/PID", str(pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        return utility.wait(timeout=_remaining(deadline))
    except (OSError, subprocess.TimeoutExpired) as error:
        details = [f"taskkill utility PID {utility.pid} failed: {error}"]
        _stop_direct(utility, deadline, details)
        if utility.poll() is None:
            _remember_failure(utility, True, "; ".join(details))
        raise subprocess.SubprocessError("; ".join(details)) from error


def _terminate_owned_tree(child):
    deadline = monotonic() + _CLEANUP_SECONDS
    errors = []
    tree_confirmed = False
    try:
        if os.name == "nt":
            code = _taskkill_tree(child.pid, deadline)
            tree_confirmed = code == 0
            if not tree_confirmed:
                errors.append(f"taskkill /T /F returned {code}")
        else:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            tree_confirmed = True
    except (OSError, subprocess.SubprocessError) as error:
        errors.append(f"owned-tree termination failed: {error}")
    try:
        child.wait(timeout=_remaining(deadline))
    except (OSError, subprocess.TimeoutExpired) as error:
        errors.append(f"bounded final wait failed: {error}")
        _stop_direct(child, deadline, errors)
    if child.poll() is None or not tree_confirmed:
        detail = "; ".join(errors) or "the owned process remains alive"
        _remember_failure(child, tree_confirmed, detail)
        raise ValidationCleanupError(child.pid, detail, tree_termination_confirmed=tree_confirmed)


def _close_owned_pipes(child, options):
    # Popen only creates these pipe wrappers for PIPE. A log stream supplied by
    # the caller remains the caller's property, on both success and failure.
    for name in ("stdin", "stdout", "stderr"):
        if options.get(name) == subprocess.PIPE:
            stream = getattr(child, name, None)
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass


def run_bounded(command, *, timeout, **kwargs):
    """Return CompletedProcess; 124 means timed out and owned-tree kill confirmed.

    Failed cleanup raises with the exact owned PID and blocks later validation
    in this process. No process-name search or unbounded Popen context exit is
    used. Windows venv launchers still receive owned-tree termination.
    """
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Validation timeout must be finite and positive")
    options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
               if os.name == "nt" else {"start_new_session": True})
    with _CLEANUP_LOCK:
        _check_failed_cleanups()
        child = subprocess.Popen(command, **options, **kwargs)
    try:
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate_owned_tree(child)
            return subprocess.CompletedProcess(command, 124)
        except BaseException:
            _terminate_owned_tree(child)
            raise
        return subprocess.CompletedProcess(command, child.returncode)
    finally:
        _close_owned_pipes(child, kwargs)
