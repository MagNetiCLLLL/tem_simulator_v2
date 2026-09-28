"""Deterministic private-IPC fault fixture; does not simulate electron physics.

Only tests launch this helper through the backend's child-command hook. It
executes no field solve or integration. Long waits deliberately represent a
child which is alive but unresponsive and must be stopped by its owner.
"""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
import struct
import sys
import time

from temsim.test_electron_execution import RemoteElectronScene, _receive, _send


def fixture_result(*, unfinished=False):
    import numpy as np
    from temsim.magnetic_test_particle import TestElectronTrajectory

    scalar = np.array([0.])
    vector = np.array([[0., 0., 0.]])
    return TestElectronTrajectory(
        positions_m=vector, directions=np.array([[0., 0., 1.]]),
        time_s=scalar, path_length_m=scalar, energy_invariant_error_ev=0.,
        energy_invariant_relative_error=0., reason="in_progress" if unfinished else "path_limit",
        completed=not unfinished, steps=0, kinetic_energy_ev=np.array([.3]),
        electrostatic_potential_v=scalar, speed_m_per_s=scalar,
        momentum_kg_m_per_s=vector,
        notes=("Scripted protocol fixture only; no physical integration executed.",))


def wait_forever():
    while True:
        time.sleep(.02)


def main():
    identity, mode, marker = sys.argv[1:]
    output, input_stream = sys.stdout.buffer, sys.stdin.buffer
    bounds = ((-.001, -.001, 0.), (.001, .001, .01))
    token, traces = "fixture-scene", 0

    def reached(stage):
        Path(marker).write_text(stage, encoding="utf-8")
        print(f"FAULT_WORKER_ENTERED:{stage}", file=sys.stderr, flush=True)

    if mode == "ignore_input":
        reached("before_read")
        wait_forever()

    while True:
        try:
            sequence, command = _receive(input_stream)
        except EOFError:
            return 0
        kind = command[0]
        if kind == "ready":
            if mode == "ignore_startup":
                reached("startup")
                wait_forever()
            _send(output, (sequence, "ok", {"process_id": os.getpid(),
                "python_executable": sys.executable, "python_prefix": sys.prefix,
                "fixture": "No numerical work"}))
        elif kind in ("prepare", "install"):
            reached(kind)
            if mode == "exit_prepare":
                print("INTENTIONAL_PREPARE_EXIT_17", file=sys.stderr, flush=True)
                os._exit(17)
            if mode == "ignore_prepare":
                wait_forever()
            if mode == "bad_response_shape":
                _send(output, {"unexpected": "mapping"})
                continue
            if mode == "partial_frame":
                output.write(struct.pack("!Q", 100) + b"partial")
                output.flush()
                os._exit(23)
            if mode == "oversized_frame":
                output.write(struct.pack("!Q", 4*1024**3 + 1))
                output.flush()
                wait_forever()
            owner = "other-process" if mode == "wrong_owner" else identity
            returned_token = "" if mode == "empty_token" else token
            remote = RemoteElectronScene(returned_token, owner, (0., 0., 0.), .3, .01,
                                         bounds, bounds, ("Scripted lifecycle fixture",))
            if mode == "invalid_bounds":
                remote = replace(remote, bounds_m=(bounds[1], bounds[0]))
            if mode == "not_metadata":
                remote = {"token": token, "process_identity": identity}
            response_sequence = sequence + 1 if mode == "wrong_sequence" else sequence
            _send(output, (response_sequence, "ok", remote))
        elif kind == "trace":
            traces += 1
            reached("trace")
            if mode == "exit_trace":
                print("INTENTIONAL_TRACE_EXIT_19", file=sys.stderr, flush=True)
                os._exit(19)
            if mode == "python_traceback":
                raise RuntimeError("INJECTED_PYTHON_TRACEBACK_FROM_SCRIPTED_WORKER")
            if mode == "stderr_flood":
                block = b"FAULT_STDERR_FLOOD:" + b"x" * (16*1024 - 20) + b"\n"
                for _ in range(512):
                    sys.stderr.buffer.write(block)
                sys.stderr.buffer.write(b"\nFAULT_STDERR_FLOOD_FINISHED\n")
                sys.stderr.buffer.flush()
                os._exit(29)
            if mode == "ignore_cancel":
                wait_forever()
            if mode == "cooperative_cancel" and traces == 1:
                cancel_sequence, cancel = _receive(input_stream)
                if cancel_sequence != sequence or cancel != ("cancel",):
                    raise RuntimeError("Fixture expected cancellation of the active request")
                _send(output, (sequence, "cancelled", None))
                continue
            if mode == "slow_trace":
                time.sleep(.2)
            if mode == "bad_progress":
                _send(output, (sequence, "progress", fixture_result(unfinished=False)))
                wait_forever()
            if mode == "stale_progress":
                _send(output, (sequence - 1, "progress", fixture_result(unfinished=True)))
                wait_forever()
            if mode == "unrequested_progress":
                _send(output, (sequence, "progress", fixture_result(unfinished=True)))
                wait_forever()
            if mode == "terminal_prefix":
                _send(output, (sequence, "ok", fixture_result(unfinished=True)))
                continue
            if mode in ("nan_position", "wrong_direction_shape", "forged_execution_identity"):
                import numpy as np
                result = fixture_result()
                if mode == "nan_position":
                    result = replace(result, positions_m=np.array([[np.nan, 0., 0.]]))
                elif mode == "wrong_direction_shape":
                    result = replace(result, directions=np.zeros((1, 2)))
                else:
                    result = replace(result, execution_identity="forged-execution")
                _send(output, (sequence, "ok", result))
                continue
            _send(output, (sequence, "ok", fixture_result()))
        elif kind == "cancel":
            # A late cancellation cannot fabricate a successful current path.
            _send(output, (sequence, "cancelled", None))
        else:
            raise RuntimeError(f"Unknown fixture command: {kind}")


if __name__ == "__main__":
    raise SystemExit(main())
