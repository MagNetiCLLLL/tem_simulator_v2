# Interrupted diagnostic, not a result

The first energy was still preparing its gun operators (last observed progress:
704/1381, approximately 407 seconds). The owned diagnostic processes were
stopped before changing the aperture integration implementation. No complete
gun checkpoint, TEM image or STEM image was produced by this run.

Reason: a separately tested closed-form Laguerre aperture integral agrees with
the positive high-order reference to approximately 1.7e-14 for the tested
128-mode matrices while avoiding repeated costly quadrature. A new evidence
directory will be used for the rerun; this directory is not a PASS receipt.
