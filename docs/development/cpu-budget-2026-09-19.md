# Numerical CPU budget

The default numerical budget is half the logical CPUs available to the process,
rounded down with a minimum of one. OS process affinity is respected. On the
current 32-logical-CPU host this is 16. This limits calculation threads; it does
not promise a constant 50% operating-system CPU-utilisation reading.

`TEMSIM_CPU_THREADS` can impose a lower process budget. Existing lower Numba or
OpenMP limits are preserved. Explicit job/benchmark limits are additionally
capped by the process budget. Limits apply in the actual numerical worker:
Numba's mask is thread local, so setting it only on the GUI thread is insufficient.

BLAS uses one thread. OpenMP nested active levels are disabled. Numerical GUI
jobs share a process-wide, reentrant admission lock even across separate job
coordinators, avoiding simultaneous full-budget jobs and process-global library
limit restoration races. Waiting jobs observe cancellation. Normal UI work and
unrelated external processes are outside this numerical budget.

Existing independent interval executors are capped at the same total budget;
their child workers use one Numba/BLAS thread. No coherent calculation is enabled
by this resource policy. Numerical equations, particle counts and tolerances
are unchanged.

The particle-stage benchmark uses this policy by default. `--threads N` is an
optional lower cap; reported Numba and BLAS counts are separate. Completed
5,000-particle timings using this policy are recorded in
[the gun report](gun-performance-2026-09-19.md) and
[the archive/continuation report](particle-speed-and-autosave-2026-09-19.md).

Validation covers affinity arithmetic, lower environment limits, thread-local
Numba masks inside Python and Qt workers, nested contexts, shared admission,
cancellation and exception-safe release. Tests use bounded scheduling fixtures,
not a claim of full physical-chain or performance qualification.
