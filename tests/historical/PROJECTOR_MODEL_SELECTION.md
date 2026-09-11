# Historical projector-model selection expectation

The baseline `5eac9855ff8eefa2a3d68ddba3029f7c23e03b0c` test
`tests/test_direct_alignment.py::test_image_working_points_commit_the_five_lens_solution`
requested magnifications `10, 100, 1000, 10000, 100000, 1000000` after applying
the nano/imaging operating mode, without explicitly enabling the equivalent
image-lens model. It expected a successful solution to enable that model.

HANDOFF v2 forbids a solve from silently selecting a different physical model.
The current equivalent-model sweep therefore sets
`equivalent_image_lenses_enabled = True` **before** the request and has been
renamed to make this scope explicit. The old target values, three-percent
achieved-value tolerance, relay constraint and permitted five-lens changes
remain unchanged. This does not establish that the distributed-field model
can reach these targets, nor qualify a coherent source-to-image workflow.

The original broad-audit failures remain recorded in
`.pytest_cache/handoff-regression-audit.xml` and the HANDOFF acceptance ledger.
The original test implementation remains in the fixed Git baseline; it is not
claimed as a new passing production-chain test.
