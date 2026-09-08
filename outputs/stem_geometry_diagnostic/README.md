# STEM image diagnosis: finite ray counts and display range

The reported HAADF/DF/BF levels are consistent with the project's geometric or material-particle STEM preview. High accuracy increases optical ray sampling but **does not enable the separately selected STEM wave model**. The production request capture was checked with that switch both off and on; it preserved each value at 15,000 rays with a valid CIF selected.

| Reported channel | Equal-weight ray interpretation | Fraction of emitted current |
| --- | --- | --- |
| HAADF | 16 / 15,000, constant | 0.001066666667 |
| DF | 2,423 / 15,000, constant | 0.161533333333 |
| BF lower / upper | 3,531 / 15,000 and 3,532 / 15,000 | 0.2354 / 0.235466666667 |

The BF difference is one ray, an absolute fraction change of 0.000066666667 and a relative peak-to-peak change of **0.0283166%**. Independent minimum/maximum grayscale mapping nevertheless makes the lower value black and upper value white. Constant positive HAADF/DF values can also appear black when mapped to a display range starting at their exact value. Neither appearance implies zero measured signal.

At 32 × 32 pixels, 0.02 nm pitch gives a 0.64 nm pixel-edge field of view; 0.05 nm gives 1.6 nm. Changing pitch changes the scan geometry, but does not activate coherent CIF propagation.

The preview path may use CIF-derived composition and density in the finite-specimen particle transport calculated at a reference scan point. It then routes those particles through shifted detector masks. It does **not** propagate the atomic CIF potential at every pixel, so detector clipping boundaries are not atomic contrast. The previous blanket notice “selected CIF not used” was therefore too broad when material-particle transport was actually used.

Evidence and limits:

- `diagnostic.json` records the real request-capture checks and actual latest cached incident weights, all equal to 1/15,000.
- `labelled_counting_fixture.npz` is an explicitly labelled demonstration constructed from the screenshot-reported levels, **not a recovered frame or simulated specimen image**. No complete saved screenshot state or raw STEM frame was available in the incident artifact cache.
- Reproduce the lightweight diagnosis with `.venv\Scripts\python.exe scripts/diagnose_stem_ray_quantization.py` from the project root. It performs no new column trace or multislice acquisition.
- The UI regression should preserve raw arrays, show explicit constant values, use a fixed 0–1 range for nonconstant geometry/particle previews, preserve ordinary wave-image auto contrast, and retain each frame's captured metadata when paused or displayed from the bank. These cases are now covered in `tests/test_stem_image_presentation.py` by the UI agent.

Production locations examined: `gui/calculation_request.py`, `optics/model.py`, `detector/stem_signal.py`, `gui/scan_panel.py`. The wave flag participates in the STEM cache identity, so the reported High accuracy label alone is not evidence of a cache collision or of the wave model having been calculated.
