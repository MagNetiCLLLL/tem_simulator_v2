# TEM projector and Super-X / Ultra-X EDS geometry research

Research date: 2026-08-30. English edition: 2026-09-05.

This is a historical research record, not a fresh verification of all sources
or current implementation status. Preserve it for later updates from user-supplied
projector cross sections, EDS component drawings and scaled photographs.
Separate manufacturer specifications, patent/paper structures, single-instrument
metadata and non-OEM reconstruction. The latter two are not manufacturer dimensions.

## 1. Projector pole topology

Update, 2026-09-05: see [magnetic circuit models](MAGNETIC_CIRCUIT_MODELS.md).
FEI US9595359B2 explicitly describes a monolithic saturation insert without a
traditional pole gap; its Figure 6 compares gapped and ungapped simulations.
Neither that patent nor the shared-pole examples establish Titan production
topology. Air-core coils can also focus electrons without ferromagnetic poles.
The shipped independent D/I/P1/P2 structures remain unverified engineering
assumptions; no dimensions were replaced using patent drawing proportions.

1. D, I, P1 and P2 are modelled as axisymmetric magnetic round lenses. A magnetic
   circuit with pole faces/gaps is consistent with localised axial fields;
   describing the projector system simply as having no pole pieces is misleading.
2. A named optical control does not necessarily correspond to an independent
   upper pole, lower pole and yoke. Patents describe three-pole/two-gap/two-coil
   projectors with shared middle poles, as well as compound pole assemblies.
   Control-channel names therefore do not determine the number of separate parts.
3. Each current D/I/P1/P2 TOML uses two_pole_single_gap: an editable principle
   model compatible with distributed fields, not a production FEI/Thermo Fisher section.
4. Do not merge/delete projector pole children before service sections or
   disassembly evidence establish shared poles/yokes, coil count, gap locations,
   vacuum continuity and assembly boundaries.

Sources:

- [US4450357A: three-pole, two-gap, two-coil projector](https://patents.google.com/patent/US4450357A/en)
- [US2472315A: compound objective/projector poles](https://patents.google.com/patent/US2472315A/en)
- [US5304801A: intermediate/projector excitation channels](https://patents.google.com/patent/US5304801A/en)

These establish possible topologies, not which patented embodiment or dimensions
a current Thermo Fisher instrument uses.

## 2. Super-X: documented geometry

The published FEI Super-X model describes four symmetric windowless SDDs around
the specimen, each with 30 mm² active area (120 mm² total). Aggregate acceptance
is approximately 0.7 or 0.9 sr depending on objective pole geometry. One
experimentally checked model uses an 18-degree take-off angle and azimuths
45, 135, 225 and 315 degrees. That is the paper's instrument geometry, not a
universal OEM angle. Holder, grid, tilt and poles affect each detector's collection.

Active area is not package cross section. An equivalent circular 30 mm² sensor
would have diameter 2*sqrt(30/pi)=6.18 mm. Under the additional assumptions of
sample-facing disks, no collimator and equal solid-angle division, equivalent
distances are approximately 12.8/11.2 mm for 0.7/0.9 sr. Shape, tilt and
collimation change this inference; do not store it as a measured mechanical distance.

Sources:

- [Ultramicroscopy 164 (2016), 51–61](https://doi.org/10.1016/j.ultramic.2016.02.004)
- [US8410439B2: multi-detector TEM objective geometry](https://patents.google.com/patent/US8410439B2/en)

## 3. Ultra-X: knowns, unknowns and model choices

Recorded manufacturer information: windowless Ultra-X; greater than 4.45 sr
without holder shadowing; 4.04 sr with an analytical double-tilt holder;
zero-tilt sensitivity approximately six times Super-X and 2.5 times Dual-X;
paired with a large-gap S-TWIN objective.

Six segments are supported by published Spectra Ultra methods and six
SpectrumStream signals in a user EMD, not an explicit segment-count statement
in the cited manufacturer datasheet. That instrument's elevation metadata
is approximately 32.055–32.065 degrees; use 32.06 degrees only as its reference
drawing centreline.

No attributable production values were found for each segment's area/shape,
sample distance, collimator/cold shield/support ring/package dimensions, or
absolute azimuth phase. Do not copy Super-X's 30 mm² or patent examples'
50–100 mm² ranges. Six symmetric segments imply 60-degree spacing in the
schematic; its zero-degree phase is a replaceable non-OEM convention.

Angular scale checks:

- 4.45 sr is 35.4% of 4*pi; 4.04 sr is 32.1%.
- Equal division among six gives at least 0.7417 or 0.6733 sr per segment.
- Equal-solid-angle cones have half-angles about 28.1 or 26.8 degrees.

Equivalent cones summarise acceptance; they do not determine area or distance.
Physical Layout shows two opposite azimuth projections, centre directions and
equivalent acceptance boundaries. Head size/distance are display-only, not TOML
physical dimensions or electron-optical inputs.

An earlier schematic placed solid heads inside projected Objective material.
Positioning active-face/housing vertices beyond the maximum upper/lower pole
radius plus 1 mm removes that 2-D solid overlap only. The 1 mm is not a product
installation dimension. Angular lines may still cross a pole projection because
they are not material. Without sections, this is not proof of 3-D collimator
clearance or absence of holder/pole shadowing; do not change the objective gap
merely to fit unknown hardware.

Sources:

- [Spectra Ultra datasheet](https://documents.thermofisher.com/TFS-Assets/MSD/Datasheets/spectra-ultra-ds0361-materials-science-datasheet.pdf)
- [Iliad Ultra datasheet](https://documents.thermofisher.com/TFS-Assets/MSD/Datasheets/iliad-ultra-datasheet-ds0510.pdf)
- [Published six-segment Spectra Ultra methods](https://doi.org/10.1021/acsami.4c18243)
- [US8410439B2: detectors inside the lens and pole-supported mounting](https://patents.google.com/patent/US8410439B2/en)

## 4. Post-P2 detector spacing and viewing chamber

Public Titan diagrams place HAADF first in the post-P2 viewing/recording section,
followed by the main screen, DF/BF stack and Camera. Fischione documents retractable
STEM detector mechanisms with vacuum bellows. None of these establishes a
production distance from the P2 housing end to the active detector surface.
HAADF immediately downstream is topologically plausible; 7.25 mm is not OEM evidence.

In the recorded two recording TOMLs, P2 ends at local Z=772.5 mm:

| Active surface | Distance below P2 housing |
| --- | --- |
| HAADF | 7.25 mm |
| Fluorescent Screen | 127.25 mm |
| DF | 217.25 mm |
| BF | 287.25 mm |
| Camera | 399.75 mm |

Only HAADF is very close; the rest are 127–400 mm downstream. The schematic
post_projector_detector_chamber extends from P2 to local Z=1100 mm and contains
HAADF, screen, DF and BF. Its 180/200 mm ID/OD and endpoint are replaceable
non-OEM envelopes; Camera remains beyond it.

The chamber is mechanical_only and axial_vacuum_context_only. It introduces
no propagation plane/cutoff, moves no active surface and triggers no preset solve.

Sources:

- [Antwerp Titan³ column, Figure 1.4](https://repository.uantwerpen.be/docman/irua/6e3afa/130499.pdf)
- [Oregon Titan topology](https://scholarsbank.uoregon.edu//bitstreams/d56b2b6e-e2d1-4fbe-abe0-eafe58bfee67/download)
- [Thermo Fisher MiCo detector layout](https://documents.thermofisher.com/TFS-Assets/MSD/manuals/mico-user-guide-1-16.pdf)
- [Fischione Model 3000 retractable HAADF](https://sfilev2.f-static.com/image/users/390742/ftp/my_files/PB3000.pdf?id=26921072)

## 5. Historical implementation boundary

At the original mechanical stage, one eds_detector_system="ultra_x" used
configs/detectors/eds/UltraX.toml. Five columns referenced its zero-thickness
sample-plane installation. Super-X was comparison research, not a second instance
or selector. The detector was a transverse Objective child sharing the sample
anchor, not an aperture, axial detector or propagation stop.

That stage implemented structure, provenance and angular acceptance only:
generation, absorption, holder shadowing, efficiency, pulse pile-up, energy
response and quantitative spectra were still pending. Later generic EDS.toml
and signal work supersede the implementation status, not the evidence limits.

Future drawings should include a scale/known dimension and identify detector
orientation relative to holder axis, upper/lower poles, aperture port and
cooling/vacuum interfaces. No preset strengths were recalculated for this work.

## 6. Take-off line, Z displacement and Objective constraints

Retain the instrument-specific 32.06-degree centreline and documented
4.45/4.04 sr acceptances.

1. Angle and total solid angle alone cannot determine millimetre positions.
   Assuming equal sample-facing disks of radius a gives a 28.1203-degree
   half-angle and d=a/tan(28.1203 degrees). Actual area, shape and distance remain
   unknown, so no inferred d is written into TOML.
2. The displayed 4.5 mm active-face line is only a symbol. Treating it as a real
   disk would imply d=4.2103 mm, axial offset=2.2349 mm and radial position=3.5682 mm,
   inside the pole/stage region. At the approximately 59.58 mm drawing distance,
   the same cone would require about 63.68 mm disk diameter and 3185 mm² area.
   Both calculations demonstrate why schematic size is not sensor size.
3. Moving along a fixed 32.06-degree line does not change its pole intersection.
   Moving only Z at fixed radius changes the angle and, for a fixed sensor, solid
   angle. There is no independent Z-only solution preserving all other quantities.
4. With the original 5.4 mm gap, 96 mm OD, 10 mm flat-tip OD and 23.56 mm nose,
   the centreline reaches radius 4.3109 mm at the pole face, below the 5 mm tip
   radius, and penetrates approximately 6.0730 mm radially at the shoulder.
5. The subsequent non-OEM constrained design retained gap=5.4 mm, OD=96 mm and
   bore=5.76 mm, with flat-tip OD=8.0 mm and nose length=27.5584 mm.
   A 57.94-degree cone angle relative to the axis is parallel to the reference
   take-off line, giving about 0.3109 mm constant radial centreline clearance.
   Zero-clearance limits: tip OD <=8.6217 mm; nose length >=27.3637 mm.
6. This proves 2-D centreline clearance only, not unobstructed 4.45 sr.
   Six axes at 60-degree azimuth spacing and 32.06-degree elevation have minimum
   spherical separation 50.1427 degrees; equivalent cone diameter is
   56.2406 degrees, so the cones overlap. They are aggregate summaries, not six
   separate circular collimator holes. Full validation requires sensor,
   collimator, upper/lower arrangement and 3-D pole window/cutout geometry.

Only mechanical profiles/display diagnostics changed; Objective/preset strengths
were not recalculated.

Additional sources:

- [Spectra Ultra: S-TWIN and 4.45/4.04 sr](https://documents.thermofisher.com/TFS-Assets/MSD/Datasheets/spectra-ultra-ds0361-materials-science-datasheet.pdf)
- [Iliad Ultra specifications](https://documents.thermofisher.com/TFS-Assets/MSD/Datasheets/iliad-ultra-datasheet-ds0510.pdf)
- [Zaluzec: measured XPAD acceptance and holder penumbra](https://www.osti.gov/servlets/purl/1894230)
- [Early XPAD results: custom ZTwin poles](https://er-c.org/wp-content/uploads/2021/05/pico2021-programme.pdf)

## 7. Projection-chamber differential-pumping aperture

FEI/Thermo Fisher manuals call the small vacuum restriction between column and
projection chamber the differential pumping aperture (DPA). It sits at the top
of the chamber, after D/I/P1/P2 and before HAADF/viewing/recording detectors.
The public Tecnai/Talos reference bore is 200 um, not a confirmed Titan/Iliad
production dimension. Material and axial thickness were not reliably documented.

This is distinct from the farther-downstream spectrometer entrance aperture:
the former partitions vacuum; the latter defines spectrometer acceptance.
Conjugacy depends on projector mode. TEM/EFTEM image conditions can place a
diffraction/crossover plane at DPA; TEM diffraction or STEM-EELS image coupling
can place an image plane there, with diffraction at spectrometer entrance.
The fixed component must not be permanently named a diffraction plane.

Recorded implementation:

- Name: Projection-Chamber Differential-Pumping Aperture.
- Key: projection_chamber_dpa_aperture.
- Both recording TOMLs: local Z=772.5 mm, at P2 housing end/chamber start;
  HAADF active surface is 7.25 mm downstream.
- 0.2 mm series-reference bore, 20 mm surrounding-channel/schematic plate OD.
  Unknown thickness is a zero-length boundary with separate display thickness.
- Fixed/non-retractable; TOML diameter remains editable for future design.
- Initial mechanical_only stage has no optical_reference_local_z_mm, APERTURE_KEYS
  entry, ray clipping, alignment constraint or preset solve.

Implementation update (2026-09-06): the user subsequently requested fixed stops
in Optical calculation components. DPA now has a coincident optical stop plane,
ray/coherent-wave clipping, adjustable opening and TOML axial placement, and
permanent insertion. The reference dimensions/evidence above are unchanged;
no conjugate plane or automatic preset solve is imposed.

Sources:

- [Talos general information: 200 um projection-chamber DPA](https://imf.ucmerced.edu/sites/g/files/ufvvjh1081/f/documents/tem_talosbasicguide_1.pdf)
- [Tecnai column description](https://www.dartmouth.edu/emlab/docs/fei_tecnai_f20_column_description_doc.pdf)
- [Tecnai modes: DPA and spectrometer entrance conjugacy](https://www.dartmouth.edu/emlab/docs/fei_tecnai_f20_modes_doc.pdf)
- [Titan TIA help: STEM-EELS image coupling](https://www.manuallib.com/download/2023-10-20/Titan%20on-line%20help%20manual%20--%20TIA.pdf)
