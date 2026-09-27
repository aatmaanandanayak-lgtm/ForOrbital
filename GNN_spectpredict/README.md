# Chlorin Structure Elucidation from UV/Vis Spectra

Physics-informed pipeline for predicting structural modifications to the
chlorin macrocycle (substituent identity and position) from UV/Vis
absorption spectra, built around an analytic inversion of Gouterman's
four-orbital model rather than a black-box structure-to-spectrum map.

Core idea: invert a measured spectrum's Q_x, Q_y, B_x, B_y band positions
and intensities into the four Gouterman frontier orbital energies via
closed-form 2x2 configuration-interaction algebra, then use a
sparse-recovery step (orbital-shift vector -> substituent identity and
position, via a |c(r)|^2 coefficient dictionary) to infer candidate
structures. A trained GNN can later serve as a fast surrogate for the
orbital-energy step, pretrained on cheap semiempirical data and fine-tuned
on real spectra via this same inversion pipeline.

## Reference set

Chlorophyll a, b, d, f. All four in the SAME solvent (diethyl ether),
same source paper (Kobayashi 2013), same measurement campaign - chosen
specifically to eliminate both solvent and inter-laboratory variability
as confounds. Source: Taniguchi & Lindsey 2021, Photochem. Photobiol. 97,
136-165 (php.13319), PhotochemCAD database, "Natural_Chlorophylls.zip".

Pheophytin a was considered and DROPPED from the active set: the only
readily available digitized spectrum was in acetone, not diethyl ether,
which would confound the demetalation signal with an uncontrolled solvent
shift. Its structure remains verified (see
data/raw/structures/reference_structures.csv) for later use if a
matched-solvent measurement is found.

Scope: the project targets chemical modifications to the chlorin
macrocycle ring (substituent identity/position) - NOT metallation state,
which was only ever a secondary/scaffold-level consideration.

## Structure verification (Phase 1)

All five reference structures (a, b, d, f, pheophytin a) were verified
with RDKit: parsed SMILES/InChI, computed molecular formula and formal
charge, checked against literature values. This caught two real errors
before they could propagate silently downstream:
- Chlorophyll f: the Wikidata-mirrored InChI (via PubChem CID 152743444)
  gives a +1 formal charge, not neutral - a real RDKit mobile-H mismatch.
  Fixed by using ChEBI's own CHEBI:61290 entry (dative-bond SMILES) instead.
- Pheophytin a: ChEBI's plain SMILES field is missing both inner NH
  protons (2H short). Fixed by using ChEBI's InChI instead, which carries
  the correct mobile-H layer.

See `scripts/phase1_reference_data/verify_reference_structures.py`.

## Phase 2: spectral deconvolution

### Q_y - solid for all four compounds
Two-mode Franck-Condon (Huang-Rhys) vibronic fit, validated over the
single-mode version by BIC in every case (delta-BIC -62 to -89 across
a/b/d/f - a general feature of the series, not a Chl a quirk). The
primary vibronic mode clusters tightly for a/d/f (1111-1156 cm^-1) -
a real, reproducible, shared macrocycle mode.

Chl b's Q_y fit is an outlier (761 cm^-1 primary mode vs the a/d/f
cluster) - traced to Q_x and the Q_y vibronic satellite overlapping
for this compound specifically (visible pre-fit in the raw band shape:
no distinct third peak, unlike a/d/f's clean three-feature pattern).

### Q_x - validated for a/d/f, honestly unresolved for Chl b
Key lesson: a bare Gaussian fails badly even for a visually "isolated"
Q_x band (Chl a alone: R^2=0.33) because Q_x sits in a valley between
the B-band's red tail and the Q_y-manifold's blue tail, which are not
negligible there even when the raw plot looks flat. A baseline term
(Gaussian + linear background) fixes this - but the window must also
stay clear of the Q_y satellite's own tail (found via residual
inspection, not assumption).

Validated results (Gaussian + linear baseline):
| Compound | Position | Width (cm^-1) | R^2 |
|---|---|---|---|
| Chl a | 575.4 nm | 222 +/- 18 | 0.981 |
| Chl d | 592.3 nm | 234.2 +/- 11.0 | 0.984 |
| Chl f | 600.3 nm | 280.7 +/- 26.0 | 0.984 |

Chl b: multiple joint Q_x+Q_y-satellite fits attempted (unconstrained,
then with vibronic modes fixed from the a/d/f consensus) - both revealed
severe parameter degeneracy (S1 collapsing to ~0, correlations up to
-0.95) rather than resolving it. Conclusion: Chl b's 594.7 nm feature is
a genuine, inseparable Q_x + Q_y-satellite BLEND (position 594.73 nm,
width 149.3 +/- 5.2 cm^-1 as a single effective quantity) - not a clean
Q_x value comparable to a/d/f. Treated as flagged/provisional, same
spirit as pheophytin a being set aside.

### B-band - component count confirmed, decomposition underdetermined
Second-derivative analysis resolves 3 components for Chl a (431, 405,
373 nm) with UNEQUAL spacing (1515, 2083 cm^-1) - rules out a single
vibronic progression. A proper 3-Gaussian+baseline fit confirms 3
components decisively beat 2 (delta-BIC=-126.8). BUT widening the width
bounds to stop them pegging reveals severe correlations (~10 parameter
pairs above 0.85) - the component COUNT is real, but individual
position/width/amplitude VALUES are not independently determined by a
single linear absorption spectrum.

Practical resolution: use model-free aggregate quantities (intensity-
weighted centroid, integrated intensity) for the B-band, not the
individual sub-component fits. The B_x/B_y electronic split is deferred
to a computed orbital picture (see Phase 3).

Aggregate B-band values (crude linear-baseline-between-window-edges
method - ballpark, not precision):
| Compound | Centroid | Integrated intensity | Peak abs |
|---|---|---|---|
| Chl a | 408.5 nm | 1982 | 1.000 |
| Chl b | 449.9 nm | 708  | 1.000 |
| Chl d | 417.3 nm | 1993 | 0.862 |
| Chl f | 410.0 nm | 2140 | 0.664 |

## Phase 3: Gouterman inversion - first real run

The full per-polarization inversion (getting A_x, B_x, W_x separately
from A_y, B_y, W_y) needs E(B_x) and E(B_y) as two distinct numbers.
Phase 2 could only deliver an aggregate B value - so this is currently
BLOCKED. This is a genuine project-structure finding: Phase 3's full
completion depends on some output from Phase 4 (a computed orbital
picture that can split B into x/y), which wasn't visible in the original
toy-case design.

Partial progress made using only the reliable half of the data (Q_x,
Q_y - no intensities, no B-split needed): the redundancy relation
B_x - B_y = Q_y - Q_x gives a direct, falsifiable PREDICTION.

Results (Monte Carlo uncertainty propagation from real Phase 2 fit
uncertainties):
- Chl a: predicted |B_x - B_y| = 2240.7 +/- 11.8 cm^-1
- Chl d: predicted |B_x - B_y| = 2305.5 +/- 8.5 cm^-1
- Chl f: predicted |B_x - B_y| = 2267.0 +/- 16.5 cm^-1

Striking: consistent within 3% across three chemically different
compounds, from fully independent fits - real evidence the four-orbital
model is capturing genuine physics, not curve-fitting noise.

Cross-check against Chl a's ambiguous B-band sub-components: predicted
2240.7 cm^-1 vs the empirical E3-E2 gap of 2102.7 cm^-1 (6.2% off) -
encouraging given how uncertain those individual B-band positions are.

### Current test in progress
A single targeted ORCA ground-state DFT calculation (B3LYP/def2-SVP,
RIJCOSX+D3BJ) on Chl a (phytyl truncated to methyl ester - spectroscopically
silent per project scope, 29% fewer heavy atoms) to check whether computed
HOMO-1/HOMO/LUMO/LUMO+1 splittings predict a Bx-By gap consistent with
~2240 cm^-1. See `orca_inputs/phase3_test/` and
`scripts/phase3_inversion/parse_orca_frontier_orbitals.py`.

## Environment notes (NEMO-specific, useful if resuming there)
- Physics/cheminformatics work: clone of `qchem` conda env (RDKit, xtb) +
  scipy, lmfit added.
- ORCA: `module load ORCA/5.0.4-gompi-2022a`, requires
  `module purge; conda deactivate` first (ORCA's bundled MPI conflicts
  with conda's libraries otherwise).
- Home quota is small (5GB) - check `df -h ~` before bulk installs.

## Directory structure
```
data/raw/              - reference structures (CSV) and spectra (tab-separated .abs.txt)
scripts/                - organized by phase/subtask, see scripts/INDEX.md
orca_inputs/phase3_test/ - the Chl a frontier-orbital DFT calculation
results/phase2_figures/  - diagnostic and validated plots from Phase 2
models/, notebooks/, src/, slurm/, logs/, envs/ - reserved for later phases
```
