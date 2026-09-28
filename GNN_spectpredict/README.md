# Chlorin Structure Elucidation from UV/Vis Spectra

Attempt at a physics-informed pipeline for predicting structural modifications to
chlorin macrocycles from its UV/Vis absorption spectra (assuming user-provided UV/Vis spectra
of known chlorin compound - for e.g., chl a - measured with same instrument) built around analytically 
inverting Gouterman's four-orbital model.

The idea is to invert a measured spectrum's Q_x, Q_y, B_x, B_y band positions and intensities into the four frontier orbital energies via
closed-form 2x2 configuration-interaction algebra and then (by using the orbital-shift vector to obtain substituent identity and
position, via a |c(r)|^2 coefficient dictionary) inferring possible candidate structures. (Ideally) a trained GNN can later replace
the orbital-energy step, by pretraining on semiempirical data and fine-tuning on real spectra (same inversion pipeline).

## Reference set

Chlorophyll a, b, d, f. All four in the SAME solvent (diethyl ether),
same source paper (Kobayashi 2013), same measurement campaign.
Source: Taniguchi & Lindsey 2021, Photochem. Photobiol. 97,
136-165 (php.13319), PhotochemCAD database, "Natural_Chlorophylls.zip".

Disclaimer: Initially considered Pheophytin a but then dropped it because its only
readily available digitized spectrum was in acetone not diethyl ether (the solvent-shift would 
have added a confounding factor on top of the demetallation). However, if anyone has a spectrum of phe a
in diethyl ether and would be happy to share it please do email me!

## Structure verification (Phase 1)

Ref structures (a, b, d, f, phe a) verified with RDKit: parsed SMILES/InChI, computed molecular formula and formal
charge, checked against literature values. 

This caught two errors:
- Chl f: the Wikidata-mirrored InChI (via PubChem CID 152743444)
  gives a +1 formal charge, not neutral.
  Fixed using ChEBI's own CHEBI:61290 entry (dative-bond SMILES) instead.
- Phe a: ChEBI's plain SMILES field is missing both inner NH
  protons (2H short). Fixed using ChEBI's InChI instead.

See `scripts/phase1_reference_data/verify_reference_structures.py`.

## Phase 2: spectral deconvolution

### Q_y - solid for all four compounds
Two-mode Franck-Condon (Huang-Rhys) vibronic fit, validated over the single-mode version by BIC in every case (delta-BIC -62 to -89 across
a/b/d/f - a general feature of the series, not a Chl a quirk). The primary vibronic mode clusters tightly for a/d/f (1111-1156 cm^-1).

Chl b's Q_y fit is an outlier (761 cm^-1 primary mode vs the a/d/f cluster) - traced to Q_x and the Q_y vibronic satellite overlapping
for this compound specifically (visible pre-fit in the raw band shape: no distinct third peak, unlike a/d/f's three-feature pattern).

### Q_x - validated for a/d/f, honestly unresolved for Chl b
Takeaway was that a bare Gaussian fails badly even for a visually "isolated" Q_x band (Chl a alone: R^2=0.33) because Q_x is in a minimum 
between B-band's red tail and Q_y-manifold's blue tail: not negligible there even when the raw plot looks flat. A baseline term
(Gaussian + linear background) fixes this - but the window also needs to stay clear of the Q_y satellite's own tail (found by residual
inspection).

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
