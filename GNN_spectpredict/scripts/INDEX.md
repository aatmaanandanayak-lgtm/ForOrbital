# Scripts index

Status tags: **VALIDATED** (trust the numbers) - **SUPERSEDED** (works, but a
better version exists) - **DIAGNOSTIC** (deliberately kept - documents a
real failure and what it revealed, not clutter)

## phase1_reference_data/
- `verify_reference_structures.py` - **VALIDATED**. RDKit formula/charge
  check for all 5 reference compounds. Caught 2 real errors (Chl f InChI,
  pheophytin a SMILES) - see comments in-file.

## phase2_qy/
- `explore_raw_spectrum.py` - peak-finding before fitting, Chl a.
- `fit_qy_single_mode.py` - **SUPERSEDED** by two-mode (R^2=0.976 vs 0.993).
- `fit_qy_two_mode.py` - **VALIDATED**. BIC-justified two-mode Franck-Condon
  fit, the standard going forward.
- `peakfind_qregion_bdf.py` - peak-finding for Chl b/d/f.
- `fit_qy_bdf_comparison.py` - **VALIDATED**. Confirms two-mode structure
  general across a/b/d/f. Flags Chl b as an outlier (contaminated fit).

## phase2_qx/
- `qx_isolated_chla_v1_no_baseline.py` - **DIAGNOSTIC**. Bare Gaussian on
  Chl a's Q_x: R^2=0.33. Shows a baseline term is not optional.
- `qx_isolated_chla_v2_with_baseline.py` - added baseline: R^2=0.79, still
  window-contaminated by the Q_y satellite tail.
- `qx_chla_residual_diagnosis.py` - **DIAGNOSTIC**. Found the exact cause
  (red-edge contamination) via residual inspection.
- `qx_chla_validated_narrow_window.py` - **VALIDATED**. R^2=0.981.
- `qx_qy_joint_chlb_v1_unconstrained.py` - **DIAGNOSTIC**. Looked good
  (R^2=0.998) but S1 collapsed to ~0, Ax-S1 correlation -0.91 - not
  trustworthy despite the R^2.
- `qx_qy_joint_chlb_v2_modes_fixed.py` - **DIAGNOSTIC**. Attempted fix
  made the degeneracy worse, not better. Rules out "uncertain mode
  frequency" as the cause.
- `qx_chlb_window_test.py` - final honest read: Chl b's weak feature is a
  genuine Qx+Qy-satellite BLEND, not separable from this data.
- `peakfind_qx_df.py` - peak-finding for Chl d/f.
- `qx_fit_chld_chlf.py` - **VALIDATED**. Both clean, R^2=0.984 each.

## phase2_bband/
- `explore_bband_2ndderiv.py` - resolved 3 components (not the assumed 2).
- `bband_chla_modelselect.py` - 3-component model decisively beats
  2-component (delta-BIC=-126.8).
- `bband_chla_correlation_check.py` - **DIAGNOSTIC**. Widened bounds,
  revealed severe parameter correlations (~10 pairs >0.85) - component
  COUNT is real, individual values are not independently determined.
- `bband_aggregate_all_compounds.py` - **VALIDATED** (as an aggregate
  quantity only - not a per-component breakdown). The actual B-band
  numbers to use until Phase 4 informs the x/y split.

## phase3_inversion/
- `phase3_gather_inputs.py` - precise Qy/Qx values + uncertainties, a/d/f.
  Output saved as `phase3_inputs.json`.
- `phase3_inversion.py` - the core result: predicted Bx-By splitting from
  the redundancy relation, consistent within 3% across 3 compounds.
- `phase3_truncate_chla.py` - phytyl -> methyl ester truncation for DFT.
- `phase3_build_chla_geometry.py` - 3D geometry, Mg placed at ring-N
  centroid.
- `parse_orca_frontier_orbitals.py` - awaiting the running ORCA job
  (see orca_inputs/phase3_test/).
