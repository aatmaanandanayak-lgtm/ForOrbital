"""
Phase 2: single-mode Franck-Condon (Huang-Rhys) vibronic fit to the Q_y
band region. Fits in wavenumber (energy) space, since vibronic spacing
is uniform in energy, not wavelength.

Edit SPECTRUM_FILE / COMPOUND_LABEL / WL_RANGE / seed params below for
other compounds. For Chl a this recovered E00 = 660.31 nm, R^2 = 0.976 -
see fit_qy_two_mode.py for the improved (R^2 = 0.993) version, which is
the one to prefer going forward.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lmfit import Model
from math import factorial

SPECTRUM_FILE = "../data/raw/spectra/chl_a_et2o_kobayashi2013.abs.txt"
COMPOUND_LABEL = "Chl a"
WL_RANGE = (590, 710)     # Q_y window - adjust per compound
SEED_E00_NM = 661         # rough 0-0 position from explore_raw_spectrum.py
SEED_HW = 1130            # rough vibronic spacing, cm^-1
OUTPUT_PNG = "qy_single_mode_fit.png"

df = pd.read_csv(SPECTRUM_FILE, sep="\t", encoding="utf-8")
df.columns = ["wl_nm", "abs"]
df = df.sort_values("wl_nm").reset_index(drop=True)
mask = (df.wl_nm >= WL_RANGE[0]) & (df.wl_nm <= WL_RANGE[1])
sub = df[mask].copy()
sub["wn"] = 1e7 / sub.wl_nm
sub = sub.sort_values("wn").reset_index(drop=True)

def franck_condon_progression(wn, E00, hw, S, sigma, A, n_terms=4):
    total = np.zeros_like(wn)
    for n in range(n_terms):
        weight = np.exp(-S) * S**n / factorial(n)
        total += A * weight * np.exp(-0.5 * ((wn - (E00 + n*hw)) / sigma) ** 2)
    return total

# NOTE: n_terms must be excluded via param_names explicitly - lmfit's
# automatic signature introspection still treats keyword-defaulted args
# as fittable Parameters even via functools.partial, which crashes on
# a non-integer n_terms. This bit us once; don't remove param_names.
model = Model(franck_condon_progression, independent_vars=["wn"],
              param_names=["E00", "hw", "S", "sigma", "A"])
params = model.make_params(
    E00=dict(value=1e7/SEED_E00_NM, min=14800, max=15400),
    hw=dict(value=SEED_HW, min=800, max=1600),
    S=dict(value=0.7, min=0.05, max=3),
    sigma=dict(value=150, min=50, max=500),
    A=dict(value=sub["abs"].max()*1.3, min=0),
)
result = model.fit(sub["abs"].values, params, wn=sub["wn"].values)
print(result.fit_report())
print(f"\nRecovered 0-0 origin: {result.params['E00'].value:.1f} cm^-1 "
      f"= {1e7/result.params['E00'].value:.2f} nm")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sub.wl_nm, sub["abs"], 'ko', ms=3, label="data")
wn_fine = np.linspace(sub.wn.min(), sub.wn.max(), 600)
ax.plot(1e7/wn_fine, model.eval(result.params, wn=wn_fine), 'r-', lw=1.5, label="fit")
ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Absorbance"); ax.legend()
ax.set_title(f"{COMPOUND_LABEL} Q_y: single-mode Franck-Condon fit")
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=130)
