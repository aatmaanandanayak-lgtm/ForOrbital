"""
Phase 2: two-mode Franck-Condon fit to Q_y - extends fit_qy_single_mode.py
with a second, independent vibrational mode (product of two Poisson
distributions, standard multi-mode FC treatment). For Chl a this gave a
real, BIC-confirmed improvement (R^2 0.993 vs 0.976 single-mode) - see
README "Phase 2 progress" notes. PREFER THIS OVER THE SINGLE-MODE VERSION.

Edit SPECTRUM_FILE / COMPOUND_LABEL / WL_RANGE / seed params for other
compounds - seed hw1 from the single-mode fit's result, hw2 can start
around 400-600 cm^-1 as a first guess.
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
WL_RANGE = (590, 710)
SEED_E00_NM = 661
SEED_HW1 = 1080     # from single-mode fit
SEED_HW2 = 400      # lower-frequency second mode, first guess
OUTPUT_PNG = "qy_two_mode_fit.png"

df = pd.read_csv(SPECTRUM_FILE, sep="\t", encoding="utf-8")
df.columns = ["wl_nm", "abs"]
df = df.sort_values("wl_nm").reset_index(drop=True)
mask = (df.wl_nm >= WL_RANGE[0]) & (df.wl_nm <= WL_RANGE[1])
sub = df[mask].copy()
sub["wn"] = 1e7 / sub.wl_nm
sub = sub.sort_values("wn").reset_index(drop=True)

def fc_2mode(wn, E00, hw1, S1, hw2, S2, sigma, A, n1max=3, n2max=2):
    total = np.zeros_like(wn)
    for n1 in range(n1max):
        w1 = np.exp(-S1) * S1**n1 / factorial(n1)
        for n2 in range(n2max):
            w2 = np.exp(-S2) * S2**n2 / factorial(n2)
            center = E00 + n1*hw1 + n2*hw2
            total += A * w1 * w2 * np.exp(-0.5*((wn-center)/sigma)**2)
    return total

model = Model(fc_2mode, independent_vars=["wn"],
              param_names=["E00", "hw1", "S1", "hw2", "S2", "sigma", "A"])
params = model.make_params(
    E00=dict(value=1e7/SEED_E00_NM, min=14900, max=15300),
    hw1=dict(value=SEED_HW1, min=900, max=1300),
    S1=dict(value=0.21, min=0.02, max=1.5),
    hw2=dict(value=SEED_HW2, min=100, max=800),
    S2=dict(value=0.15, min=0.0, max=1.0),
    sigma=dict(value=150, min=50, max=400),
    A=dict(value=sub["abs"].max()*1.3, min=0),
)
result = model.fit(sub["abs"].values, params, wn=sub["wn"].values)
print(result.fit_report())
print(f"\nRecovered 0-0 origin: {result.params['E00'].value:.1f} cm^-1 "
      f"= {1e7/result.params['E00'].value:.2f} nm")
print("\nCompare AIC/BIC against fit_qy_single_mode.py's output before")
print("trusting the extra complexity - it should win on BOTH, not just R^2.")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sub.wl_nm, sub["abs"], 'ko', ms=3, label="data")
wn_fine = np.linspace(sub.wn.min(), sub.wn.max(), 600)
ax.plot(1e7/wn_fine, model.eval(result.params, wn=wn_fine), 'b--', lw=1.5, label="two-mode fit")
ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Absorbance"); ax.legend()
ax.set_title(f"{COMPOUND_LABEL} Q_y: two-mode Franck-Condon fit")
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=130)
