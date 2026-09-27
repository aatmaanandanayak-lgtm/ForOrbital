"""
Phase 2: resolve overlapping/shoulder features in the B-band region using
second-derivative analysis - much more sensitive than raw peak-finding
for unresolved components. See README "Phase 2 progress" notes: for
Chl a this revealed 3 components with UNEQUAL spacing, ruling out a
simple single-mode vibronic progression and leaving the Bx/By electronic
assignment genuinely open pending Phase 4's computed orbital picture.

Edit SPECTRUM_FILE / COMPOUND_LABEL / WL_RANGE below for other compounds.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

SPECTRUM_FILE = "../data/raw/spectra/chl_a_et2o_kobayashi2013.abs.txt"
COMPOUND_LABEL = "Chl a"
WL_RANGE = (350, 470)  # adjust per compound's B-band location
OUTPUT_PNG = "bband_2ndderiv.png"

df = pd.read_csv(SPECTRUM_FILE, sep="\t", encoding="utf-8")
df.columns = ["wl_nm", "abs"]
df = df.sort_values("wl_nm").reset_index(drop=True)

mask = (df.wl_nm >= WL_RANGE[0]) & (df.wl_nm <= WL_RANGE[1])
sub = df[mask].reset_index(drop=True)
sub["wn"] = 1e7 / sub.wl_nm
sub = sub.sort_values("wn").reset_index(drop=True)

d2 = savgol_filter(sub["abs"].values, window_length=9, polyorder=3, deriv=2)

peaks_idx, _ = find_peaks(sub["abs"].values, prominence=0.005)
print("Raw-trace local maxima (prominence > 0.005):")
for i in peaks_idx:
    print(f"  {sub.wl_nm[i]:7.2f} nm  ({sub.wn[i]:8.1f} cm^-1)   abs={sub['abs'][i]:.4f}")

neg_d2_peaks, _ = find_peaks(-d2, prominence=0.0005)
print("\nSecond-derivative minima (component centers, more sensitive):")
for i in neg_d2_peaks:
    print(f"  {sub.wl_nm[i]:7.2f} nm  ({sub.wn[i]:8.1f} cm^-1)   abs={sub['abs'][i]:.4f}")

if len(neg_d2_peaks) >= 2:
    wn_vals = sub.wn[neg_d2_peaks].values
    print("\nSpacings between consecutive resolved components:")
    for i in range(len(wn_vals) - 1):
        print(f"  {abs(wn_vals[i+1]-wn_vals[i]):.1f} cm^-1")
    print("(equal spacings => consistent with single-mode vibronic progression;")
    print(" unequal spacings => rules that out, see script docstring)")

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
axes[0].plot(sub.wl_nm, sub["abs"], 'k-', lw=1.3)
axes[0].plot(sub.wl_nm[peaks_idx], sub["abs"][peaks_idx], 'ro', ms=5, label='raw local max')
axes[0].plot(sub.wl_nm[neg_d2_peaks], sub["abs"][neg_d2_peaks], 'b^', ms=7, label='2nd-deriv component')
for i in neg_d2_peaks:
    axes[0].annotate(f"{sub.wl_nm[i]:.0f}", (sub.wl_nm[i], sub["abs"][i]),
                      textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='b')
axes[0].set_ylabel("Absorbance")
axes[0].legend()
axes[0].set_title(f"{COMPOUND_LABEL} B-band region: raw trace")
axes[1].plot(sub.wl_nm, d2, 'g-', lw=1.2)
axes[1].axhline(0, color='gray', lw=0.5)
axes[1].plot(sub.wl_nm[neg_d2_peaks], d2[neg_d2_peaks], 'b^', ms=7)
axes[1].set_xlabel("Wavelength (nm)")
axes[1].set_ylabel("2nd derivative")
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=130)
