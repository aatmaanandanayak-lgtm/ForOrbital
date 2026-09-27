"""
Phase 2, step 1: load a raw absorption spectrum and locate approximate
band positions before attempting any fit. "Look before you fit."

Edit SPECTRUM_FILE / COMPOUND_LABEL below to run on a different compound.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

SPECTRUM_FILE = "../data/raw/spectra/chl_a_et2o_kobayashi2013.abs.txt"
COMPOUND_LABEL = "Chlorophyll a, Et2O (Kobayashi 2013)"
OUTPUT_PNG = "chla_raw_peakfind.png"
PROMINENCE = 0.02  # lower this (e.g. 0.005) to catch subtle shoulders

df = pd.read_csv(SPECTRUM_FILE, sep="\t", encoding="utf-8")
df.columns = ["wl_nm", "abs"]
df = df.sort_values("wl_nm").reset_index(drop=True)

print(f"Points: {len(df)}, range {df.wl_nm.min():.1f}-{df.wl_nm.max():.1f} nm")
print(f"Max absorbance: {df['abs'].max():.4f} at {df.loc[df['abs'].idxmax(),'wl_nm']:.1f} nm")

peaks_idx, props = find_peaks(df["abs"].values, prominence=PROMINENCE)
print(f"\nPeaks found (prominence > {PROMINENCE}):")
for i in peaks_idx:
    print(f"  {df.wl_nm[i]:7.2f} nm   abs={df['abs'][i]:.4f}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(df.wl_nm, df["abs"], 'k-', lw=1.2)
ax.plot(df.wl_nm[peaks_idx], df["abs"][peaks_idx], 'ro', ms=5)
for i in peaks_idx:
    ax.annotate(f"{df.wl_nm[i]:.0f}", (df.wl_nm[i], df["abs"][i]),
                textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Absorbance")
ax.set_title(f"{COMPOUND_LABEL} - raw data + peak-find")
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=130)
print(f"\nSaved {OUTPUT_PNG}")
