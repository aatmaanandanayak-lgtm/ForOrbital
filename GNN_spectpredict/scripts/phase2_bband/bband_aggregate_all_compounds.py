import pandas as pd
import numpy as np
from scipy.integrate import trapezoid

files = {
    "Chl a": "Natural Chlorophylls/CHL007_Chl a, Et2O (Kobayashi, 2013).abs.TXT",
    "Chl b": "Natural Chlorophylls/CHL015_Chl b, Et2O (Kobayashi, 2013).abs.txt",
    "Chl d": "Natural Chlorophylls/CHL023_Chl d, Et2O (Kobayashi, 2013).abs.txt",
    "Chl f": "Natural Chlorophylls/CHL031_Ch f, Et2O (Kobayashi, 2013).abs.txt",
}
windows = {"Chl a": (350,455), "Chl b": (350,470), "Chl d": (350,480), "Chl f": (350,490)}

print(f"{'Compound':<10}{'Centroid (nm)':<16}{'Integrated intensity':<22}{'Peak abs':<10}")
for name, path in files.items():
    df = pd.read_csv(path, sep="\t", encoding="utf-8")
    df.columns = ["wl_nm", "abs"]
    df = df.sort_values("wl_nm").reset_index(drop=True)
    lo, hi = windows[name]
    sub = df[(df.wl_nm>=lo)&(df.wl_nm<=hi)].copy()
    sub["wn"] = 1e7/sub.wl_nm
    sub = sub.sort_values("wn").reset_index(drop=True)
    # simple linear baseline subtraction using the window edges, then
    # intensity-weighted centroid + integrated area - model-free, robust
    baseline = np.linspace(sub["abs"].iloc[0], sub["abs"].iloc[-1], len(sub))
    corrected = np.clip(sub["abs"].values - baseline, 0, None)
    centroid_wn = trapezoid(corrected*sub.wn.values, sub.wn.values) / trapezoid(corrected, sub.wn.values)
    integrated = trapezoid(corrected, sub.wn.values)
    print(f"{name:<10}{1e7/centroid_wn:<16.2f}{integrated:<22.1f}{sub['abs'].max():<10.4f}")
