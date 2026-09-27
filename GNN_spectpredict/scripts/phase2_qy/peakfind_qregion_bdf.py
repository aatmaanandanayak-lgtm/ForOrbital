import pandas as pd
from scipy.signal import find_peaks

files = {
    "Chl b": "Natural Chlorophylls/CHL015_Chl b, Et2O (Kobayashi, 2013).abs.txt",
    "Chl d": "Natural Chlorophylls/CHL023_Chl d, Et2O (Kobayashi, 2013).abs.txt",
    "Chl f": "Natural Chlorophylls/CHL031_Ch f, Et2O (Kobayashi, 2013).abs.txt",
}

for name, path in files.items():
    df = pd.read_csv(path, sep="\t", encoding="utf-8")
    df.columns = ["wl_nm", "abs"]
    df = df.sort_values("wl_nm").reset_index(drop=True)
    # restrict to the Q-band region broadly (past the B-band tail)
    sub = df[(df.wl_nm >= 560) & (df.wl_nm <= 740)].reset_index(drop=True)
    peaks_idx, _ = find_peaks(sub["abs"].values, prominence=0.01)
    print(f"=== {name} (Q-band region, 560-740nm) ===")
    for i in peaks_idx:
        wn = 1e7/sub.wl_nm[i]
        print(f"  {sub.wl_nm[i]:7.2f} nm  ({wn:8.1f} cm^-1)   abs={sub['abs'][i]:.4f}")
    print()
