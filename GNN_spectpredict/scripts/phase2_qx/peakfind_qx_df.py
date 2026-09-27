import pandas as pd
from scipy.signal import find_peaks

files = {
    "Chl d": "Natural Chlorophylls/CHL023_Chl d, Et2O (Kobayashi, 2013).abs.txt",
    "Chl f": "Natural Chlorophylls/CHL031_Ch f, Et2O (Kobayashi, 2013).abs.txt",
}
for name, path in files.items():
    df = pd.read_csv(path, sep="\t", encoding="utf-8")
    df.columns = ["wl_nm", "abs"]
    df = df.sort_values("wl_nm").reset_index(drop=True)
    sub = df[(df.wl_nm >= 560) & (df.wl_nm <= 630)].reset_index(drop=True)
    peaks_idx, _ = find_peaks(sub["abs"].values, prominence=0.005)
    print(f"=== {name}: local features, 560-630nm ===")
    for i in peaks_idx:
        print(f"  {sub.wl_nm[i]:7.2f} nm   abs={sub['abs'][i]:.4f}")
    print()
