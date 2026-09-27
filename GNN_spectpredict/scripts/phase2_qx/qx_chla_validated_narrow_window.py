import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lmfit import Model

df = pd.read_csv("Natural Chlorophylls/CHL007_Chl a, Et2O (Kobayashi, 2013).abs.TXT",
                  sep="\t", encoding="utf-8")
df.columns = ["wl_nm", "abs"]
df = df.sort_values("wl_nm").reset_index(drop=True)

def gauss_plus_linear(wn, Ex, sigma_x, Ax, slope, intercept):
    return Ax * np.exp(-0.5*((wn-Ex)/sigma_x)**2) + slope*wn + intercept

model = Model(gauss_plus_linear, independent_vars=["wn"])

windows = {"original (555-600)": (555, 600), "narrowed (555-593)": (555, 593)}
results = {}

for label, (lo, hi) in windows.items():
    mask = (df.wl_nm >= lo) & (df.wl_nm <= hi)
    sub = df[mask].copy()
    sub["wn"] = 1e7/sub.wl_nm
    sub = sub.sort_values("wn").reset_index(drop=True)
    params = model.make_params(Ex=1e7/577, sigma_x=dict(value=300,min=50,max=800),
                                Ax=dict(value=0.03,min=0), slope=0.0, intercept=0.0)
    result = model.fit(sub["abs"].values, params, wn=sub["wn"].values)
    results[label] = (result, sub)
    print(f"=== {label} ===")
    print(f"  R^2={result.rsquared:.4f}  sigma_x={result.params['sigma_x'].value:.1f}"
          f"+/-{result.params['sigma_x'].stderr or float('nan'):.1f} cm^-1"
          f"  ({'AT BOUND' if result.params['sigma_x'].value>=790 else 'interior'})")
    print(f"  Ex={1e7/result.params['Ex'].value:.2f} nm   Ax={result.params['Ax'].value:.4f}"
          f"+/-{result.params['Ax'].stderr or float('nan'):.4f}")
    print()

fig, ax = plt.subplots(figsize=(8,5))
for label, (result, sub) in results.items():
    ax.plot(sub.wl_nm, sub["abs"], 'o', ms=4, label=f"{label} data")
    wn_fine = np.linspace(sub.wn.min(), sub.wn.max(), 300)
    ax.plot(1e7/wn_fine, model.eval(result.params, wn=wn_fine), '-', lw=1.5,
            label=f"{label} fit (R²={result.rsquared:.3f})")
ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Absorbance"); ax.legend(fontsize=8)
ax.set_title("Chl a Q_x: effect of excluding the Q_y-satellite-contaminated red edge")
plt.tight_layout()
plt.savefig("/tmp/chla_qx_narrowed.png", dpi=130)
