import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lmfit import Model

df = pd.read_csv("Natural Chlorophylls/CHL015_Chl b, Et2O (Kobayashi, 2013).abs.txt",
                  sep="\t", encoding="utf-8")
df.columns = ["wl_nm", "abs"]
df = df.sort_values("wl_nm").reset_index(drop=True)

def gauss_plus_linear(wn, Ex, sigma_x, Ax, slope, intercept):
    return Ax * np.exp(-0.5*((wn-Ex)/sigma_x)**2) + slope*wn + intercept

model = Model(gauss_plus_linear, independent_vars=["wn"])

# Chl b's satellite/Qx overlap right AT the apparent peak (~595nm), unlike
# Chl a where they were well-separated - so try windows that stop AT or
# BEFORE the peak, not past it, testing how far red we can safely go.
windows = {
    "blue-only (560-595)": (560, 595),
    "blue-only, tighter (560-592)": (560, 592),
    "blue-only, wider (555-598)": (555, 598),
}

fig, ax = plt.subplots(figsize=(8,5))
full = df[(df.wl_nm>=555)&(df.wl_nm<=610)]
ax.plot(full.wl_nm, full["abs"], 'k.', ms=4, alpha=0.4, label="full local data")

for label, (lo, hi) in windows.items():
    mask = (df.wl_nm >= lo) & (df.wl_nm <= hi)
    sub = df[mask].copy()
    sub["wn"] = 1e7/sub.wl_nm
    sub = sub.sort_values("wn").reset_index(drop=True)
    params = model.make_params(Ex=1e7/594, sigma_x=dict(value=250,min=50,max=800),
                                Ax=dict(value=0.02,min=0), slope=0.0, intercept=0.0)
    result = model.fit(sub["abs"].values, params, wn=sub["wn"].values)
    at_bound = result.params['sigma_x'].value >= 790
    print(f"=== {label} ===")
    print(f"  R^2={result.rsquared:.4f}  sigma_x={result.params['sigma_x'].value:.1f}"
          f"+/-{result.params['sigma_x'].stderr or float('nan'):.1f} cm^-1  "
          f"({'AT BOUND' if at_bound else 'interior'})")
    print(f"  Ex={1e7/result.params['Ex'].value:.2f} nm   "
          f"Ax={result.params['Ax'].value:.4f}+/-{result.params['Ax'].stderr or float('nan'):.4f}\n")
    wn_fine = np.linspace(sub.wn.min(), sub.wn.max(), 300)
    ax.plot(1e7/wn_fine, model.eval(result.params, wn=wn_fine), '-', lw=1.5,
            label=f"{label} (R²={result.rsquared:.3f})")

ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Absorbance"); ax.legend(fontsize=8)
ax.set_title("Chl b Q_x: testing windows against the validated Chl a model")
plt.tight_layout()
plt.savefig("/tmp/chlb_qx_windows.png", dpi=130)
