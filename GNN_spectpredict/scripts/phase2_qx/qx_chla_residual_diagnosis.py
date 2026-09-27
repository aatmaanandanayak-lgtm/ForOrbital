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
mask = (df.wl_nm >= 555) & (df.wl_nm <= 600)
sub = df[mask].copy()
sub["wn"] = 1e7/sub.wl_nm
sub = sub.sort_values("wn").reset_index(drop=True)

def gauss_plus_linear(wn, Ex, sigma_x, Ax, slope, intercept):
    return Ax * np.exp(-0.5*((wn-Ex)/sigma_x)**2) + slope*wn + intercept

model = Model(gauss_plus_linear, independent_vars=["wn"])
params = model.make_params(Ex=1e7/577, sigma_x=dict(value=300,min=50,max=800),
                            Ax=dict(value=0.03,min=0), slope=0.0, intercept=0.0)
result = model.fit(sub["abs"].values, params, wn=sub["wn"].values)

fig, axes = plt.subplots(2, 1, figsize=(8,7), sharex=True, gridspec_kw={'height_ratios':[3,1]})
axes[0].plot(sub.wl_nm, sub["abs"], 'ko', ms=4, label="data")
wn_fine = np.linspace(sub.wn.min(), sub.wn.max(), 300)
axes[0].plot(1e7/wn_fine, model.eval(result.params, wn=wn_fine), 'r-', label=f"linear baseline (R²={result.rsquared:.3f})")
axes[0].legend(); axes[0].set_ylabel("Absorbance")
axes[0].set_title("Chl a Q_x: linear-baseline fit + residuals")

residuals = sub["abs"].values - model.eval(result.params, wn=sub["wn"].values)
axes[1].axhline(0, color='gray', lw=0.8)
axes[1].plot(sub.wl_nm, residuals, 'b.-', ms=5)
axes[1].set_ylabel("Residual"); axes[1].set_xlabel("Wavelength (nm)")
plt.tight_layout()
plt.savefig("/tmp/chla_qx_residuals.png", dpi=130)

print("Residuals by wavelength:")
for wl, r in zip(sub.wl_nm, residuals):
    print(f"  {wl:7.2f} nm   residual={r:+.4f}")
