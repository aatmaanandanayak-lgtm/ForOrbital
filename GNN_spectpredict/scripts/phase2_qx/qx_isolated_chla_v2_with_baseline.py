import pandas as pd
import numpy as np
from lmfit import Model

df = pd.read_csv("Natural Chlorophylls/CHL007_Chl a, Et2O (Kobayashi, 2013).abs.TXT",
                  sep="\t", encoding="utf-8")
df.columns = ["wl_nm", "abs"]
df = df.sort_values("wl_nm").reset_index(drop=True)
mask = (df.wl_nm >= 555) & (df.wl_nm <= 600)
sub = df[mask].copy()
sub["wn"] = 1e7/sub.wl_nm
sub = sub.sort_values("wn").reset_index(drop=True)

def gauss_plus_baseline(wn, Ex, sigma_x, Ax, slope, intercept):
    return Ax * np.exp(-0.5*((wn-Ex)/sigma_x)**2) + slope*wn + intercept

model = Model(gauss_plus_baseline, independent_vars=["wn"])
params = model.make_params(
    Ex=1e7/577, sigma_x=dict(value=300, min=50, max=800),
    Ax=dict(value=0.03, min=0),
    slope=0.0, intercept=0.0,
)
result = model.fit(sub["abs"].values, params, wn=sub["wn"].values)
print(result.fit_report())
print(f"\nR^2 with baseline: {result.rsquared:.4f}  (was 0.332 without)")
print(f"sigma_x: {result.params['sigma_x'].value:.1f} +/- {result.params['sigma_x'].stderr or float('nan'):.1f} cm^-1  "
      f"({'still at bound' if result.params['sigma_x'].value>=790 else 'interior - well-determined now'})")
