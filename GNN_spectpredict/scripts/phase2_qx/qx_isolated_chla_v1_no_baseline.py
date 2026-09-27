import pandas as pd
import numpy as np
from lmfit import Model

df = pd.read_csv("Natural Chlorophylls/CHL007_Chl a, Et2O (Kobayashi, 2013).abs.TXT",
                  sep="\t", encoding="utf-8")
df.columns = ["wl_nm", "abs"]
df = df.sort_values("wl_nm").reset_index(drop=True)

# Chl a's Q_x sits cleanly isolated ~577nm, well clear of the 615nm
# satellite and the B-band tail - fit it alone to get an independently
# determined width, to use as a template for Chl b's entangled case.
mask = (df.wl_nm >= 555) & (df.wl_nm <= 600)
sub = df[mask].copy()
sub["wn"] = 1e7/sub.wl_nm
sub = sub.sort_values("wn").reset_index(drop=True)

def gauss(wn, Ex, sigma_x, Ax):
    return Ax * np.exp(-0.5*((wn-Ex)/sigma_x)**2)

model = Model(gauss, independent_vars=["wn"])
params = model.make_params(Ex=1e7/577, sigma_x=dict(value=300, min=50, max=800), Ax=dict(value=0.06, min=0))
result = model.fit(sub["abs"].values, params, wn=sub["wn"].values)
print(result.fit_report())
print(f"\nChl a Q_x width (sigma_x): {result.params['sigma_x'].value:.1f} +/- {result.params['sigma_x'].stderr:.1f} cm^-1")
print(f"(bound was 50-800: {'AT BOUND - even this isolated fit is not clean' if result.params['sigma_x'].value>=790 else 'interior, well-determined'})")
