import pandas as pd
import numpy as np
from lmfit import Model

df = pd.read_csv("Natural Chlorophylls/CHL007_Chl a, Et2O (Kobayashi, 2013).abs.TXT",
                  sep="\t", encoding="utf-8")
df.columns = ["wl_nm", "abs"]
df = df.sort_values("wl_nm").reset_index(drop=True)
mask = (df.wl_nm >= 350) & (df.wl_nm <= 455)
sub = df[mask].copy()
sub["wn"] = 1e7/sub.wl_nm
sub = sub.sort_values("wn").reset_index(drop=True)

def three_gauss_baseline(wn, E1, s1, A1, E2, s2, A2, E3, s3, A3, slope, intercept):
    return (A1*np.exp(-0.5*((wn-E1)/s1)**2) + A2*np.exp(-0.5*((wn-E2)/s2)**2)
            + A3*np.exp(-0.5*((wn-E3)/s3)**2) + slope*wn + intercept)

model = Model(three_gauss_baseline, independent_vars=["wn"])
params = model.make_params(
    E1=1e7/431, s1=dict(value=300,min=100,max=2500), A1=dict(value=0.9,min=0),
    E2=1e7/405, s2=dict(value=300,min=100,max=2500), A2=dict(value=0.5,min=0),
    E3=1e7/373, s3=dict(value=300,min=100,max=2500), A3=dict(value=0.3,min=0),
    slope=0.0, intercept=0.0,
)
result = model.fit(sub["abs"].values, params, wn=sub["wn"].values)
print(result.fit_report())
for lbl in ["s1","s2","s3"]:
    val = result.params[lbl].value
    err = result.params[lbl].stderr
    print(f"{lbl}={val:.1f} +/- {err if err else 'UNDETERMINED'}  "
          f"{'** STILL AT/NEAR BOUND **' if val>=2400 else '(interior)'}")
