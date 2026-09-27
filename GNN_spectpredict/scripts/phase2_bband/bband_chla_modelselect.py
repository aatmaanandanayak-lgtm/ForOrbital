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

# Window: from the true data start (350nm - can't see further blue) out to
# 455nm (clear margin past the 431nm main peak, well short of the Qx/Qy region)
mask = (df.wl_nm >= 350) & (df.wl_nm <= 455)
sub = df[mask].copy()
sub["wn"] = 1e7/sub.wl_nm
sub = sub.sort_values("wn").reset_index(drop=True)

def two_gauss_baseline(wn, E1, s1, A1, E2, s2, A2, slope, intercept):
    return (A1*np.exp(-0.5*((wn-E1)/s1)**2) + A2*np.exp(-0.5*((wn-E2)/s2)**2)
            + slope*wn + intercept)

def three_gauss_baseline(wn, E1, s1, A1, E2, s2, A2, E3, s3, A3, slope, intercept):
    return (A1*np.exp(-0.5*((wn-E1)/s1)**2) + A2*np.exp(-0.5*((wn-E2)/s2)**2)
            + A3*np.exp(-0.5*((wn-E3)/s3)**2) + slope*wn + intercept)

# 2-component: seeded from the two strongest 2nd-deriv features (431, 405)
m2 = Model(two_gauss_baseline, independent_vars=["wn"])
p2 = m2.make_params(
    E1=1e7/431, s1=dict(value=300,min=100,max=900), A1=dict(value=0.9,min=0),
    E2=1e7/405, s2=dict(value=300,min=100,max=900), A2=dict(value=0.5,min=0),
    slope=0.0, intercept=0.0,
)
r2 = m2.fit(sub["abs"].values, p2, wn=sub["wn"].values)

# 3-component: seeded from all three 2nd-deriv features (431, 405, 373)
m3 = Model(three_gauss_baseline, independent_vars=["wn"])
p3 = m3.make_params(
    E1=1e7/431, s1=dict(value=250,min=100,max=700), A1=dict(value=0.9,min=0),
    E2=1e7/405, s2=dict(value=250,min=100,max=700), A2=dict(value=0.5,min=0),
    E3=1e7/373, s3=dict(value=250,min=100,max=700), A3=dict(value=0.3,min=0),
    slope=0.0, intercept=0.0,
)
r3 = m3.fit(sub["abs"].values, p3, wn=sub["wn"].values)

print("=== 2-component + baseline ===")
print(f"R^2={r2.rsquared:.4f}  AIC={r2.aic:.1f}  BIC={r2.bic:.1f}  n_params=8")
for lbl in ["E1","E2"]:
    print(f"  {lbl}: {1e7/r2.params[lbl].value:.2f} nm")
print()
print("=== 3-component + baseline ===")
print(f"R^2={r3.rsquared:.4f}  AIC={r3.aic:.1f}  BIC={r3.bic:.1f}  n_params=11")
for lbl in ["E1","E2","E3"]:
    print(f"  {lbl}: {1e7/r3.params[lbl].value:.2f} nm")

print(f"\nDelta-BIC (3-component minus 2-component): {r3.bic-r2.bic:+.1f}")
print("(more negative = 3-component justified despite extra params; ")
print(" near zero or positive = 2-component is sufficient, 3rd is overfitting)")

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(sub.wl_nm, sub["abs"], 'ko', ms=3, label="data")
wn_fine = np.linspace(sub.wn.min(), sub.wn.max(), 500)
ax.plot(1e7/wn_fine, m2.eval(r2.params, wn=wn_fine), 'r-', lw=1.3, label=f"2-comp (R²={r2.rsquared:.4f})")
ax.plot(1e7/wn_fine, m3.eval(r3.params, wn=wn_fine), 'b--', lw=1.3, label=f"3-comp (R²={r3.rsquared:.4f})")
ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Absorbance"); ax.legend()
ax.set_title("Chl a B-band: 2- vs 3-component model selection")
plt.tight_layout()
plt.savefig("/tmp/chla_bband_modelselect.png", dpi=130)

print("\n=== Full 3-component parameter report (checking for failure signatures) ===")
print(r3.fit_report())
for lbl, bound_lo, bound_hi in [("s1",100,700), ("s2",100,700), ("s3",100,700)]:
    val = r3.params[lbl].value
    flag = "** AT BOUND **" if (val<=bound_lo*1.02 or val>=bound_hi*0.98) else "(interior)"
    print(f"{lbl}={val:.1f}  {flag}")
for lbl in ["E1","E2","E3"]:
    nm = 1e7/r3.params[lbl].value
    edge_flag = "** NEAR WINDOW EDGE **" if (nm<=352 or nm>=453) else "(interior)"
    print(f"{lbl}={nm:.2f}nm  {edge_flag}")
