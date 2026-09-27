import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lmfit import Model
from math import factorial

df = pd.read_csv("Natural Chlorophylls/CHL015_Chl b, Et2O (Kobayashi, 2013).abs.txt",
                  sep="\t", encoding="utf-8")
df.columns = ["wl_nm", "abs"]
df = df.sort_values("wl_nm").reset_index(drop=True)
mask = (df.wl_nm >= 560) & (df.wl_nm <= 670)
sub = df[mask].copy()
sub["wn"] = 1e7/sub.wl_nm
sub = sub.sort_values("wn").reset_index(drop=True)

def joint_qx_qy2mode(wn, Ex, sigma_x, Ax,
                      E00, hw1, S1, hw2, S2, sigma_y, Ay,
                      n1max=3, n2max=2):
    qx = Ax * np.exp(-0.5*((wn-Ex)/sigma_x)**2)
    qy = np.zeros_like(wn)
    for n1 in range(n1max):
        w1 = np.exp(-S1) * S1**n1 / factorial(n1)
        for n2 in range(n2max):
            w2 = np.exp(-S2) * S2**n2 / factorial(n2)
            qy += Ay * w1 * w2 * np.exp(-0.5*((wn-(E00+n1*hw1+n2*hw2))/sigma_y)**2)
    return qx + qy

# hw1/hw2 FIXED at the a/d/f average - well-determined, mutually consistent,
# physically the same macrocycle mode. Chl b's fit only resolves amplitudes,
# positions, and widths - not asked to redetermine mode frequencies from
# data where Qx and the satellite are entangled.
HW1_FIXED = (1146 + 1111 + 1156) / 3   # 1137.7 cm^-1
HW2_FIXED = (558 + 512 + 636) / 3       # 568.7 cm^-1

model = Model(joint_qx_qy2mode, independent_vars=["wn"],
              param_names=["Ex","sigma_x","Ax","E00","hw1","S1","hw2","S2","sigma_y","Ay"])
params = model.make_params(
    Ex=dict(value=1e7/596, min=1e7/620, max=1e7/580),
    sigma_x=dict(value=300, min=80, max=900),   # widened - let's see where it actually wants to go
    Ax=dict(value=0.06, min=0),
    E00=dict(value=1e7/644, min=1e7/650, max=1e7/638),
    hw1=dict(value=HW1_FIXED, vary=False),
    S1=dict(value=0.25, min=0.02, max=1.5),
    hw2=dict(value=HW2_FIXED, vary=False),
    S2=dict(value=0.13, min=0, max=1.0),
    sigma_y=dict(value=150, min=50, max=300),
    Ay=dict(value=sub["abs"].max()*1.3, min=0),
)
result = model.fit(sub["abs"].values, params, wn=sub["wn"].values)
print(result.fit_report())

print(f"\nRecovered Q_x origin: {1e7/result.params['Ex'].value:.2f} nm  "
      f"width(sigma_x)={result.params['sigma_x'].value:.1f} cm^-1 "
      f"(bound was 80-900 - {'AT BOUND' if result.params['sigma_x'].value>=895 or result.params['sigma_x'].value<=85 else 'interior, well-behaved'})")
print(f"Recovered Q_y 0-0:    {1e7/result.params['E00'].value:.2f} nm")
print(f"S1={result.params['S1'].value:.3f}+/-{result.params['S1'].stderr or float('nan'):.3f}  "
      f"S2={result.params['S2'].value:.3f}+/-{result.params['S2'].stderr or float('nan'):.3f}")
print(f"\nKey correlation to check: C(Ax,S1) = ", end="")
try:
    print(f"{result.params['Ax'].correl['S1']:.3f}  (was -0.91 in unconstrained fit)")
except (KeyError, TypeError):
    print("not available")

def qx_only(wn, Ex, sigma_x, Ax, **kw):
    return Ax * np.exp(-0.5*((wn-Ex)/sigma_x)**2)

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(sub.wl_nm, sub["abs"], 'ko', ms=3, label="data")
wn_fine = np.linspace(sub.wn.min(), sub.wn.max(), 500)
wl_fine = 1e7/wn_fine
full = model.eval(result.params, wn=wn_fine)
qx_component = qx_only(wn_fine, result.params["Ex"].value, result.params["sigma_x"].value, result.params["Ax"].value)
ax.plot(wl_fine, full, 'b-', lw=1.5, label=f"joint fit, modes fixed (R²={result.rsquared:.4f})")
ax.plot(wl_fine, qx_component, 'g:', lw=1.5, label="Q_x component alone")
ax.plot(wl_fine, full-qx_component, 'm:', lw=1.2, label="Q_y component alone")
ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Absorbance"); ax.legend(fontsize=8)
ax.set_title("Chl b: joint fit, Q_y modes fixed from a/d/f")
plt.tight_layout()
plt.savefig("/tmp/chlb_constrained_fit.png", dpi=130)
