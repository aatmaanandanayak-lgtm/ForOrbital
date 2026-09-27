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

# Wide window: from well below Qx through the full Qy band, so the fit
# can properly apportion intensity rather than have Qx truncated out
mask = (df.wl_nm >= 560) & (df.wl_nm <= 670)
sub = df[mask].copy()
sub["wn"] = 1e7/sub.wl_nm
sub = sub.sort_values("wn").reset_index(drop=True)

def joint_qx_qy2mode(wn, Ex, sigma_x, Ax,
                      E00, hw1, S1, hw2, S2, sigma_y, Ay,
                      n1max=3, n2max=2):
    """Q_x as a single Gaussian (start simple - add vibronic structure
    only if residuals demand it) + Q_y as the same two-mode Franck-Condon
    progression validated on a/d/f, fit SIMULTANEOUSLY so Qx and the
    Qy satellite aren't forced into an artificial window boundary."""
    qx = Ax * np.exp(-0.5*((wn-Ex)/sigma_x)**2)
    qy = np.zeros_like(wn)
    for n1 in range(n1max):
        w1 = np.exp(-S1) * S1**n1 / factorial(n1)
        for n2 in range(n2max):
            w2 = np.exp(-S2) * S2**n2 / factorial(n2)
            qy += Ay * w1 * w2 * np.exp(-0.5*((wn-(E00+n1*hw1+n2*hw2))/sigma_y)**2)
    return qx + qy

model = Model(joint_qx_qy2mode, independent_vars=["wn"],
              param_names=["Ex","sigma_x","Ax","E00","hw1","S1","hw2","S2","sigma_y","Ay"])
params = model.make_params(
    Ex=dict(value=1e7/596, min=1e7/615, max=1e7/580),
    sigma_x=dict(value=250, min=80, max=600),
    Ax=dict(value=0.06, min=0),
    E00=dict(value=1e7/644, min=1e7/650, max=1e7/638),
    hw1=dict(value=1130, min=900, max=1300),   # seeded from the a/d/f consistent cluster
    S1=dict(value=0.25, min=0.02, max=1.5),
    hw2=dict(value=550, min=200, max=800),
    S2=dict(value=0.13, min=0, max=1.0),
    sigma_y=dict(value=150, min=50, max=300),
    Ay=dict(value=sub["abs"].max()*1.3, min=0),
)
result = model.fit(sub["abs"].values, params, wn=sub["wn"].values)
print(result.fit_report())

print(f"\nRecovered Q_x origin: {1e7/result.params['Ex'].value:.2f} nm")
print(f"Recovered Q_y 0-0:    {1e7/result.params['E00'].value:.2f} nm")
print(f"Recovered Q_y hw1:    {result.params['hw1'].value:.1f} cm^-1  "
      f"(cf. isolated-window fit: 761 cm^-1; a/d/f cluster: 1111-1156 cm^-1)")
print(f"Recovered Q_y hw2:    {result.params['hw2'].value:.1f} cm^-1  "
      f"(cf. isolated-window fit: 354 cm^-1; a/d/f cluster: 512-636 cm^-1)")

# Decompose for plotting
def qx_only(wn, Ex, sigma_x, Ax, **kw):
    return Ax * np.exp(-0.5*((wn-Ex)/sigma_x)**2)

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(sub.wl_nm, sub["abs"], 'ko', ms=3, label="data")
wn_fine = np.linspace(sub.wn.min(), sub.wn.max(), 500)
wl_fine = 1e7/wn_fine
full = model.eval(result.params, wn=wn_fine)
qx_component = qx_only(wn_fine, result.params["Ex"].value, result.params["sigma_x"].value, result.params["Ax"].value)
ax.plot(wl_fine, full, 'b-', lw=1.5, label=f"joint fit (R²={result.rsquared:.4f})")
ax.plot(wl_fine, qx_component, 'g:', lw=1.5, label="Q_x component alone")
ax.plot(wl_fine, full-qx_component, 'm:', lw=1.2, label="Q_y component alone")
ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Absorbance"); ax.legend(fontsize=8)
ax.set_title("Chl b: joint Q_x + two-mode Q_y fit")
plt.tight_layout()
plt.savefig("/tmp/chlb_joint_fit.png", dpi=130)
