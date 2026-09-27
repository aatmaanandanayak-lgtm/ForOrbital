import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lmfit import Model

def gauss_plus_linear(wn, Ex, sigma_x, Ax, slope, intercept):
    return Ax * np.exp(-0.5*((wn-Ex)/sigma_x)**2) + slope*wn + intercept

model = Model(gauss_plus_linear, independent_vars=["wn"])

compounds = {
    "Chl d": dict(path="Natural Chlorophylls/CHL023_Chl d, Et2O (Kobayashi, 2013).abs.txt",
                  window=(565, 615), seed_ex=594),
    "Chl f": dict(path="Natural Chlorophylls/CHL031_Ch f, Et2O (Kobayashi, 2013).abs.txt",
                  window=(578, 625), seed_ex=605),
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, (name, cfg) in zip(axes, compounds.items()):
    df = pd.read_csv(cfg["path"], sep="\t", encoding="utf-8")
    df.columns = ["wl_nm", "abs"]
    df = df.sort_values("wl_nm").reset_index(drop=True)
    lo, hi = cfg["window"]
    mask = (df.wl_nm >= lo) & (df.wl_nm <= hi)
    sub = df[mask].copy()
    sub["wn"] = 1e7/sub.wl_nm
    sub = sub.sort_values("wn").reset_index(drop=True)

    params = model.make_params(Ex=1e7/cfg["seed_ex"], sigma_x=dict(value=200,min=50,max=800),
                                Ax=dict(value=0.03,min=0), slope=0.0, intercept=0.0)
    result = model.fit(sub["abs"].values, params, wn=sub["wn"].values)
    at_bound = result.params['sigma_x'].value >= 790
    ex_at_edge = abs(1e7/result.params['Ex'].value - hi) < 2 or abs(1e7/result.params['Ex'].value - lo) < 2

    print(f"=== {name} (window {lo}-{hi}nm) ===")
    print(f"  R^2={result.rsquared:.4f}")
    print(f"  Ex={1e7/result.params['Ex'].value:.2f} nm  "
          f"{'** AT WINDOW EDGE - SUSPECT **' if ex_at_edge else '(interior, trustworthy)'}")
    print(f"  sigma_x={result.params['sigma_x'].value:.1f}+/-{result.params['sigma_x'].stderr or float('nan'):.1f} cm^-1  "
          f"{'** AT BOUND **' if at_bound else '(interior)'}")
    print(f"  Ax={result.params['Ax'].value:.4f}+/-{result.params['Ax'].stderr or float('nan'):.4f}\n")

    ax.plot(sub.wl_nm, sub["abs"], 'ko', ms=4, label="data")
    wn_fine = np.linspace(sub.wn.min(), sub.wn.max(), 300)
    ax.plot(1e7/wn_fine, model.eval(result.params, wn=wn_fine), 'r-', lw=1.5,
            label=f"fit (R²={result.rsquared:.3f})")
    ax.set_title(f"{name} Q_x"); ax.set_xlabel("nm"); ax.legend(fontsize=8)

axes[0].set_ylabel("Absorbance")
plt.tight_layout()
plt.savefig("/tmp/qx_df_fits.png", dpi=130)
