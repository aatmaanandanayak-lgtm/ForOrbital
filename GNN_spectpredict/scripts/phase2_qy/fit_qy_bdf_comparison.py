import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lmfit import Model
from math import factorial

def fc_1mode(wn, E00, hw, S, sigma, A, n_terms=4):
    total = np.zeros_like(wn)
    for n in range(n_terms):
        w = np.exp(-S) * S**n / factorial(n)
        total += A * w * np.exp(-0.5*((wn-(E00+n*hw))/sigma)**2)
    return total

def fc_2mode(wn, E00, hw1, S1, hw2, S2, sigma, A, n1max=3, n2max=2):
    total = np.zeros_like(wn)
    for n1 in range(n1max):
        w1 = np.exp(-S1) * S1**n1 / factorial(n1)
        for n2 in range(n2max):
            w2 = np.exp(-S2) * S2**n2 / factorial(n2)
            total += A * w1 * w2 * np.exp(-0.5*((wn-(E00+n1*hw1+n2*hw2))/sigma)**2)
    return total

compounds = {
    "Chl b": dict(path="Natural Chlorophylls/CHL015_Chl b, Et2O (Kobayashi, 2013).abs.txt",
                  wl_range=(610, 665), seed_e00=644, seed_hw1=1080, seed_hw2=400),
    "Chl d": dict(path="Natural Chlorophylls/CHL023_Chl d, Et2O (Kobayashi, 2013).abs.txt",
                  wl_range=(620, 720), seed_e00=686, seed_hw1=1072, seed_hw2=400),
    "Chl f": dict(path="Natural Chlorophylls/CHL031_Ch f, Et2O (Kobayashi, 2013).abs.txt",
                  wl_range=(630, 730), seed_e00=695, seed_hw1=959, seed_hw2=400),
}

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
results_summary = []

for ax, (name, cfg) in zip(axes, compounds.items()):
    df = pd.read_csv(cfg["path"], sep="\t", encoding="utf-8")
    df.columns = ["wl_nm", "abs"]
    df = df.sort_values("wl_nm").reset_index(drop=True)
    mask = (df.wl_nm >= cfg["wl_range"][0]) & (df.wl_nm <= cfg["wl_range"][1])
    sub = df[mask].copy()
    sub["wn"] = 1e7/sub.wl_nm
    sub = sub.sort_values("wn").reset_index(drop=True)

    m1 = Model(fc_1mode, independent_vars=["wn"], param_names=["E00","hw","S","sigma","A"])
    p1 = m1.make_params(E00=1e7/cfg["seed_e00"], hw=cfg["seed_hw1"], S=0.5, sigma=150, A=sub["abs"].max()*1.3)
    p1["S"].min=0.02; p1["S"].max=3; p1["sigma"].min=50; p1["sigma"].max=500
    p1["hw"].min=700; p1["hw"].max=1500
    r1 = m1.fit(sub["abs"].values, p1, wn=sub["wn"].values)

    m2 = Model(fc_2mode, independent_vars=["wn"], param_names=["E00","hw1","S1","hw2","S2","sigma","A"])
    p2 = m2.make_params(E00=1e7/cfg["seed_e00"], hw1=cfg["seed_hw1"], S1=0.3,
                         hw2=cfg["seed_hw2"], S2=0.15, sigma=150, A=sub["abs"].max()*1.3)
    p2["S1"].min=0.02; p2["S1"].max=2; p2["S2"].min=0; p2["S2"].max=1.5
    p2["hw1"].min=700; p2["hw1"].max=1500; p2["hw2"].min=100; p2["hw2"].max=800
    p2["sigma"].min=50; p2["sigma"].max=400
    r2 = m2.fit(sub["abs"].values, p2, wn=sub["wn"].values)

    e00_1 = 1e7/r1.params["E00"].value
    e00_2 = 1e7/r2.params["E00"].value
    delta_bic = r2.bic - r1.bic
    print(f"=== {name} ===")
    print(f"  Single-mode: R2={r1.rsquared:.4f} BIC={r1.bic:.1f}  E00={e00_1:.2f}nm  hw={r1.params['hw'].value:.1f}cm-1")
    print(f"  Two-mode:    R2={r2.rsquared:.4f} BIC={r2.bic:.1f}  E00={e00_2:.2f}nm  hw1={r2.params['hw1'].value:.1f}  hw2={r2.params['hw2'].value:.1f}cm-1  S2={r2.params['S2'].value:.3f}+/-{r2.params['S2'].stderr or float('nan'):.3f}")
    print(f"  Delta-BIC (two minus single): {delta_bic:+.1f}  ({'two-mode favored' if delta_bic < -2 else 'NOT favored / ambiguous'})")
    print()
    results_summary.append((name, r1, r2, delta_bic))

    ax.plot(sub.wl_nm, sub["abs"], 'ko', ms=3)
    wn_fine = np.linspace(sub.wn.min(), sub.wn.max(), 400)
    ax.plot(1e7/wn_fine, m1.eval(r1.params, wn=wn_fine), 'r-', lw=1.3, label=f"1-mode R2={r1.rsquared:.3f}")
    ax.plot(1e7/wn_fine, m2.eval(r2.params, wn=wn_fine), 'b--', lw=1.3, label=f"2-mode R2={r2.rsquared:.3f}")
    ax.set_title(name); ax.set_xlabel("nm"); ax.legend(fontsize=8)

axes[0].set_ylabel("Absorbance")
plt.tight_layout()
plt.savefig("/tmp/bdf_qy_fits.png", dpi=130)
