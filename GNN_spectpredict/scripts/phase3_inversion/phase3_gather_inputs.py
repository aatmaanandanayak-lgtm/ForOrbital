"""
Re-run the VALIDATED Phase 2 fits (two-mode Qy, baseline-corrected Qx)
for a/d/f, with full parameter + stderr reporting, to get a clean,
complete, precisely-recorded set of inputs for Phase 3 - rather than
transcribe partial numbers from memory across many turns.
"""
import pandas as pd
import numpy as np
from lmfit import Model
from math import factorial
import json

def fc_2mode(wn, E00, hw1, S1, hw2, S2, sigma, A, n1max=3, n2max=2):
    total = np.zeros_like(wn)
    for n1 in range(n1max):
        w1 = np.exp(-S1) * S1**n1 / factorial(n1)
        for n2 in range(n2max):
            w2 = np.exp(-S2) * S2**n2 / factorial(n2)
            total += A * w1 * w2 * np.exp(-0.5*((wn-(E00+n1*hw1+n2*hw2))/sigma)**2)
    return total

def gauss_plus_linear(wn, Ex, sigma_x, Ax, slope, intercept):
    return Ax * np.exp(-0.5*((wn-Ex)/sigma_x)**2) + slope*wn + intercept

compounds = {
    "Chl a": dict(path="Natural Chlorophylls/CHL007_Chl a, Et2O (Kobayashi, 2013).abs.TXT",
                  qy_window=(590,710), qy_seed=(661,1080,400),
                  qx_window=(555,593), qx_seed=577),
    "Chl d": dict(path="Natural Chlorophylls/CHL023_Chl d, Et2O (Kobayashi, 2013).abs.txt",
                  qy_window=(620,720), qy_seed=(686,1072,400),
                  qx_window=(565,615), qx_seed=594),
    "Chl f": dict(path="Natural Chlorophylls/CHL031_Ch f, Et2O (Kobayashi, 2013).abs.txt",
                  qy_window=(630,730), qy_seed=(695,959,400),
                  qx_window=(578,625), qx_seed=605),
}

results = {}
for name, cfg in compounds.items():
    df = pd.read_csv(cfg["path"], sep="\t", encoding="utf-8")
    df.columns = ["wl_nm","abs"]; df = df.sort_values("wl_nm").reset_index(drop=True)

    # Qy two-mode
    lo,hi = cfg["qy_window"]
    sub = df[(df.wl_nm>=lo)&(df.wl_nm<=hi)].copy()
    sub["wn"]=1e7/sub.wl_nm; sub=sub.sort_values("wn").reset_index(drop=True)
    e00_seed, hw1_seed, hw2_seed = cfg["qy_seed"]
    m = Model(fc_2mode, independent_vars=["wn"], param_names=["E00","hw1","S1","hw2","S2","sigma","A"])
    p = m.make_params(E00=1e7/e00_seed, hw1=hw1_seed, S1=0.25, hw2=hw2_seed, S2=0.15, sigma=150, A=sub["abs"].max()*1.3)
    p["S1"].min=0.02; p["S1"].max=2; p["S2"].min=0; p["S2"].max=1.5
    p["hw1"].min=700; p["hw1"].max=1500; p["hw2"].min=100; p["hw2"].max=800
    p["sigma"].min=50; p["sigma"].max=400
    r_qy = m.fit(sub["abs"].values, p, wn=sub["wn"].values)

    # Qx baseline-corrected
    lo,hi = cfg["qx_window"]
    sub2 = df[(df.wl_nm>=lo)&(df.wl_nm<=hi)].copy()
    sub2["wn"]=1e7/sub2.wl_nm; sub2=sub2.sort_values("wn").reset_index(drop=True)
    m2 = Model(gauss_plus_linear, independent_vars=["wn"])
    p2 = m2.make_params(Ex=1e7/cfg["qx_seed"], sigma_x=dict(value=250,min=50,max=800),
                         Ax=dict(value=0.03,min=0), slope=0.0, intercept=0.0)
    r_qx = m2.fit(sub2["abs"].values, p2, wn=sub2["wn"].values)

    results[name] = dict(
        qy_E00=r_qy.params["E00"].value, qy_E00_err=r_qy.params["E00"].stderr,
        qx_Ex=r_qx.params["Ex"].value, qx_Ex_err=r_qx.params["Ex"].stderr,
        qy_R2=r_qy.rsquared, qx_R2=r_qx.rsquared,
    )
    print(f"{name}:")
    print(f"  Q_y E00 = {results[name]['qy_E00']:.2f} +/- {results[name]['qy_E00_err']:.2f} cm^-1  "
          f"({1e7/results[name]['qy_E00']:.2f} nm)   R2={results[name]['qy_R2']:.4f}")
    print(f"  Q_x Ex  = {results[name]['qx_Ex']:.2f} +/- {results[name]['qx_Ex_err']:.2f} cm^-1  "
          f"({1e7/results[name]['qx_Ex']:.2f} nm)   R2={results[name]['qx_R2']:.4f}")

with open("/tmp/phase3_inputs.json","w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to phase3_inputs.json")
