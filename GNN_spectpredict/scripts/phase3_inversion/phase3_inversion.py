"""
Phase 3: the Gouterman 2x2 CI inversion, run for the first time on real
data (Chl a, d, f).

Per polarization p, H_p = [[A_p, W_p],[W_p, B_p]]:
  Trace:    E(Q_p) + E(B_p) = A_p + B_p
  Splitting: E(B_p) - E(Q_p) = 2*sqrt(delta_p^2 + W_p^2) = 2*R_p
  delta_p = (A_p - B_p)/2
  Intensity ratio: f(Q_p)/f(B_p) = (1-sin2theta_p)/(1+sin2theta_p)
    => sin2theta_p = (1-r)/(1+r), r = f(Q_p)/f(B_p)
  W_p = R_p * sin2theta_p ;  delta_p = R_p * cos2theta_p
  => A_p = (E(Qp)+E(Bp))/2 + delta_p ;  B_p = (E(Qp)+E(Bp))/2 - delta_p

Redundancy relation (model-consistency check / here used PREDICTIVELY):
  A_x + B_x = A_y + B_y  =>  E(Qx)+E(Bx) = E(Qy)+E(By)
  =>  B_x - B_y = E(Qy) - E(Qx)
This holds regardless of intensities - a pure consequence of the trace
relation on both polarizations sharing the same four-orbital energy sum.
"""
import numpy as np
import json

with open("/tmp/phase3_inputs.json") as f:
    inputs = json.load(f)

def invert_polarization(E_Q, E_B, f_Q, f_B):
    """Full per-polarization inversion - needs E_Q, E_B, f_Q, f_B all for
    the SAME polarization p. Not yet usable for x or y individually since
    Phase 2 could not split the B-band into Bx/By - kept here as real,
    tested infrastructure for when Phase 4 unblocks that."""
    S = E_Q + E_B
    D = E_B - E_Q
    R = D / 2
    r = f_Q / f_B
    sin2theta = (1 - r) / (1 + r)
    cos2theta = np.sqrt(max(0, 1 - sin2theta**2))
    W = R * sin2theta
    delta = R * cos2theta
    A = S/2 + delta
    B = S/2 - delta
    return dict(A=A, B=B, W=W, delta=delta, sin2theta=sin2theta)

print("="*70)
print("BLOCKER: full per-polarization inversion needs E(Bx) and E(By)")
print("SEPARATELY. Phase 2 could only deliver an AGGREGATE B value (the")
print("3-component decomposition was confirmed real but not individually")
print("trustworthy - severe parameter correlations). So invert_polarization()")
print("above is real, tested code, but cannot be run on either x or y yet.")
print("="*70)

print("\nWhat CAN be done with real data right now: the redundancy relation")
print("does NOT need intensities or a Bx/By split - it gives a PREDICTION")
print("for how large that split must be, purely from Qx and Qy:\n")

N_MC = 200000
rng = np.random.default_rng(42)

bband_chla_components_cm = {"E1 (431nm)": 23260.5, "E2 (405nm)": 24316.4, "E3 (373nm)": 26419.1}

for name in ["Chl a", "Chl d", "Chl f"]:
    d = inputs[name]
    qy_samples = rng.normal(d["qy_E00"], d["qy_E00_err"], N_MC)
    qx_samples = rng.normal(d["qx_Ex"], d["qx_Ex_err"], N_MC)
    split_samples = qy_samples - qx_samples  # predicted Bx - By (or By-Bx, sign is a labeling choice)
    pred = np.mean(split_samples)
    pred_err = np.std(split_samples)
    print(f"{name}: predicted |B_x - B_y| = {abs(pred):.1f} +/- {pred_err:.1f} cm^-1  "
          f"(from Qy-Qx = {d['qy_E00']:.1f} - {d['qx_Ex']:.1f})")

print(f"\nCross-check against Chl a's empirical B-band sub-component gaps")
print(f"(from the Phase 2 3-component fit - real, if individually uncertain, positions):")
labels = list(bband_chla_components_cm.keys())
vals = list(bband_chla_components_cm.values())
for i in range(len(vals)):
    for j in range(i+1, len(vals)):
        print(f"  {labels[i]} <-> {labels[j]}: gap = {vals[j]-vals[i]:.1f} cm^-1")

d = inputs["Chl a"]
qy_samples = rng.normal(d["qy_E00"], d["qy_E00_err"], N_MC)
qx_samples = rng.normal(d["qx_Ex"], d["qx_Ex_err"], N_MC)
pred_split = np.mean(qy_samples - qx_samples)
print(f"\nPredicted splitting: {abs(pred_split):.1f} cm^-1")
print(f"Closest empirical gap: E3-E2 = {vals[2]-vals[1]:.1f} cm^-1 "
      f"(difference: {abs(abs(pred_split)-(vals[2]-vals[1])):.1f} cm^-1, "
      f"{100*abs(abs(pred_split)-(vals[2]-vals[1]))/abs(pred_split):.1f}% off)")
