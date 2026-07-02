"""
AOP-BN Model v5: Nd Immunotoxicity
Chain: Nd → NLRP3 → M2 → IL1b → AO

Strategy: For each Nd discrete state, sample from coefficient posterior
          at the state midpoint, add measurement noise, discretize → CPT.
          This is transparent and avoids joint-sampling artifacts.
"""
import numpy as np
import os, json
np.random.seed(42)

# ============================================================
# 1. DATA
# ============================================================
Nd_all = np.array([0.0, 1.0, 2.5])
ub_nlrp3_m = np.array([0.802, 0.589, 0.733])
ub_nlrp3_s = np.array([0.020, 0.025, 0.019])
ub_noise = ub_nlrp3_s.mean()  # ~0.021

idx02 = np.array([0, 2])
Nd02 = Nd_all[idx02]
m2_m = np.array([30.2, 52.4])
m2_s = np.array([4.5, 7.1])
m2_noise = m2_s.mean()  # ~5.8

il1b_m = np.array([10.77, 15.12])
il1b_s = np.array([0.62, 0.19])
il1b_noise = il1b_s.mean()  # ~0.41

# ============================================================
# 2. BAYESIAN REGRESSION (coefficient posterior)
# ============================================================
def bayes_lr(x, y, y_sd, degree=1, n_samp=20000):
    n = len(x)
    cols = [np.ones(n)]
    for d in range(1, degree+1):
        cols.append(x**d)
    X = np.column_stack(cols)
    k = X.shape[1]
    tau2 = 100.0
    L0 = np.eye(k) / tau2
    W = np.diag(1.0 / (y_sd**2))
    Ln = X.T @ W @ X + L0
    bh = np.linalg.solve(Ln, X.T @ W @ y + L0 @ np.zeros(k))
    r = y - X @ bh
    an = 3.0; bn = 1.0 + 0.5 * (r.T @ W @ r + bh.T @ L0 @ bh)  # a0=2→an≈3.5, adjusted here
    # Actually use proper a0=2
    a0 = 2.0; an2 = a0 + n/2.0
    bn2 = 1.0 + 0.5*(r.T @ W @ r + bh.T @ L0 @ bh)
    sig2 = 1.0 / np.random.gamma(an2, 1.0/bn2, n_samp)
    beta = np.zeros((n_samp, k))
    iLn = np.linalg.inv(Ln)
    for i in range(n_samp):
        beta[i] = np.random.multivariate_normal(bh, iLn * sig2[i])
    return beta, np.sqrt(sig2)

def pred_at_x(x_val, beta, degree, noise_sd, n_per=500):
    """Predict at a single x_val, using all beta samples."""
    n_samp = len(beta)
    total = n_samp * n_per
    y = np.zeros(total)
    for i in range(n_samp):
        mu = beta[i, 0]
        for d in range(1, degree+1):
            mu += beta[i, d] * (x_val ** d)
        for j in range(n_per):
            y[i*n_per + j] = mu + noise_sd * np.random.randn()
    return y

beta_nlrp3, _ = bayes_lr(Nd_all, ub_nlrp3_m, ub_nlrp3_s, degree=2)
beta_m2, _    = bayes_lr(Nd02, m2_m, m2_s, degree=1)
beta_il1b, _  = bayes_lr(Nd02, il1b_m, il1b_s, degree=1)

print("=" * 65)
print("  AOP-BN v5: Nd → NLRP3 → M2 → IL1b → AO")
print("=" * 65)
print(f"\n  Nd→NLRP3 quadratic: a={beta_nlrp3[:,0].mean():.4f}±{beta_nlrp3[:,0].std():.4f}")
print(f"    b1={beta_nlrp3[:,1].mean():.4f}±{beta_nlrp3[:,1].std():.4f}")
print(f"    b2={beta_nlrp3[:,2].mean():.4f}±{beta_nlrp3[:,2].std():.4f}")
print(f"  Nd→M2 linear:       a={beta_m2[:,0].mean():.2f}±{beta_m2[:,0].std():.2f}, b={beta_m2[:,1].mean():.2f}±{beta_m2[:,1].std():.2f}")
print(f"  Nd→IL1b linear:     a={beta_il1b[:,0].mean():.2f}±{beta_il1b[:,0].std():.2f}, b={beta_il1b[:,1].mean():.2f}±{beta_il1b[:,1].std():.2f}")

# ============================================================
# 3. DISCRETIZATION SCHEMES
# ============================================================
Nd_states   = ["Low", "Medium", "High"]
Nd_rep      = [0.5, 1.5, 2.5]   # representative points

NLRP3_states = ["Low", "Medium", "High"]
NLRP3_bounds = [(1.0, 10.0), (0.6, 1.0), (0.0, 0.6)]

M2_states    = ["Low", "Medium", "High"]
M2_bounds    = [(0.0, 30.0), (30.0, 50.0), (50.0, 100.0)]

IL1b_states  = ["Low", "Medium", "High"]
IL1b_bounds  = [(0.0, 12.0), (12.0, 16.0), (16.0, 100.0)]

AO_states    = ["Unlikely", "Possible", "Likely"]

def discretize(y_vals, bounds):
    """Count fraction in each bound-defined state."""
    n = len(y_vals)
    counts = np.zeros(len(bounds))
    for i, (lo, hi) in enumerate(bounds):
        counts[i] = np.sum((y_vals >= lo) & (y_vals < hi))
    # Anything above the last upper bound goes to last state
    counts[-1] += np.sum(y_vals >= bounds[-1][1])
    probs = counts / n
    probs = probs / probs.sum()
    return probs

# ============================================================
# 4. CPT: Nd (prior)
# ============================================================
cpt_Nd = np.ones((1,3)) / 3

# ============================================================
# 5. CPT: NLRP3 | Nd
# ============================================================
print("\n" + "=" * 65)
print("  BUILDING CPTs (per-state posterior predictive)")
print("=" * 65)

cpt_NLRP3_Nd = np.zeros((3, 3))
print(f"\n[CPT] NLRP3_deubiquitination | Nd (quadratic KER)")
for i, (nd_s, nd_r) in enumerate(zip(Nd_states, Nd_rep)):
    y_vals = pred_at_x(nd_r, beta_nlrp3, degree=2, noise_sd=ub_noise)
    cpt_NLRP3_Nd[i] = discretize(y_vals, NLRP3_bounds)
    # Debug
    print(f"  Nd={nd_s:6s} (x={nd_r}): pred Ub/NLRP3 mean={y_vals.mean():.4f} "
          f"| L={cpt_NLRP3_Nd[i,0]:.3f} M={cpt_NLRP3_Nd[i,1]:.3f} H={cpt_NLRP3_Nd[i,2]:.3f}")

# ============================================================
# 6. CPT: M2 | NLRP3
# ============================================================
# Strategy: For each Nd state, predict M2 and NLRP3 jointly, then
# condition M2 on NLRP3 state. Use Nd as the common driver.
cpt_M2_NLRP3 = np.zeros((3, 3))
print(f"\n[CPT] M2_polarization | NLRP3_deubiquitination")

# Generate joint (NLRP3, M2) for Nd uniform 0-3
N_joint = 50000
Nd_j = np.random.uniform(0, 3, N_joint)
nlrp3_j = np.zeros(N_joint)
m2_j = np.zeros(N_joint)
for i in range(N_joint):
    nd = Nd_j[i]
    ib = np.random.randint(0, len(beta_nlrp3))
    nlrp3_j[i] = beta_nlrp3[ib,0] + beta_nlrp3[ib,1]*nd + beta_nlrp3[ib,2]*nd**2 + ub_noise*np.random.randn()
    ib = np.random.randint(0, len(beta_m2))
    m2_j[i] = beta_m2[ib,0] + beta_m2[ib,1]*nd + m2_noise*np.random.randn()

# Discretize NLRP3 (safely handle upper boundary of first bound)
nlrp3_s_j = np.zeros(N_joint, dtype=int)
for k, (lo, hi) in enumerate(NLRP3_bounds):
    nlrp3_s_j[(nlrp3_j >= lo) & (nlrp3_j < hi)] = k
nlrp3_s_j[nlrp3_j >= NLRP3_bounds[0][1]] = 0   # >=10 → Low

for p in range(3):
    mask = (nlrp3_s_j == p)
    n_p = mask.sum()
    if n_p > 0:
        for c in range(3):
            lo, hi = M2_bounds[c]
            cnt = np.sum((m2_j[mask] >= lo) & (m2_j[mask] < hi))
            if c == 2:
                cnt += np.sum(m2_j[mask] >= M2_bounds[-1][1])
            cpt_M2_NLRP3[p, c] = (cnt + 1.0) / (n_p + 3.0)
        cpt_M2_NLRP3[p] /= cpt_M2_NLRP3[p].sum()
    else:
        cpt_M2_NLRP3[p] = np.ones(3) / 3

for i, nlrp3_s in enumerate(NLRP3_states):
    print(f"  NLRP3={nlrp3_s:6s}: L={cpt_M2_NLRP3[i,0]:.3f} M={cpt_M2_NLRP3[i,1]:.3f} H={cpt_M2_NLRP3[i,2]:.3f}")

# ============================================================
# 7. CPT: IL1b | M2
# ============================================================
cpt_IL1b_M2 = np.zeros((3, 3))
print(f"\n[CPT] IL1b_secretion | M2_polarization")

il1b_j = np.zeros(N_joint)
for i in range(N_joint):
    nd = Nd_j[i]
    ib = np.random.randint(0, len(beta_il1b))
    il1b_j[i] = beta_il1b[ib,0] + beta_il1b[ib,1]*nd + il1b_noise*np.random.randn()

m2_s_j = np.zeros(N_joint, dtype=int)
for k, (lo, hi) in enumerate(M2_bounds):
    m2_s_j[(m2_j >= lo) & (m2_j < hi)] = k
m2_s_j[m2_j >= M2_bounds[-1][1]] = len(M2_bounds) - 1   # ≥100 → High

for p in range(3):
    mask = (m2_s_j == p)
    n_p = mask.sum()
    if n_p > 0:
        for c in range(3):
            lo, hi = IL1b_bounds[c]
            cnt = np.sum((il1b_j[mask] >= lo) & (il1b_j[mask] < hi))
            if c == 2:
                cnt += np.sum(il1b_j[mask] >= IL1b_bounds[-1][1])
            cpt_IL1b_M2[p, c] = (cnt + 1.0) / (n_p + 3.0)
        cpt_IL1b_M2[p] /= cpt_IL1b_M2[p].sum()
    else:
        cpt_IL1b_M2[p] = np.ones(3) / 3

for i, m2_s in enumerate(M2_states):
    print(f"  M2={m2_s:6s}: L={cpt_IL1b_M2[i,0]:.3f} M={cpt_IL1b_M2[i,1]:.3f} H={cpt_IL1b_M2[i,2]:.3f}")

# ============================================================
# 8. CPT: AO | IL1b (expert-informed)
# ============================================================
cpt_AO_IL1b = np.array([
    [0.70, 0.25, 0.05],
    [0.15, 0.55, 0.30],
    [0.05, 0.25, 0.70],
])
print(f"\n[CPT] AO | IL1b")
for i, il1b_s in enumerate(IL1b_states):
    print(f"  IL1b={il1b_s:6s}: U={cpt_AO_IL1b[i,0]:.2f} P={cpt_AO_IL1b[i,1]:.2f} L={cpt_AO_IL1b[i,2]:.2f}")

# ============================================================
# 9. FORWARD PREDICTION
# ============================================================
print("\n" + "=" * 65)
print("  FORWARD PREDICTION")
print("=" * 65)

Nd_bounds = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]

def forward(nd_c):
    nd_idx = 0
    for i, (lo, hi) in enumerate(Nd_bounds):
        if lo <= nd_c < hi: nd_idx = i; break
    if nd_c >= Nd_bounds[-1][1]: nd_idx = 2

    p_nlrp3 = cpt_NLRP3_Nd[nd_idx]
    p_m2 = np.zeros(3)
    for j in range(3): p_m2 += p_nlrp3[j] * cpt_M2_NLRP3[j]
    p_il1b = np.zeros(3)
    for j in range(3): p_il1b += p_m2[j] * cpt_IL1b_M2[j]
    p_ao = np.zeros(3)
    for j in range(3): p_ao += p_il1b[j] * cpt_AO_IL1b[j]
    ao_w = p_ao[0]*0.10 + p_ao[1]*0.35 + p_ao[2]*0.75
    return nd_idx, p_nlrp3, p_m2, p_il1b, p_ao, ao_w

tests = [
    (0.00038, "Q50 human lung"),
    (0.00158, "Q95 human lung"),
    (0.262,    "BMDL"),
    (0.357,    "BMD"),
    (0.5,      "0.5 μM (Low)"),
    (1.0,      "1.0 μM"),
    (1.5,      "1.5 μM (Med)"),
    (2.0,      "2.0 μM"),
    (2.5,      "2.5 μM (High)"),
]

for nd_c, label in tests:
    nd_idx, p_nlrp3, p_m2, p_il1b, p_ao, ao_w = forward(nd_c)
    print(f"\n  [{label}] Nd={nd_c:.5f} → {Nd_states[nd_idx]}")
    print(f"    NLRP3: L={p_nlrp3[0]:.3f} M={p_nlrp3[1]:.3f} H={p_nlrp3[2]:.3f}")
    print(f"    M2:    L={p_m2[0]:.3f} M={p_m2[1]:.3f} H={p_m2[2]:.3f}")
    print(f"    IL1b:  L={p_il1b[0]:.3f} M={p_il1b[1]:.3f} H={p_il1b[2]:.3f}")
    print(f"    AO:    U={p_ao[0]*100:.1f}% P={p_ao[1]*100:.1f}% L={p_ao[2]*100:.1f}%  |  E[AO]={ao_w*100:.1f}%")

# ============================================================
# 10. DOSE-RESPONSE
# ============================================================
print("\n" + "=" * 65)
print("  DOSE-RESPONSE CURVE")
print("=" * 65)
nd_curve = np.linspace(0.0001, 3.0, 30)
for nd_c in nd_curve[::2]:
    _, _, _, _, _, ao_w = forward(nd_c)
    bar = "#" * int(ao_w * 80)
    print(f"  {nd_c:.4f} μM  {ao_w*100:.1f}%  {bar}")

# ============================================================
# 11. NETICA .dne
# ============================================================
output_dir = r"C:\Users\wangning\AppData\Roaming\Tencent\Marvis\User\oAN1i2eqkIJ6qD9R98ixvZQsjxjI\workspace\conv_19e4a798d7a_75e346a17a3b\output"
os.makedirs(output_dir, exist_ok=True)

Q = '"'
def fmt_row(p): return "(" + " ".join([f"{x:.4f}" for x in p]) + ")"

def add_node(lines, name, states, parents, cpt):
    lines.append(f"node {name} {{")
    lines.append("    kind = NATURE;")
    lines.append("    discrete = TRUE;")
    lines.append(f"    states = ({' '.join([f'{Q}{s}{Q}' for s in states])});")
    if parents:
        lines.append(f"    parents = ({' '.join(parents)});")
    lines.append("    prob = (")
    for i in range(cpt.shape[0]):
        cmt = ""
        if parents and len(parents) == 1:
            cmt = f"   // {parents[0]} = {states[i]}"
        lines.append(f"        {fmt_row(cpt[i])}{cmt}")
    lines.append("    );")
    lines.append("}")
    lines.append("")

lines = []
lines.append("// =================================================================")
lines.append("// AOP-BN: Nd Immunotoxicity (Chain: Nd→NLRP3→M2→IL1b→AO)")
lines.append("// Generated: 2026-05-21 | Method: Bayesian regression + CPT discretization")
lines.append("// KER Nd→NLRP3: quadratic (3 data points)")
lines.append("// KER Nd→M2:     linear   (2 data points)")
lines.append("// KER Nd→IL1b:   linear   (2 data points)")
lines.append("// Netica v6.09+ compatible")
lines.append("// =================================================================")
lines.append("")

add_node(lines, "Nd_concentration", Nd_states, [], cpt_Nd)
add_node(lines, "NLRP3_deubiquitination", NLRP3_states, ["Nd_concentration"], cpt_NLRP3_Nd)
add_node(lines, "M2_polarization", M2_states, ["NLRP3_deubiquitination"], cpt_M2_NLRP3)
add_node(lines, "IL1b_secretion", IL1b_states, ["M2_polarization"], cpt_IL1b_M2)
add_node(lines, "Immunosuppression_AO", AO_states, ["IL1b_secretion"], cpt_AO_IL1b)

dne_path = os.path.join(output_dir, "Nd_AOP_BN.dne")
with open(dne_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(lines))
print(f"\n✅ Netica: {dne_path}")

# JSON
cpt_json = {
    "network": "Nd_AOP_BN",
    "structure": "Nd → NLRP3 → M2 → IL1b → AO",
    "discretization": {
        "Nd": Nd_states,
        "NLRP3_deubiquitination": {"states": NLRP3_states, "bounds": [[1,10],[0.6,1],[0,0.6]]},
        "M2_polarization": {"states": M2_states, "bounds": [[0,30],[30,50],[50,100]]},
        "IL1b_secretion": {"states": IL1b_states, "bounds": [[0,12],[12,16],[16,100]]},
    },
    "cpt": {
        "Nd_concentration": cpt_Nd[0].tolist(),
        "NLRP3_deubiquitination": [r.tolist() for r in cpt_NLRP3_Nd],
        "M2_polarization": [r.tolist() for r in cpt_M2_NLRP3],
        "IL1b_secretion": [r.tolist() for r in cpt_IL1b_M2],
        "Immunosuppression_AO": [r.tolist() for r in cpt_AO_IL1b],
    },
    "ker_fit": {
        "Nd_to_NLRP3": {
            "model": "quadratic", "formula": "Ub/NLRP3 = a + b1*Nd + b2*Nd²",
            "a":  {"mean": float(beta_nlrp3[:,0].mean()), "sd": float(beta_nlrp3[:,0].std())},
            "b1": {"mean": float(beta_nlrp3[:,1].mean()), "sd": float(beta_nlrp3[:,1].std())},
            "b2": {"mean": float(beta_nlrp3[:,2].mean()), "sd": float(beta_nlrp3[:,2].std())},
            "noise_sd": float(ub_noise),
        },
        "Nd_to_M2": {
            "model": "linear",
            "a": {"mean": float(beta_m2[:,0].mean()), "sd": float(beta_m2[:,0].std())},
            "b": {"mean": float(beta_m2[:,1].mean()), "sd": float(beta_m2[:,1].std())},
            "noise_sd": float(m2_noise),
        },
        "Nd_to_IL1b": {
            "model": "linear",
            "a": {"mean": float(beta_il1b[:,0].mean()), "sd": float(beta_il1b[:,0].std())},
            "b": {"mean": float(beta_il1b[:,1].mean()), "sd": float(beta_il1b[:,1].std())},
            "noise_sd": float(il1b_noise),
        },
    }
}
json_path = os.path.join(output_dir, "Nd_AOP_BN.json")
with open(json_path, 'w') as f:
    json.dump(cpt_json, f, indent=2)
print(f"✅ JSON:  {json_path}")
print("\n" + "=" * 65)
print("  DONE — import Nd_AOP_BN.dne into Netica")
print("=" * 65)