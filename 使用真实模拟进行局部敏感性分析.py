import numpy as np


def monte_carlo_moe(air_conc, part_coeff, oral_bioavail, oral_nd_frac, n_sim=20000):
    np.random.seed(12345)
    air_conc_sim = np.random.lognormal(mean=np.log(air_conc), sigma=np.log(2.0), size=n_sim)
    part_coeff_sim = np.random.lognormal(mean=np.log(part_coeff), sigma=np.log(2.0), size=n_sim)
    conc_lung_inhal = air_conc_sim * part_coeff_sim

    diet_total_ree = np.random.lognormal(mean=np.log(250.9), sigma=np.log(1.5), size=n_sim)
    diet_nd_intake = diet_total_ree * oral_nd_frac
    oral_bioavail_sim = np.random.uniform(low=oral_bioavail * 0.5, high=oral_bioavail * 2.0, size=n_sim)
    absorbed_nd = diet_nd_intake * oral_bioavail_sim
    conc_lung_oral = absorbed_nd / 70 * 0.1
    conc_lung_total = conc_lung_inhal + conc_lung_oral
    q95 = np.percentile(conc_lung_total, 95)
    moe = 0.0871 / q95
    return moe


# 基准参数
params = {
    'air_conc': 0.041,
    'part_coeff': 0.0073,
    'oral_bioavail': 0.001,
    'oral_nd_frac': 0.15
}

baseline = monte_carlo_moe(**params)
print(f"Baseline MOE (Q95) = {baseline:.2f}")

delta = 0.2
sensitivity = {}
for key in params:
    sens_list = []
    for sign in (-1, 1):
        perturbed = params.copy()
        perturbed[key] = perturbed[key] * (1 + sign * delta)
        new_moe = monte_carlo_moe(**perturbed)
        rel_change = (new_moe - baseline) / baseline
        sens = rel_change / delta
        sens_list.append(abs(sens))
        sign_str = '+' if sign == 1 else '-'
        print(
            f"{key} {sign_str}{delta * 100:.0f}%: MOE = {new_moe:.2f} (change = {rel_change:+.2%}), sensitivity = {sens:.2f}")
    sensitivity[key] = np.mean(sens_list)

print("\nSensitivity ranking (average absolute sensitivity):")
sorted_keys = sorted(sensitivity, key=sensitivity.get, reverse=True)
for k in sorted_keys:
    print(f"  {k}: {sensitivity[k]:.2f}")