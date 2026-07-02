import numpy as np
import matplotlib.pyplot as plt

# Data (mean±SD, n=3)
doses = [0.2, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2]
means = [0.97875, 1.02792, 0.89083, 0.85815, 0.82752, 0.79858, 0.69226, 0.65893, 0.51436]
stds = [0.04107, 0.00742, 0.02120, 0.01436, 0.04616, 0.02561, 0.01636, 0.02686, 0.01195]

def poly3(x):
    g = 1.0129725
    beta = -0.10149897
    beta3 = -0.00467221
    return g + beta * x + beta3 * x**3

x_smooth = np.linspace(0, 3.5, 200)
y_smooth = poly3(x_smooth)

plt.errorbar(doses, means, yerr=stds, fmt='o', capsize=3, color='k', ecolor='gray', label='Observed')
plt.plot(x_smooth, y_smooth, 'b-', lw=2, label='Polynomial degree 3 fit')
bmd, bmdl, bmdu = 0.3573, 0.2619, 0.6130
plt.axvline(bmd, color='r', ls='--', label=f'BMD = {bmd} µM')
plt.axvline(bmdl, color='orange', ls='--', label=f'BMDL = {bmdl} µM')
plt.axvline(bmdu, color='green', ls='--', label=f'BMDU = {bmdu} µM')
plt.xlabel('Nd(NO₃)₃ concentration (µM)')
plt.ylabel('Cell viability (relative to control)')
plt.title('BMDS Polynomial degree 3 model fit (BMR = 1 SD)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('FigS1_BMDS_curve.png', dpi=300)
plt.show()