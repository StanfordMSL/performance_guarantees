import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import bound_utils as bu

# set font size and LaTex params
plt.rcParams.update({
    'font.size': 20,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

# set figure size ratio
plt.figure(figsize=(10,6))


# sample GMM distribution
comp_1 = np.random.normal(6, 1, 1000)
comp_2 = np.random.normal(15, 4, 1000)
data = np.hstack((comp_1, comp_2))

tau = 0.7
E = np.mean(data)
VaR = np.quantile(data, tau)
CVaR = bu.MC_CVAR(data, tau)

print("# Samples= ", data.shape[0])
print("E[Y] = ", E)
print("VaR[Y] = ", VaR)
print("CVaR[Y] = ", CVaR)


# plot density
ax = sns.kdeplot(data, fill=True, linewidth=0)

# plot E[Y]
plt.axvline(E)
plt.text(E+0.1,0.1132,r'$\mathbb{E}[Y]$',rotation=-90)

# plot VaR[Y]
plt.axvline(VaR)
plt.text(VaR+0.1,0.1051,r'$\textup{VaR}_{\tau}(Y)$',rotation=-90)

# plot CVaR[Y]
plt.axvline(CVaR)
plt.text(CVaR+0.1,0.10,r'$\textup{CVaR}_{\tau}(Y)$',rotation=-90)

# plt.axhline(0.129)

ax.set(xlabel="$Y$")


image_name = 'experiments/figures/distribution_cartoon.svg'
plt.savefig(image_name)

plt.show()
