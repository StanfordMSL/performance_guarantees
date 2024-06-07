import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom


### VaR Bound
from bound_utils import k_miscoverage_prob_bin

def get_delta_true_VaR(n, delta_sim, tau, tau_prime):
  # given n & delta_sim find k*
  k_star, _, success = k_miscoverage_prob_bin(n, tau, delta_sim)

  if not success:
    k_star = n+1

  # given k* find delta_true
  delta_true = 1 - binom.cdf(k_star-1, n, tau_prime)
  return delta_true



### Expectation/CVaR Bound
def get_delta_true_CVaR(n, delta_sim, alpha):
  # check that alpha is small enough for analysis to be valid
  assert(alpha <= np.sqrt(-np.log(delta_sim) / (2*n)) - np.sqrt(np.log(2) / (2*n)))

  # find delta_true
  delta_true = delta_sim * np.exp(-2*n*alpha**2 + 4*n*alpha*np.sqrt(-np.log(delta_sim) / (2*n)))
  return delta_true



### Failure Probability Bound
# def get_delta_true_Failure(n, delta_sim, p_sim, p_true):
#   # given n & delta_sim find t*
#   t_star = get_t_star(n, delta_sim, p_sim)

#   # given t* find delta_true
#   delta_true = 1 - binom.cdf(t_star + 1, n, p_true)
#   return delta_true


# def get_t_star(n, delta_sim, p_sim):
#   # check t \in [n, n-1, ..., 0]
#   for t in np.arange(n, -1, -1):
#     if binom.cdf(t, n, p_sim) <= 1 - delta_sim:
#       return t


def get_delta_true_Failure(n, delta_sim, p_sim, p_true):
  # Convert from input probabilities of success to probabilities of failure
  q_sim = 1 - p_sim
  q_true = 1 - p_true

  # given n & delta_sim find k*
  k_star = get_k_star(n, delta_sim, q_true)

  # given k* find delta_true
  delta_true = binom.cdf(k_star - 1, n, q_sim)
  return delta_true


def get_k_star(n, delta_sim, q_true):
  # check k \in [0, 1, ..., n-1, n]
  # min{k in [0,1,...,n] | Bin(k;n,q_true) >= delta_sim}
  for k in np.arange(1, n+1):
    if binom.cdf(k, n, q_true) >= delta_sim:
      return k





if __name__ == '__main__':
    ### Make Plots
    eps = 0.01

    delta_sim = 0.2
    n = 100


    ## VaR
    import matplotlib.colors as colors
    taus = np.linspace(eps, 1-eps, num=50)
    true_coverage = np.zeros((len(taus), len(taus)))

    for i,tau in enumerate(taus):
      for j,tau_prime in enumerate(taus):
        delta_true = get_delta_true_VaR(n, delta_sim, tau, tau_prime)
        true_coverage[i,j] = 1 - delta_true


    fig = plt.figure(figsize=(7,5))
    X, Y = np.meshgrid(taus, taus)
    plt.imshow(np.transpose(true_coverage),vmin=0, vmax=1, interpolation='none', cmap=plt.cm.RdYlGn, origin='lower', 
               extent=[X.min(), X.max(), Y.min(), Y.max()])
    plt.colorbar()

    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$\tau^\prime$')
    plt.title(r'True Coverage ($1-\delta_{true}$) Given $n=100$, $\delta_{sim}=0.2$')
    plt.show(block=True)

    fig.savefig('experiments/figures/sensitivity_var.svg', bbox_inches='tight')
    fig.savefig('experiments/figures/sensitivity_var.png', bbox_inches='tight')



    ## CVaR
    alpha_ub =  np.sqrt(-np.log(delta_sim) / (2*n)) - np.sqrt(np.log(2) / (2*n))
    alphas = np.linspace(0, np.min((alpha_ub, 1-eps)), num=50)
    true_coverage = np.zeros(len(alphas))

    for i,alpha in enumerate(alphas):
        delta_true = get_delta_true_CVaR(n, delta_sim, alpha)
        true_coverage[i] = 1 - delta_true

    fig = plt.figure(figsize=(7,5))
    plt.plot(alphas, true_coverage)
    plt.axhline(y = 1-delta_sim, linestyle='--', color='gray')

    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'True Coverage, $1-\delta_{true}$')
    plt.title(r'True Coverage ($1-\delta_{true}$) Given $n=100$, $\delta_{sim}=0.2$')
    plt.show(block=True)

    fig.savefig('experiments/figures/sensitivity_cvar.svg', bbox_inches='tight')
    fig.savefig('experiments/figures/sensitivity_cvar.png', bbox_inches='tight')



    ## Failure Probability
    ps = np.linspace(eps, 1-eps, num=50)
    true_coverage = np.zeros((len(ps), len(ps)))

    for i,p_sim in enumerate(ps):
      for j,p_true in enumerate(ps):
        delta_true = get_delta_true_Failure(n, delta_sim, p_sim, p_true)
        true_coverage[i,j] = 1 - delta_true


    fig = plt.figure(figsize=(7,5))
    X, Y = np.meshgrid(ps, ps)
    plt.imshow(np.transpose(true_coverage),vmin=0, vmax=1, interpolation='none', cmap=plt.cm.RdYlGn, origin='lower', 
               extent=[X.min(), X.max(), Y.min(), Y.max()])
    plt.colorbar()

    plt.xlabel(r'$p_{sim}$')
    plt.ylabel(r'$p_{true}$')
    plt.title(r'True Coverage ($1-\delta_{true}$) Given $n=100$, $\delta_{sim}=0.2$')
    plt.show(block=True)

    fig.savefig('experiments/figures/sensitivity_failure_prob.svg', bbox_inches='tight')
    fig.savefig('experiments/figures/sensitivity_failure_prob.png', bbox_inches='tight')


