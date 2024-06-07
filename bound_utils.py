import numpy as np
from scipy.stats import binom
import cvxpy as cp
import copy
import matplotlib.pyplot as plt
import traj_utils as trj
import pdb

### Bound Helpers ###
def compute_eps(delta, n):
	"""Compute DKW epsilon appearing in E[J], CVaR(tau) bounds."""
	return np.sqrt(-np.log(delta) / (2 * n))

def get_min_n(delta, tau=0):
	"""Compute min # samples needed for DKW epsilon to be <= 1-tau 
	(used to ensure non-trivial exp, CVaR bounds)."""
	n = int(np.ceil(-1 * np.log(delta) / (2 * (1-tau)**2)))
	return n

#### Exp Related ####

def expectation_bound(delta, b, J):
	"""E[J] upper bound holding with probability >= 1-delta from CVaR bound."""
	return CVAR_bound(0, delta, b, J)

def hoeff_expectation_bound(delta, a, b, J):
	"""E[J] upper bound holding with probability >= 1-delta using Hoeffding."""
	return (b-a) * np.sqrt(-np.log(delta) / (2*len(J))) + np.mean(J)

# Found in https://arxiv.org/pdf/2204.09833.pdf [8]
def akella_exp(delta, b, J):
	"""E[J] bound holding w.p. 1-delta given J <= b"""
	n = len(J)
	# 1. Solve for epsilon in terms of delta: 1-delta = 1-(1-epsilon)^N
	epsilon = 1-delta**(1/n)
	# 2. Find var bound
	var_bound = np.max(J)
	# 3. Use to form expectation bound
	exp_bound = var_bound * (1-epsilon) + b * epsilon

	return exp_bound

#### VaR Related ####

def miscoverage_prob_bin(n_max, tau, delta):
	"""Compute smallest n needed such that Pr(VaR(tau) <= max{sample_1,...,sample_n}) >= 1-delta."""
	n_min = 1
	iters = 0
	while (n_max - n_min > 1):
		n = np.ceil((n_min + n_max)/2)
		val = binom.cdf(n-1, n, tau)
		if val < 1-delta:
			n_min = n
		else:
			n_max = n
		iters += 1
	n = int(n_max)
	val = binom.cdf(n-1, n, tau)
	return n, val, (val >= 1-delta)

def k_miscoverage_prob_bin(n, tau, delta):
	"""Compute smallest k given n needed such that Bin(k-1; n, tau) >= 1-delta."""
	k_min = 1
	k_max = n
	iters = 0
	while (k_max - k_min > 1):
		k = np.ceil((k_min + k_max)/2)
		val = binom.cdf(k-1, n, tau)
		if val < 1-delta:
			k_min = k
		else:
			k_max = k
		iters += 1
	k = int(k_max)
	val = binom.cdf(k-1, n, tau)
	return k, val, (val >= 1-delta)

def visualize_var_tradeoff(n_values):
	"""Plot tau-delta tradeoff curves for the given n values."""
	tau_vals = np.linspace(0.001, 0.999)

	fig = plt.figure()
	plt.title('Visualizing VaR Tradeoff')
	plt.xlabel(r'$\tau$')
	plt.ylabel(r'$\delta$')

	deltas = np.zeros((len(tau_vals), len(tau_vals)))
	for i, n in enumerate(n_values):
		delta_vals = 1-binom.cdf(n-1, n, tau_vals)
		deltas[i,:] = delta_vals	
		plt.plot(tau_vals, delta_vals, label=n)
	plt.legend()
	plt.grid(True)
	plt.show(block=False)

	return fig, deltas

def k_miscoverage_prob_szorenyi(n, tau, delta):
	"""Compute smallest k given n, delta s.t. k/n >= tau + np.sqrt( (1/2*n) * log( (pi^2 * n^2) / (3*delta) ) )"""
	rhs = tau + np.sqrt( (1/(2*n)) * np.log( (np.pi**2 * n**2) / (3*delta) ) )
	k = np.ceil(n*rhs)
	success = (k <= n)
	return k, rhs, success

def k_miscoverage_prob_kolla(n, tau, delta):
	"""Compute smallest k given n, delta s.t. k/n >= tau + np.sqrt( (-2/n) * np.log( delta / 2 ) )""" 
	rhs = tau + np.sqrt( (-2/n) * np.log( delta / 2 ) )
	k = np.ceil(n * rhs)
	success = (k <= n)
	return k, rhs, success

#### CVaR Related ####

def MC_CVAR(J, tau):
	"""Use Monte Carlo cost samples to compute CVaR(tau) = E[J | J > VaR(tau)]."""
	# Our definition of VaR(tau) is as the tau quantile. 
	# The following optimization comes from source using alternate definition where VaR(tau) is 1-tau quantile
	eps = 1-tau
	n = len(J)
	t = cp.Variable()
	u = cp.Variable(n)
	obj = t + 1/(n * eps) * cp.sum(u)
	constr = [0 <= u, J - t * np.ones(n) <= u]
	prob = cp.Problem(cp.Minimize(obj), constr)
	prob.solve()
	return prob.value

def phil_CVAR_bound(tau, delta, b, J):
	"""CVaR(tau) upper bound holding with probability >= 1-delta."""
	n = len(J)
	ordered = np.append(J, b)
	ordered = np.sort(ordered)
	# i+1 - i order statistic
	diffs = ordered[1:] - ordered[:-1]
	# pos(i/n - sqrt(ln(1/delta)/(2n)) - tau) i=1,..,n
	unclipped = np.arange(1,n+1)/n - np.sqrt(np.log(1/delta) / (2*n)) - tau
	factor = np.clip(unclipped, 0, np.inf)
	bound = b - 1/(1-tau) * np.sum(diffs * factor)
	return bound

# Found in https://arxiv.org/pdf/2204.09833.pdf [8]
def akella_cvar(tau, delta, b, J):
	"""CVaR(tau)[J] bound holding w.p. 1-delta given J <= b"""
	n = len(J)
	# 1. Solve for epsilon in terms of delta
	epsilon = 1-delta**(1/n)
	# 2. Solve for alpha in terms of tau
	alpha = 1 - tau
	# 3. Get var bound
	var_bound = np.max(J)
	# 4. Setup optimization problem for CVaR
	mu = cp.Variable()
	obj = mu + 1/alpha * ( (1-epsilon) * cp.max(cp.vstack([var_bound - mu, 0])) + epsilon * cp.max(cp.vstack([b - mu, 0])) )
	prob = cp.Problem(cp.Minimize(obj))
	prob.solve()

	cvar_bound = prob.value

	return cvar_bound

def CVAR_bound(tau, delta, b, J):
	"""Our CVaR(tau) upper bound holding with probability >= 1-delta."""
	min_n = get_min_n(delta, tau)
	n = len(J)
	assert n >= min_n

	print('min_n', min_n)

	eps = compute_eps(delta, n)
	# -1 because python zero indexes
	k = int(np.ceil(n*(eps + tau)))-1
	sorted_J = np.sort(J)
	J_k = sorted_J[k]
	# -1 because python zero indexes
	if k < n-1:
		tail_sum = 1/n * np.sum(sorted_J[k+1:])
	else:
		tail_sum = 0

	return 1/(1-tau) * (eps * b + (k/n - eps - tau) * J_k + tail_sum)

#### Chance Related ####

def get_chance_stat(k, n, delta, tol=1e-3):
	"""Find smallest success probability p st. for X binomial(n, p) have P[X >= k] > delta"""
	p_min = 0
	p_max = 1
	iters = 0
	while (p_max - p_min > tol):
		p = (p_min + p_max)/2
		# P[X >= k] = 1 - P[X < k] = 1 - P[X <= k-1]
		val = 1-binom.cdf(k-1, n, p)
		if val > delta:
			p_max = p
		else:
			p_min = p
		iters += 1
	p = (p_min + p_max)/2
	val = 1 - binom.cdf(k-1, n, p)
	return p, val, (val > delta)

def min_n_chance(delta, tau):
	"""min n such that if all n samples are 1 will accept for chance constraint P[Y=1]>=tau"""
	return int(np.ceil(np.log(delta)/np.log(tau)))

def accept_chance(samples, delta, tau, tol=1e-3):
	"""Determine whether should accept or reject for chance constraint >= tau, maintaining false acceptance rate <= delta."""
	p_bar = get_chance_stat(np.sum(samples), len(samples), delta, tol)[0]
	accept = (p_bar >= tau)
	return accept, p_bar

#### Multi-hyp Related ####

def multi_hyp_correct(delta, m):
	"""Compute the new delta each individual hypothesis should use if comparing m hypotheses."""
	# Solve for delta_bar in (1 - delta_bar)^m = 1 - delta
	delta_bar = 1 - (1 - delta)**(1/m)
	return delta_bar

def visualize_multi_correct(delta_values, m_values):
	"""Plot necessary number of samples when doing multi-hypothesis correction."""
	new_deltas = np.zeros((len(delta_values), len(m_values)))
	for j, m in enumerate(m_values):
		new_deltas[:,j] = multi_hyp_correct(delta_values, m)
	
	fig = plt.figure()
	plt.title('Visualizing Multi-Hypothesis Correction')
	plt.xlabel('# Hypotheses')
	plt.ylabel(r'Corrected $\delta$')

	for i, delta in enumerate(delta_values):
		plt.plot(m_values, new_deltas[i,:], label=delta)
	plt.legend()
	plt.xticks(m_values)
	plt.grid(True)
	plt.show(block=False)

	return fig, new_deltas

#### Bound Empirical Verification ####

def analyze_bound_without_control(num_reps, action_seq, task, init_q_pos, init_q_vel, num_theory, theory_func, num_bound, bound_func, verbose=False):
	"""Repeatedly generate candidate bounds for the given quantity to see if holds, holding control sequence fixed."""
	fresh_bounds = np.zeros(num_reps)

	# Compute the theoretical value for best plan
	theory_costs, _ = trj.new_env_rollout(num_theory, init_q_pos, init_q_vel, action_seq, task)
	theory_val = theory_func(theory_costs)

	for rep in range(num_reps):
		if verbose:
			print('On rep ' + str(rep))

		# If we generate a new set of environments does bound hold?
		fresh_costs, _ = trj.new_env_rollout(num_bound, init_q_pos, init_q_vel, action_seq, task)
		fresh_bound = bound_func(fresh_costs)
		fresh_bounds[rep] = fresh_bound

		if verbose:
			print('Fresh bound: ', fresh_bound)

	return theory_costs, theory_val, fresh_bounds

def analyze_multiple_bounds_without_control(num_reps, action_seq, task, init_q_pos, init_q_vel, num_theory, theory_func, num_bound, bound_func_list, verbose=False):
	"""Repeatedly generate candidate bounds for the given quantity to see if holds, holding control sequence fixed."""
	num_bounds = len(bound_func_list)
	fresh_bounds = np.zeros((num_bounds, num_reps))

	# Compute the theoretical value for best plan
	theory_costs, _ = trj.new_env_rollout(num_theory, init_q_pos, init_q_vel, action_seq, task)
	theory_val = theory_func(theory_costs)

	for rep in range(num_reps):
		if verbose:
			print('On rep ' + str(rep))

		# If we generate a new set of environments does bound hold?
		fresh_costs, _ = trj.new_env_rollout(num_bound, init_q_pos, init_q_vel, action_seq, task)
		
		for i, bound_func in enumerate(bound_func_list):
			fresh_bound = bound_func(fresh_costs)
			fresh_bounds[i, rep] = fresh_bound

	return theory_costs, theory_val, fresh_bounds

def analyze_bound_with_shift(num_reps, action_seq, task, noise_scales, init_q_pos, init_q_vel, num_theory, theory_func, num_bound, bound_func, verbose=False):
	"""Generate bounds with nominal distribution while varying true distribution noise scales, holding control fixed."""
	fresh_bounds = np.zeros(num_reps)

	num_shifted = len(noise_scales)

	shifted_theory_costs = np.zeros((num_shifted, num_theory))
	shifted_theory_vals = np.zeros(num_shifted)

	nominal_noise = task.noise

	# Compute the theoretical value for each true distribution
	for i, noise in enumerate(noise_scales):
		if verbose:
			print('Generating shifted ' + str(i))
		# shifted_task = copy.deepcopy(task)
		task.noise = noise
		theory_costs, _ = trj.new_env_rollout(num_theory, init_q_pos, init_q_vel, action_seq, task)
		theory_val = theory_func(theory_costs)
		
		shifted_theory_costs[i] = theory_costs
		shifted_theory_vals[i] = theory_val

	# Also, compute for the nominal
	task.noise = nominal_noise
	nominal_costs, _ = trj.new_env_rollout(num_theory, init_q_pos, init_q_vel, action_seq, task)
	nominal_val = theory_func(nominal_costs)

	# Lastly, for the nominal distribution/task compute the bound distribution
	for rep in range(num_reps):
		if verbose:
			print('On rep ' + str(rep))

		# If we generate a new set of environments does bound hold?
		fresh_costs, _ = trj.new_env_rollout(num_bound, init_q_pos, init_q_vel, action_seq, task)
		fresh_bound = bound_func(fresh_costs)
		fresh_bounds[rep] = fresh_bound

		if verbose:
			print('Fresh bound: ', fresh_bound)

	return shifted_theory_costs, shifted_theory_vals, nominal_costs, nominal_val, fresh_bounds

def analyze_pr_with_shift(num_reps, action_seq, task, noise_scales, init_q_pos, init_q_vel, num_theory, constr_func, num_bound, delta, verbose=False):
	"""Like analyze_bound_with_shift but for probability of failure under distribution shift."""
	fresh_bounds = np.zeros(num_reps)

	num_shifted = len(noise_scales)

	shifted_theory_costs = np.zeros((num_shifted, num_theory))
	shifted_theory_vals = np.zeros(num_shifted)

	nominal_noise = task.noise
	# Compute the theoretical value for each true distribution
	for i, noise in enumerate(noise_scales):
		if verbose:
			print('Generating shifted ' + str(i))
		# shifted_task = copy.deepcopy(task)
		task.noise = noise
		_, observations = trj.new_env_rollout(num_theory, init_q_pos, init_q_vel, action_seq, task)
		# constr_func should take in sequence of observations in shape (# observations, horizon)
		# Assuming constr_func returns 1 when holds, 0 else
		satisfaction = [constr_func(observations[i,:,:]) for i in range(num_theory)]
		# Flip to costs
		theory_costs = 1 - np.array(satisfaction)
		# Compute the probability of failure i.e. P[J = 1] as average of 
		# theory_costs
		theory_val = np.mean(theory_costs)
		
		shifted_theory_costs[i] = theory_costs
		shifted_theory_vals[i] = theory_val

	# Also, compute for the nominal
	task.noise = nominal_noise
	_, observations = trj.new_env_rollout(num_theory, init_q_pos, init_q_vel, action_seq, task)
	satisfaction = [constr_func(observations[i,:,:]) for i in range(num_theory)]
	nominal_costs = 1 - np.array(satisfaction)
	nominal_val = np.mean(nominal_costs)

	# Lastly, for the nominal distribution/task compute the bound distribution
	for rep in range(num_reps):
		if verbose:
			print('On rep ' + str(rep))

		_, observations = trj.new_env_rollout(num_bound, init_q_pos, init_q_vel, action_seq, task)
		# samples stores whether constraint held (1) or violated (0) for each environment
		samples = np.array([constr_func(observations[i,:,:]) for i in range(num_bound)])
		# Determine lower bound on probability of success for this sample
		p_inf = get_chance_stat(np.sum(samples), len(samples), delta)[0]
		# Convert to an upper bound on probability of failure i.e. P[J = 1]
		fresh_bound = 1 - p_inf
		fresh_bounds[rep] = fresh_bound

		if verbose:
			print('Fresh bound: ', fresh_bound)

	return shifted_theory_costs, shifted_theory_vals, nominal_costs, nominal_val, fresh_bounds

def analyze_bound_random_control(num_reps, task, init_q_pos, init_q_vel, horizon, num_theory, theory_func, num_bound, bound_func, verbose=False):
	"""Repeatedly generate random control sequence, then corresponding candidate bounds for the given quantity to see if holds."""
	theory_vals = np.zeros(num_reps)
	fresh_bounds = np.zeros(num_reps)

	for rep in range(num_reps):

		if verbose:
			print('Starting rep ' + str(rep))

		# Sample a random plan
		action_seq = np.squeeze(task.sample_plans(1, horizon))

		# If we generate a new set of environments does bound hold?
		fresh_costs, _ = trj.new_env_rollout(num_bound, init_q_pos, init_q_vel, action_seq, task)
		fresh_bound = bound_func(fresh_costs)
		fresh_bounds[rep] = fresh_bound

		if verbose:
			print('Finished planning')

		# Compute the theoretical value for sampled plan
		theory_costs, _ = trj.new_env_rollout(num_theory, init_q_pos, init_q_vel, action_seq, task)
		theory_val = theory_func(theory_costs)
		theory_vals[rep] = theory_val

		if verbose:
			print('Plan theory value: ', theory_val)
			print('Plan fresh bound value: ', fresh_bound)

	return theory_vals, fresh_bounds

def analyze_bound_with_control(num_reps, solver, init_q_pos, init_q_vel, horizon, num_theory, theory_func, num_bound, bound_func, verbose=False):
	"""Repeatedly generate candidate bounds for the given quantity to see if holds, optimizing for the control sequence."""
	opt_vals = np.zeros(num_reps)
	theory_vals = np.zeros(num_reps)
	fresh_bounds = np.zeros(num_reps)

	task = solver.task

	for rep in range(num_reps):

		if verbose:
			print('Starting rep ' + str(rep))

		best_seq, best_val = solver.solve(init_q_pos, init_q_vel, horizon)
		opt_vals[rep] = best_val

		if verbose:
			print('Finished planning')

		# Compute the theoretical value for best plan
		theory_costs, _ = trj.new_env_rollout(num_theory, init_q_pos, init_q_vel, best_seq, task)
		theory_val = theory_func(theory_costs)
		theory_vals[rep] = theory_val

		if verbose:
			print('Best plan theory value: ', theory_val)
			print('Best plan optimization value: ', best_val)

		# If we generate a new set of environments does bound hold?
		fresh_costs, _ = trj.new_env_rollout(num_bound, init_q_pos, init_q_vel, best_seq, task)
		fresh_bound = bound_func(fresh_costs)
		fresh_bounds[rep] = fresh_bound

		if verbose:
			print('Fresh bound: ', fresh_bound)

	return opt_vals, theory_vals, fresh_bounds

def analyze_chance_without_control(num_reps, action_seq, task, init_q_pos, init_q_vel, num_theory, num_bound, constr_func, tau, delta, verbose=False):
	"""Repeatedly generate candidate bounds for the given quantity to see if holds, holding control sequence fixed."""
	accept_stats = np.zeros(num_reps)

	# Determine true fraction that the constraint holds for this action sequence
	# observations should have shape (# environments, # observations, horizon)
	_, observations = trj.new_env_rollout(num_theory, init_q_pos, init_q_vel, action_seq, task)
	# For the rollout results in each environment, check whether the constraint was violated
	# constr_func should take in sequence of observations in shape (# observations, horizon)
	satisfaction = [constr_func(observations[i,:,:]) for i in range(num_theory)]
	# Assuming constr_func returns 1 when holds, 0 else this is the fraction that constraint holds
	theory_val = np.mean(satisfaction)
	# 1 if chance constraint holds P[Y=1] >= tau, 0 else
	theory_holds = (theory_val >= tau)

	if verbose:
		print('True Level Satisfied: ', theory_val)

	for rep in range(num_reps):
		if verbose:
			print('On rep ' + str(rep))

		# If we generate a new set of environments does bound hold?
		_, observations = trj.new_env_rollout(num_bound, init_q_pos, init_q_vel, action_seq, task)
		# samples stores whether constraint held (1) or violated (0) for each environment
		samples = np.array([constr_func(observations[i,:,:]) for i in range(num_bound)])
		# Determine whether for this batch should accept or reject
		accept_stats[rep] = accept_chance(samples, delta, tau)[0]

		if verbose:
			print('Accept stat, pass: ', accept_stats[rep], accept_stats[rep] >= tau)

	# Compute the empirical fraction of the time that accepted
	accept_frac = np.mean(accept_stats >= tau)

	return theory_val, theory_holds, accept_stats, accept_frac

def analyze_pr_without_control(num_reps, action_seq, task, init_q_pos, init_q_vel, num_theory, num_bound, constr_func, delta, verbose=False):
	"""Repeatedly compute upper bound on probability of failure for given control sequence."""
	# Determine true fraction that the constraint holds for this action sequence
	# observations should have shape (# environments, # observations, horizon)
	_, observations = trj.new_env_rollout(num_theory, init_q_pos, init_q_vel, action_seq, task)
	# For the rollout results in each environment, check whether the constraint was violated
	# constr_func should take in sequence of observations in shape (# observations, horizon)
	# Assuming constr_func returns 1 when holds, 0 else
	satisfaction = [constr_func(observations[i,:,:]) for i in range(num_theory)]
	# Flip to costs
	theory_costs = 1 - np.array(satisfaction)
	# Compute the probability of failure i.e. P[J = 1] as average of 
	# theory_costs
	theory_val = np.mean(theory_costs)

	if verbose:
		print('theory_val', theory_val)

	fresh_bounds = np.zeros(num_reps)

	for rep in range(num_reps):
		if verbose:
			print('On rep ' + str(rep))

		_, observations = trj.new_env_rollout(num_bound, init_q_pos, init_q_vel, action_seq, task)
		# samples stores whether constraint held (1) or violated (0) for each environment
		samples = np.array([constr_func(observations[i,:,:]) for i in range(num_bound)])
		# Determine lower bound on probability of success for this sample
		p_inf = get_chance_stat(np.sum(samples), len(samples), delta)[0]
		# Convert to an upper bound on probability of failure i.e. P[J = 1]
		fresh_bound = 1 - p_inf
		fresh_bounds[rep] = fresh_bound

		if verbose:
			print('Fresh bound: ', fresh_bound)

	return theory_costs, theory_val, fresh_bounds


#### Theoretical Bound Sensitivity Utils ####


# https://towardsdatascience.com/comparing-sample-distributions-with-the-kolmogorov-smirnov-ks-test-a2292ad6fee5
def cdf(sample, x, sort=False):
    # Sorts the sample, if unsorted
    if sort:
        sample.sort()
    # Counts how many observations are below x
    cdf = sum(sample <= x)
    # Divides by the total number of observations
    cdf = cdf / len(sample)
    return cdf

# https://towardsdatascience.com/comparing-sample-distributions-with-the-kolmogorov-smirnov-ks-test-a2292ad6fee5
# Use one-sided KS distance i.e., look at nominal - shifted
def find_KS_distance(nominal_distribution, shifted_distribution):
	"""Given samples from two distributions, estimates the Kolmogorov-Smirnov distance
	as the sup-norm between the corresponding empirical CDFs."""

	sample1 = nominal_distribution.copy()
	sample2 = shifted_distribution.copy()

	# Gets all observations
	observations = np.concatenate((sample1, sample2))
	observations.sort()
	# Sorts the samples
	sample1.sort()
	sample2.sort()

	# Evaluates the KS statistic
	D_ks = [] # KS Statistic list
	for x in observations:
		cdf_sample1 = cdf(sample=sample1, x=x)
		cdf_sample2 = cdf(sample=sample2, x=x)
		# Would use abs for two-sided
		# D_ks.append(abs(cdf_sample1 - cdf_sample2))
		# One-sided because look at F_nominal - F_shifted
		D_ks.append(cdf_sample1 - cdf_sample2)
	alpha = max(D_ks)

	# Make sure non-negative
	alpha = max(alpha, 0)

	return alpha

def find_shifted_quantile(nominal_distribution, shifted_distribution, tau):
	"""Given samples from nominal and shifted distribution find tau' st. Var_{tau'}(J_sim) >= Var_{tau}(J_true).""" 
	
	# 1. Find Var_tau(J_true) := a using empirical quantile of shifted_distribution costs
	a = np.quantile(shifted_distribution, tau)

	# 2. Find smallest tau' := tau_shifted st. Var_{tau'}(J_sim) >= a
	# Equivalently, F_sim(tau')^{-1} >= a or tau' >= F_sim(a).
	# Explicitly, tau' >= P[J_sim <= a]
	# Approximate P[J_sim <= a] using empirical fraction of nominal costs <= a
	tau_shifted = np.mean(nominal_distribution <= a)

	return tau_shifted, a

### Robust Bounds ###

def robust_var_bound(tau, delta, J, alpha=0):
	"""Robust VaR(tau) upper bound holding with probability >= 1-delta even under KS shift alpha."""
	assert tau + alpha < 1
	n = len(J)
	k, val, success = k_miscoverage_prob_bin(n, tau + alpha, delta)

	assert success

	sorted_J = np.sort(J)
	J_k = sorted_J[k]
	return J_k

def robust_expectation_bound(delta, b, J, alpha=0):
	"""Robust E[J] upper bound holding with probability >= 1-delta even under KS shift alpha."""
	return robust_CVAR_bound(0, delta, b, J, alpha)

def robust_CVAR_bound(tau, delta, b, J, alpha=0):
	"""Robust CVaR(tau) upper bound holding with probability >= 1-delta even under KS shift alpha."""
	min_n = get_min_n(delta, tau + alpha)
	n = len(J)
	assert n >= min_n

	print('min_n', min_n)

	eps = compute_eps(delta, n)
	eps += alpha

	# -1 because python zero indexes
	k = int(np.ceil(n*(eps + tau)))-1
	sorted_J = np.sort(J)
	J_k = sorted_J[k]
	# -1 because python zero indexes
	if k < n-1:
		tail_sum = 1/n * np.sum(sorted_J[k+1:])
	else:
		tail_sum = 0

	return 1/(1-tau) * (eps * b + (k/n - eps - tau) * J_k + tail_sum)

def robust_q_bound(delta, J, alpha=0, tol=1e-3):
	# Assumes J=0 for success, J=1 for failure
	n = len(J)
	k = np.sum(J) # number of failures
	q, val, success = get_failure_bound(k, n, delta, alpha, tol)

	try:
		assert success
	except:
		pdb.set_trace()
	return q

def get_failure_bound(k, n, delta, alpha=0, tol=1e-3):
	q_min = 0
	q_max = 1
	iters = 0
	while (q_max - q_min > tol):
		q = (q_min + q_max)/2
		# val is decreasing wrt. increasing q
		val = binom.cdf(k, n, q - alpha)
		if val > delta:
			q_min = q
		else:
			q_max = q
		iters += 1
	# Use q_min not q = (q_min + q_max)/2
	# because only q_min is guaranteed to have val > delta
	q = q_min
	val = binom.cdf(k, n, q-alpha)
	success = val >= delta
	
	return q, val, success

def analyze_robust_bound(num_reps, action_seq, task, noise_scales, init_q_pos, init_q_vel, num_theory, theory_func, num_bound, bound_func, alphas=[], verbose=False):
	"""Generate robust bounds with nominal distribution and several alpha values while varying true distribution noise scales, holding control fixed."""
	if len(alphas) == 0:
		num_alphas = 3
	else:
		num_alphas = len(alphas)
	fresh_bounds = np.zeros((num_alphas, num_reps))

	num_shifted = len(noise_scales)

	shifted_theory_costs = np.zeros((num_shifted, num_theory))
	shifted_theory_vals = np.zeros(num_shifted)

	nominal_noise = task.noise

	# Compute the theoretical value for each true distribution
	for i, noise in enumerate(noise_scales):
		if verbose:
			print('Generating shifted ' + str(i))
		# shifted_task = copy.deepcopy(task)
		task.noise = noise
		theory_costs, _ = trj.new_env_rollout(num_theory, init_q_pos, init_q_vel, action_seq, task)
		theory_val = theory_func(theory_costs)
		
		shifted_theory_costs[i] = theory_costs
		shifted_theory_vals[i] = theory_val

	# Also, compute for the nominal
	task.noise = nominal_noise
	nominal_costs, _ = trj.new_env_rollout(num_theory, init_q_pos, init_q_vel, action_seq, task)
	nominal_val = theory_func(nominal_costs)

	# Compute for each of the shifted distributions the associated true alpha value
	ks_dists = np.array([find_KS_distance(nominal_costs, shifted_theory_costs[i]) for i in range(len(noise_scales))])

	if len(alphas) == 0:
		alphas = [0, np.median(ks_dists[ks_dists > 0]), np.max(ks_dists)]

	# Lastly, for the nominal distribution/task compute the bound distribution
	for rep in range(num_reps):
		if verbose:
			print('On rep ' + str(rep))

		# If we generate a new set of environments does bound hold?
		fresh_costs, _ = trj.new_env_rollout(num_bound, init_q_pos, init_q_vel, action_seq, task)
		for j, alpha in enumerate(alphas):
			fresh_bound = bound_func(fresh_costs, alpha)
			fresh_bounds[j, rep] = fresh_bound

			if verbose:
				print(f'Fresh bound (alpha = {alpha}): ', fresh_bound)

	return shifted_theory_costs, shifted_theory_vals, nominal_costs, nominal_val, fresh_bounds, ks_dists, alphas


def analyze_robust_pr(num_reps, action_seq, task, noise_scales, init_q_pos, init_q_vel, num_theory, constr_func, num_bound, delta, alphas=[], verbose=False):
	"""Like analyze_bound_with_shift but for probability of failure under distribution shift."""
	if len(alphas) == 0:
		num_alphas = 3
	else:
		num_alphas = len(alphas)
	fresh_bounds = np.zeros((num_alphas, num_reps))

	num_shifted = len(noise_scales)

	shifted_theory_costs = np.zeros((num_shifted, num_theory))
	shifted_theory_vals = np.zeros(num_shifted)

	nominal_noise = task.noise
	# Compute the theoretical value for each true distribution
	for i, noise in enumerate(noise_scales):
		if verbose:
			print('Generating shifted ' + str(i))
		# shifted_task = copy.deepcopy(task)
		task.noise = noise
		_, observations = trj.new_env_rollout(num_theory, init_q_pos, init_q_vel, action_seq, task)
		# constr_func should take in sequence of observations in shape (# observations, horizon)
		# Assuming constr_func returns 1 when holds, 0 else
		satisfaction = [constr_func(observations[i,:,:]) for i in range(num_theory)]
		# Flip to costs
		theory_costs = 1 - np.array(satisfaction)
		# Compute the probability of failure i.e. P[J = 1] as average of 
		# theory_costs
		theory_val = np.mean(theory_costs)
		
		shifted_theory_costs[i] = theory_costs
		shifted_theory_vals[i] = theory_val

	# Also, compute for the nominal
	task.noise = nominal_noise
	_, observations = trj.new_env_rollout(num_theory, init_q_pos, init_q_vel, action_seq, task)
	satisfaction = [constr_func(observations[i,:,:]) for i in range(num_theory)]
	nominal_costs = 1 - np.array(satisfaction)
	nominal_val = np.mean(nominal_costs)

	# Compute for each of the shifted distributions the associated true alpha value
	ks_dists = np.array([max(shifted_val - nominal_val, 0) for shifted_val in shifted_theory_vals])

	if len(alphas) == 0:
		alphas = [0, np.median(ks_dists[ks_dists > 0]), np.max(ks_dists)]

	# Lastly, for the nominal distribution/task compute the bound distribution
	for rep in range(num_reps):
		if verbose:
			print('On rep ' + str(rep))

		_, observations = trj.new_env_rollout(num_bound, init_q_pos, init_q_vel, action_seq, task)
		# samples stores whether constraint held (1) or violated (0) for each environment
		samples = np.array([constr_func(observations[i,:,:]) for i in range(num_bound)])
		
		for j, alpha in enumerate(alphas):
			# flip to J = 0 for success, J = 1 for failure
			fresh_bound = robust_q_bound(delta, 1-samples, alpha, tol=1e-3)
			fresh_bounds[j, rep] = fresh_bound

			if verbose:
				print(f'Fresh bound (alpha = {alpha}): ', fresh_bound)

	return shifted_theory_costs, shifted_theory_vals, nominal_costs, nominal_val, fresh_bounds, ks_dists, alphas

