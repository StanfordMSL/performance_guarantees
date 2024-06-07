import numpy as np
import matplotlib.pyplot as plt
import bound_utils as bu
import math
from scipy.stats import binom
import time



def k_miscoverage_prob_dkw(n, tau, delta):
	"""Compute smallest k given n needed such that k/n >= tau + np.sqrt( (1/2*n) * log( (pi^2 * n^2) / (3*delta) ) )."""
	rhs = tau + np.sqrt( (1/(2*n)) * np.log( (math.pi**2 * n**2) / (3*delta) ) )

	k_min = 1
	k_max = n
	iters = 0
	while (k_max - k_min > 1):
		k = np.ceil((k_min + k_max)/2)
		val = k / n
		if val < rhs:
			k_min = k
		else:
			k_max = k
		iters += 1
	k = int(k_max)
	val = k / n
	return k, val, (val >= rhs)

def k_miscoverage_prob_szorenyi_vec(n, taus, deltas):
	"""
	Compute smallest k given n needed such that k/n >= tau + np.sqrt( (1/2*n) * log( (pi^2 * n^2) / (3*delta) ) )
	tau, delta are 1D numpy arrays
	n is a natural number
	"""
	assert len(taus) == len(deltas)
	num_elems = len(taus)
	num_iters = int(np.ceil(np.log2(num_elems)))

	rhss = taus + np.sqrt( (1/(2*n)) * np.log( (math.pi**2 * n**2) / (3*deltas) ) )

	k_mins = np.ones(num_elems)
	k_maxs = n*np.ones(num_elems)

	# do binary search
	for iter in range(num_iters):
		# find new k and value
		ks = np.ceil((k_mins + k_maxs)/2)
		vals = ks / n

		# adjust bounds
		lb_idxs = np.where(vals < rhss)
		ub_idxs = np.setdiff1d(range(num_elems), lb_idxs)
		k_mins[lb_idxs] = ks[lb_idxs]
		k_maxs[ub_idxs] = ks[ub_idxs]

	k = k_maxs.astype(int)
	k = np.where(k/n - rhss >= 0, k, np.inf)
	return k

def k_miscoverage_prob_kolla_vec(n, taus, deltas):
	"""
	Compute smallest k given n needed such that k/n >= tau + np.sqrt( (1/2*n) * log( (pi^2 * n^2) / (3*delta) ) )
	tau, delta are 1D numpy arrays
	n is a natural number
	"""
	assert len(taus) == len(deltas)
	num_elems = len(taus)
	num_iters = int(np.ceil(np.log2(num_elems)))

	rhss = taus + np.sqrt( (-2/n) * np.log( deltas / 2 ) )

	k_mins = np.ones(num_elems)
	k_maxs = n*np.ones(num_elems)

	# do binary search
	for iter in range(num_iters):
		# find new k and value
		ks = np.ceil((k_mins + k_maxs)/2)
		vals = ks / n

		# adjust bounds
		lb_idxs = np.where(vals < rhss)
		ub_idxs = np.setdiff1d(range(num_elems), lb_idxs)
		k_mins[lb_idxs] = ks[lb_idxs]
		k_maxs[ub_idxs] = ks[ub_idxs]

	k = k_maxs.astype(int)
	k = np.where(k/n - rhss >= 0, k, np.inf)
	return k


def k_miscoverage_prob_bin_vec(n, taus, deltas):
	"""
	Compute smallest k given n needed such that k/n >= tau + np.sqrt( (1/2*n) * log( (pi^2 * n^2) / (3*delta) ) )
	tau, delta are 1D numpy arrays
	n is a natural number
	"""
	assert len(taus) == len(deltas)
	num_elems = len(taus)
	num_iters = int(np.ceil(np.log2(num_elems)))

	rhss = 1-deltas
	k_mins = np.ones(num_elems)
	k_maxs = n*np.ones(num_elems)

	# do binary search
	for iter in range(num_iters):
		# find new k and value
		ks = np.ceil((k_mins + k_maxs)/2)
		vals = binom.cdf(ks-1, n, taus)

		# adjust bounds
		lb_idxs = np.where(vals < rhss)
		ub_idxs = np.setdiff1d(range(num_elems), lb_idxs)
		k_mins[lb_idxs] = ks[lb_idxs]
		k_maxs[ub_idxs] = ks[ub_idxs]

	k = k_maxs.astype(int)
	k = np.where(binom.cdf(k-1, n, taus) - rhss >= 0, k, np.inf)
	return k


def single_plot(ns, tau, delta):
	our_bound = np.zeros(len(ns))
	dkw_bound = np.zeros(len(ns))

	for (i,n) in enumerate(ns):
		k_ours, val_ours, bool_ours = bu.k_miscoverage_prob_bin(n, tau, delta)
		k_dkw, val_dkw, bool_dkw = k_miscoverage_prob_dkw(n, tau, delta)

		our_bound[i] = k_ours if bool_ours else None 
		dkw_bound[i] = k_dkw if bool_dkw else None 


	first_k_ours = np.argmax(our_bound > 0)
	first_k_dkw = np.argmax(dkw_bound > 0)

	fig = plt.figure()
	plt.plot(ns[first_k_ours:], our_bound[first_k_ours:], label='Our Bound')
	plt.plot(ns[first_k_dkw:], dkw_bound[first_k_dkw:], label='DKW Bound')
	plt.legend()
	plt.xlabel('n')
	plt.ylabel('k')
	plt.grid(True)
	plt.title('Comparing VaR Bounds')
	plt.show()

def lower_bound_Bin(k,n,p):
	D = (k/n)*math.log(k/(n*p)) + (1-(k/n))*math.log((1-(k/n)) / (1-p))
	frac = (8*n*(k/n) * (1-(k/n)))**-0.5
	return frac*math.exp(-n*D)

def tradeoff_plot(n, taus, deltas):
	n = int(n)
	Gaps = np.zeros( (len(taus), len(deltas)) )
	min_gap = float('inf')
	min_bound = float('inf')
	min_obj = float('inf')

	for (j,tau) in enumerate(taus):
		for (k,delta) in enumerate(deltas):
			k_ours, val_ours, bool_ours = bu.k_miscoverage_prob_bin(n, tau, delta)
			k_dkw, val_dkw, bool_dkw = k_miscoverage_prob_dkw(n, tau, delta)

			true_obj = binom.cdf(k_dkw-1, n, tau) - 1 + delta
			# lower_bound = lower_bound_Bin(k_dkw-1,n,tau) - 1 + delta

			if bool_ours and not bool_dkw:
				Gaps[j,k] = n
			elif bool_ours and bool_dkw:
				Gaps[j,k]= k_dkw - k_ours
				if k_dkw - k_ours < min_gap:
					min_gap = k_dkw - k_ours
				# if lower_bound < min_bound:
				# 	min_bound = lower_bound
				if true_obj < min_obj:
					min_obj = true_obj
			elif not bool_ours and not bool_dkw:
				Gaps[j,k] = -n

				

	print("Min k Gap: ", min_gap)
	# print("Min Bound: ", min_bound)
	print("Min Obj: ", min_obj)


	fig = plt.figure()
	plt.imshow(Gaps, cmap='viridis', origin='lower')
	plt.colorbar()
	plt.grid(True)
	plt.title('Visualizing VaR Tradeoff')
	plt.xlabel(r'$\tau$')
	plt.ylabel(r'$\delta$')
	plt.title('k_dkw - k_ours: n=' + str(n))
	plt.show()

					
def plot_gaps(min_gaps_szorenyi, max_gaps_szorenyi, min_gaps_kolla, max_gaps_kolla):
	# only plot gaps where bounds were feasible
	max_idxs_szorenyi = np.flatnonzero(max_gaps_szorenyi != np.NAN)
	min_idxs_szorenyi = np.flatnonzero(min_gaps_szorenyi != np.NAN)
	max_idxs_kolla = np.flatnonzero(max_gaps_kolla != np.NAN)
	min_idxs_kolla = np.flatnonzero(min_gaps_kolla != np.NAN)

	fig = plt.figure(figsize=(5,4))
	plt.plot(max_idxs_kolla + 1, max_gaps_kolla[max_idxs_kolla],linewidth=2, color='orchid', label="Max Gap [6]")
	plt.plot(min_idxs_kolla + 1, min_gaps_kolla[min_idxs_kolla],linewidth=2, color='orchid', linestyle='dashed', label="Min Gap [6]")
	plt.plot(max_idxs_szorenyi + 1, max_gaps_szorenyi[max_idxs_szorenyi],linewidth=2, color='mediumblue', label="Max Gap [11]")
	plt.plot(min_idxs_szorenyi + 1, min_gaps_szorenyi[min_idxs_szorenyi],linewidth=2, color='mediumblue', linestyle='dashed', label="Min Gap [11]")
	plt.ylabel(r'$k_{others} - k_{ours}$')
	plt.xlabel(r'Sample Size, $n$')
	plt.legend(fontsize="9", loc="upper left")
	fig.savefig('experiments/figures/var_bound_comparison.svg')
	fig.savefig('experiments/figures/var_bound_comparison.png')
	plt.show(block=True)

	


def get_gaps(n_max, discretization):
	"""gap := k_dkw - k_ours"""
	# n_max = config["n_max"]
	# discretization = config["discretization"]

	# get discretization of tau and delta
	grid_pts = np.linspace(10**(-discretization), 1 - 10**(-discretization), num=int(10**(discretization) - 1))
	deltas, taus = np.meshgrid(grid_pts, grid_pts)
	deltas = deltas.flatten()
	taus = taus.flatten()
	M = len(deltas)

	# for each n record the max and min gap
	max_gaps_szorenyi = np.full(n_max, -np.Inf)
	min_gaps_szorenyi = np.full(n_max, np.Inf)

	max_gaps_kolla = np.full(n_max, -np.Inf)
	min_gaps_kolla = np.full(n_max, np.Inf)

	# loop through each n
	for n in np.arange(1,n_max+1,1):
		print("n: ", n, "/", n_max)
		ks_ours = k_miscoverage_prob_bin_vec(n, taus, deltas)
		ks_szorenyi = k_miscoverage_prob_szorenyi_vec(n, taus, deltas)
		ks_kolla = k_miscoverage_prob_kolla_vec(n, taus, deltas)

		# get binary arrays where 1 if others are feasible when we're not and 0 otherwise
		feas_comp_szorenyi = np.where( np.logical_and(ks_ours == np.full(M, np.inf), ks_szorenyi != np.full(M, np.inf)), 1, 0)
		feas_comp_kolla = np.where( np.logical_and(ks_ours == np.full(M, np.inf), ks_kolla != np.full(M, np.inf)), 1, 0)

		# record where they're feasible and we're not, then quit
		if np.any(feas_comp_szorenyi):
			idxs = feas_comp_szorenyi.nonzero()
			print("Szorenyi is feasible when we're not")
			print("n: ", n)
			print("taus: ", taus[idxs])
			print("deltas: ", deltas[idxs])

		if np.any(feas_comp_kolla):
			idxs = feas_comp_kolla.nonzero()
			print("Kolla is feasible when we're not")
			print("n: ", n)
			print("taus: ", taus[idxs])
			print("deltas: ", deltas[idxs])

		assert (not np.any(feas_comp_szorenyi)) and (not np.any(feas_comp_kolla))


		# find indices where k_szorenyi feasible, then get gaps
		feasible_szorenyi = np.where(ks_szorenyi != np.full(M, np.inf), 1,0).nonzero()
		gaps_szorenyi = ks_szorenyi[feasible_szorenyi] - ks_ours[feasible_szorenyi]
		if len(gaps_szorenyi) > 0:
			max_gaps_szorenyi[n-1] = np.max(gaps_szorenyi)
			min_gaps_szorenyi[n-1] = np.min(gaps_szorenyi)
		else:
			max_gaps_szorenyi[n-1] = np.NAN
			min_gaps_szorenyi[n-1] = np.NAN


		# find indices where k_kolla feasible, then get gaps
		feasible_kolla = np.where(ks_kolla != np.full(M, np.inf), 1,0).nonzero()
		gaps_kolla = ks_kolla[feasible_kolla] - ks_ours[feasible_kolla]
		if len(gaps_kolla) > 0:
			max_gaps_kolla[n-1] = np.max(gaps_kolla)
			min_gaps_kolla[n-1] = np.min(gaps_kolla)
		else:
			max_gaps_kolla[n-1] = np.NAN
			min_gaps_kolla[n-1] = np.NAN

	return min_gaps_szorenyi, max_gaps_szorenyi, min_gaps_kolla, max_gaps_kolla
				


##################################
if __name__ == "__main__":

	# single_plot(range(1, 500+1), 0.8, 0.1)

	# tradeoff_plot(100_000, np.linspace(0.001, 0.999, num=50), np.linspace(0.001, 0.999, num=50))

	# non-vectorized functions
	# start = time.time()
	# max_gaps, min_gaps = get_gaps(20)
	# print("elapsed non-vec: ", time.time() - start)

	# vectorized functions
	n_max, discretization = 100, 3
	start = time.time()
	min_gaps_szorenyi, max_gaps_szorenyi, min_gaps_kolla, max_gaps_kolla = get_gaps(n_max, discretization)
	print("elapsed vec: ", time.time() - start)

	print("min of min_gaps_szorenyi: ", np.min(min_gaps_szorenyi))
	print("min of min_gaps_kolla: ", np.min(min_gaps_kolla))

	# plot results and save figure
	plot_gaps(min_gaps_szorenyi, max_gaps_szorenyi, min_gaps_kolla, max_gaps_kolla)
	# plot_gaps(max_gaps_kolla, min_gaps_kolla)


