import numpy as np
from traj_utils import eval_plans, new_env_eval_plans
from scipy.stats import truncnorm

class CEM():
	"""An implementation of cross-entropy method for action sequence selection."""
	def __init__(self, max_gen, num_samples, num_elite, num_envs, task, cost_stat,
		feas_stat=None, num_chance=None, rand_envs=True, processes=1, verbose=False):
		# Maximum number of generations to perform
		self.max_gen = max_gen
		# Number of action sequences to draw in each generation
		self.num_samples = num_samples
		# Number of elite samples to keep in each generation
		self.num_elite = num_elite
		# Number of environments to draw for evaluating each plan
		self.num_envs = num_envs
		# Task object for which are planning
		self.task = task
		# Cost statistic (e.g. mean, quantile) which specifies how to aggregate costs for one plan across environments
		self.cost_stat = cost_stat
		# Feasibility function which should take in a sequence of observations in shape (num_envs, # observations/step, horizon)
		# determine whether to accept given action sequence as feasible
		self.feas_stat = feas_stat
		# How many environments to use when assessing feasibility
		self.num_chance = num_chance if num_chance is not None else num_envs
		# True if want CEM to generate random envs when evaluating each plan, False if want to fix same draws for all plans in a generation
		self.rand_envs = rand_envs
		# Number of processes to use, 1 if no multiprocess
		self.processes = processes
		# True to print progress
		self.verbose = verbose

	# TODO: Potentially use truncnorm.fit instead
	def get_moments(self, plans):
		"""Get mean and variance for each element of each action sequence in plans."""
		# plans should have shape (N = # action_seqs, d = # actions/env, H = horizon) 
		# Now d x H
		mus = np.mean(plans, axis=0)
		sigmas = np.std(plans, axis=0)

		return mus, sigmas

	def truncated_normal_sample(self, num_samples, mus, sigmas):
		"""Allow for both full and truncated normal sampling."""
		# mus, sigmas should have shape d = # actions/env, H = horizon
		d, H = mus.shape
		
		# low should be -inf, high should be inf in a given component if action_space unbounded there
		rep_low = np.repeat(self.task.action_space.low[:, np.newaxis], repeats=H, axis=1)
		rep_high = np.repeat(self.task.action_space.high[:, np.newaxis], repeats=H, axis=1)

		return truncnorm.rvs(a=rep_low, b=rep_high, loc=mus, scale=sigmas, size=(num_samples, *mus.shape))

	def one_generation(self, mus, sigmas, init_q_pos, init_q_vel):
		"""Perform one generation of CEM."""
		# 1. Draw num_samples new action sequences
		plans = self.truncated_normal_sample(self.num_samples, mus, sigmas)

		# 2. Assess each of the action sequences for feasibility and cost
		feas, tot_costs = self.compute_plan_info(init_q_pos, init_q_vel, plans)

		# 3. Identify the elite samples
		# Compute a statistic across environments to evaluate each plan
		stats = np.zeros(self.num_samples)

		# Infeasible plans are considered to have infinite cost
		count = 0
		for i in range(self.num_samples):
			if feas[i]:
				stats[i] = self.cost_stat(tot_costs[count,:])
				count += 1
			else:
				stats[i] = np.inf

		# Assumes lower cost is better
		ordered = np.argsort(stats)
		
		# Return the elite sequences and their corresponding statistics
		elite_inds = ordered[:self.num_elite] 
		elite_seqs = plans[elite_inds,:,:]
		elite_stats= stats[elite_inds]

		return elite_seqs, elite_stats

	def compute_plan_info(self, init_q_pos, init_q_vel, plans, num_draws=None, num_draws_chance=None, feas_stat=None):
		if num_draws is None:
			num_draws = self.num_envs
		if num_draws_chance is None:
			num_draws_chance = self.num_chance
		if feas_stat is None:
			feas_stat = self.feas_stat

		# Identify which action sequences are deemed feasible/acceptable
		if feas_stat is not None:
			if not self.rand_envs: 
				if self.processes != 1:
					raise ValueError("Cannot parallelize while fixing environments. Need processes = 1 when rand_envs = False")
				
				# 2. Draw num_envs new environments
				environments = self.task.create_envs(num_draws_chance)

				# 3. Evaluate the action sequences on the new environments
				tot_costs, obs_arr = eval_plans(environments, init_q_pos, init_q_vel, plans, self.task)

				environments.close()
			else:
				tot_costs, obs_arr = new_env_eval_plans(num_draws_chance, init_q_pos, init_q_vel, plans, self.task, self.processes)

			feas = np.zeros(len(plans)).astype('bool')

			for i in range(len(plans)):
				if feas_stat(obs_arr[i,:,:,:]):
					feas[i] = True

			# Filter out infeasible plans
			feas_plans = plans[feas]
		else:
			feas = np.ones(len(plans))
			feas_plans = plans

		# Get costs for each remaining action sequence
		if not self.rand_envs: 
			if self.processes != 1:
				raise ValueError("Cannot parallelize while fixing environments. Need processes = 1 when rand_envs = False")
			
			# Draw num_envs new environments
			environments = self.task.create_envs(num_draws)

			# Evaluate the action sequences on the new environments
			tot_costs, obs_arr = eval_plans(environments, init_q_pos, init_q_vel, feas_plans, self.task)

			environments.close()
		else:
			tot_costs, obs_arr = new_env_eval_plans(num_draws, init_q_pos, init_q_vel, feas_plans, self.task, self.processes)

		return feas, tot_costs

	def solve(self, init_q_pos, init_q_vel, horizon, init_plans=None):
		"""Perform CEM possibly starting from an initial set of plans."""
		if init_plans is None:
			init_plans = self.task.sample_plans(self.num_samples, horizon)

		mus, sigmas = self.get_moments(init_plans)
		
		if self.verbose:
			print("Initial fit done")
		
		avg_stats = []
		
		elite_seqs = init_plans[:]
		elite_stats = np.zeros(len(elite_seqs))
		
		# TODO: Potentially add an early stopping tolerance criterion
		for gen in range(self.max_gen):

			elite_seqs, elite_stats = self.one_generation(mus, sigmas, init_q_pos, init_q_vel)
			avg_stats.append(np.mean(elite_stats))
			mus, sigmas = self.get_moments(elite_seqs)
			if self.verbose:
				print("Finished gen " + str(gen) + " avg_stats[gen] = " + str(avg_stats[gen])) 

		# Record/store the final outputs
		self.mus = mus
		self.sigmas = sigmas
		self.elite_seqs = elite_seqs
		self.elite_stats = elite_stats
		self.avg_stats = avg_stats

		return elite_seqs[0], elite_stats[0]