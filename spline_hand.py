import numpy as np
from traj_utils_hand import eval_plans, new_env_eval_plans
from scipy.interpolate import CubicSpline

class Spline():
	"""
	Sample-based control method from MJPC paper
	Currently only for hand because of dependence on traj_utils_hand, but should be easy to generalize
	"""
	def __init__(self, task, noise_std, deterministic, num_envs, cost_stat, feas_stat=None, processes=1):
		# Task object for which are planning
		self.task = task
		self.noise_std = noise_std
		# boolean of whether the task is deterministic (1) or stochastic (2)
		self.deterministic = deterministic
		# how many environments to evaluate each plan on
		self.num_envs = num_envs
		# function  which assigns a scalar cost for a plan given multiple observed rollouts of plan
		# input to cost_stat function is observations of shape: (self.num_risk, # observations, horizon)
		self.cost_stat = cost_stat
		# function which evaluates feasibility of a sequence of observations
		# input to feas_stat function is observations of shape: (self.num_chance, # observations, horizon)
		self.feas_stat = feas_stat
		# Number of processes to use, 1 if no multiprocess
		self.processes = processes

	def get_splines(self, action_dim, num_spline, num_rollouts):
		# use cubic spline to interpolate noise trajectories
		# splines shape = (num_rollouts, action_dim)
		x = np.linspace(0, 1, num=num_spline) # assume horizon time is 1 wlog
		splines = []
		for _ in range(num_rollouts):
			splines.append([CubicSpline(x, np.random.normal(0, self.noise_std, size=num_spline)) for _ in range(action_dim)])

		return splines

	def get_candidate_plans(self, init_plan, splines, action_dim, horizon, num_rollouts):
		# evaluate the splines at the control times and add to init_plan
		t = np.linspace(0, 1, num=horizon)
		candidate_plans = np.zeros((num_rollouts, action_dim, horizon))
		for rollout in range(num_rollouts):
			for action in range(action_dim):
				candidate_plans[rollout,action,:] = np.clip(init_plan[action] + splines[rollout][action](t), -1,1)

		return candidate_plans

	def evaluate_plans(self, set_obs, candidate_plans, num_envs):
		# determine which plan is best
		# plans.shape = (num_rollouts, action_dim, horizon) 
		# tot_costs.shape = (num_rollouts, self.num_envs)
		# obs_arr.shape = (num_rollouts, self.num_envs, # observations, horizon)
		if self.deterministic:
			environment = self.task.create_envs(1) 
			tot_costs, obs_arr = eval_plans(environment, set_obs, candidate_plans, self.task)
		else:
			tot_costs, obs_arr = new_env_eval_plans(num_envs, set_obs, candidate_plans, self.task, processes=self.processes)

		return tot_costs, obs_arr

	def get_feasibility(self, obs_arr, feas_stat):
		# get boolean array for whether each rollout is feasible or not
		num_rollouts = obs_arr.shape[0]
		if self.feas_stat == None:
			feasibility = np.full(num_rollouts, True)
		else:
			feasibility = [feas_stat(obs_arr[i,:,:,:]) for i in range(num_rollouts)]

		return feasibility

	def get_best_plan(self, candidate_plans, feasibility, tot_costs, cost_stat):
		# tot_costs.shape = (num_rollouts, self.num_envs)
		# get single cost for each rollout
		plan_costs = cost_stat(tot_costs)

		# return feasible plan with lowest cost, else infeasible plan with lowest cost
		if np.any(feasibility):
			inf_arr = np.full(len(feasibility), np.Inf)
			plan_costs = np.where(feasibility, plan_costs, inf_arr)

		best_idx = np.argmin(plan_costs)
		best_plan = candidate_plans[best_idx,:,:]

		return best_plan

	def solve(self, set_obs, init_plan, num_spline, num_rollouts):
		#Return the best plan
		# init_plan.shape = (action_dim, horizon)
		action_dim, horizon = init_plan.shape
		
		splines = self.get_splines(action_dim, num_spline, num_rollouts)

		candidate_plans = self.get_candidate_plans(init_plan, splines, action_dim, horizon, num_rollouts)
		
		tot_costs, obs_arr = self.evaluate_plans(set_obs, candidate_plans, self.num_envs)
		feasibility = self.get_feasibility(obs_arr, self.feas_stat)

		best_plan = self.get_best_plan(candidate_plans, feasibility, tot_costs, self.cost_stat)

		return best_plan