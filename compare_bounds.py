from task import Task
from cem import CEM
import numpy as np
import bound_utils as bu

def compare_fixed_bounds(experiment_name, config):

	# simulation parameters
	task_name = config["simulation"]["task_name"]
	noise = config["simulation"]["noise"]
	num_reps = config["simulation"]["num_reps"]

	# solver parameters
	horizon = config["CEM"]["horizon"]
	max_gen = config["CEM"]["max_gen"]
	num_samples = config["CEM"]["num_samples"]
	num_elite = config["CEM"]["num_elite"]
	rand_envs = config["CEM"]["rand_envs"]
	processes = config["CEM"]["processes"]
	verbose = config["CEM"]["verbose"]

	# bound parameters
	num_theory = config["bounds"]["num_theory"]
	num_envs = config["bounds"]["num_envs"]
	delta = config["bounds"]["delta"]
	feas_stat = None
	num_chance = None

	# initialize task
	if task_name == 'Ant-v4':
		cost_low = -2
		cost_high = 0
	elif task_name == 'HalfCheetah-v4':
		cost_low = -0.25
		cost_high = 0.05
	elif task_name == 'Hopper-v4':
		cost_low = -1.28
		cost_high = -1.22
	elif task_name == 'Swimmer-v4':
		cost_low = -0.325
		cost_high = 0.1
	else:
		raise ValueError("\nInvalid task name given in config file!")

	# Bounds on the total cost, derive from bounds on stepwise costs
	a_fixed = horizon * cost_low
	b_fixed = horizon * cost_high
	
	# make task
	if experiment_name == "compare_fixed_var":
		# Don't need to clip cost for VaR bound
		task = Task(task_name, noise)
	else:
		task = Task(task_name, noise, cost_low, cost_high)

	# Get the noiseless initial position
	temp_task = Task(task_name, 0)
	temp_env = temp_task.create_envs(1)
	init_q_pos = temp_env.env_fns[0]().data.qpos # initial position 
	init_q_vel = temp_env.env_fns[0]().data.qvel # initial velocity
	temp_env.close()

	# Solve for the fixed control sequence to use in later analysis
	solver = CEM(max_gen, num_samples, num_elite, 1, task, lambda costs : np.mean(costs), feas_stat, num_chance, rand_envs, processes, False)
	action_seq, _ = solver.solve(init_q_pos, init_q_vel, horizon)

	# compute bound_specific functions
	if experiment_name == "compare_fixed_exp":	
		# define functions for expected value	
		theory_func = lambda costs : np.mean(costs)
		bound_func = lambda costs : bu.expectation_bound(delta, b_fixed, costs)
		ref_bound_func = lambda costs : bu.akella_exp(delta, b_fixed, costs)
		bound_func_list = [bound_func, ref_bound_func]
	
	elif experiment_name == "compare_fixed_var":
		tau = config["bounds"]["tau"]
		theory_func = lambda costs : np.quantile(costs, tau)

		k, val, success = bu.k_miscoverage_prob_bin(num_envs, tau, delta)
		padded_tau = k / num_envs

		if not success:
			raise ValueError("Increase sampling budget to satisfy given tau, delta")

		k_kolla, val, success = bu.k_miscoverage_prob_kolla(num_envs, tau, delta)
		kolla_tau = k_kolla / num_envs

		if not success:
			raise ValueError("Increase sampling budget to satisfy given tau, delta")

		k_szorenyi, val, success = bu.k_miscoverage_prob_szorenyi(num_envs, tau, delta)
		szorenyi_tau = k_szorenyi / num_envs

		if not success:
			raise ValueError("Increase sampling budget to satisfy given tau, delta")

		bound_func = lambda costs : np.quantile(costs, padded_tau)
		ref_bound_func_1 = lambda costs : np.quantile(costs, kolla_tau)
		ref_bound_func_2 = lambda costs : np.quantile(costs, szorenyi_tau)
		bound_func_list = [bound_func, ref_bound_func_1, ref_bound_func_2]

	elif experiment_name == "compare_fixed_cvar":
		tau = config["bounds"]["tau"]
		
		theory_func = lambda costs : bu.MC_CVAR(costs, tau)
		bound_func = lambda costs : bu.CVAR_bound(tau, delta, b_fixed, costs)
		ref_bound_func = lambda costs : bu.akella_cvar(tau, delta, b_fixed, costs)
		bound_func_list = [bound_func, ref_bound_func]
	
	else:
		print("\nInvalid experiment name!")
		assert(False)

	# run simulations	
	cost_samples, true_val, bound_samples = bu.analyze_multiple_bounds_without_control(num_reps, action_seq, 
	task, init_q_pos, init_q_vel, num_theory, theory_func, num_envs, bound_func_list, verbose)
	bound_quantiles = np.quantile(bound_samples, delta, axis=1)

	return cost_samples, true_val, bound_samples, bound_quantiles