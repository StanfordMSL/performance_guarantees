import numpy as np
from task import Task
from cem import CEM
import bound_utils as bu
import bound_sensitivity as bs
import pdb

def compute_cvar_shifted_delta(nominal_distribution, shifted_distribution, delta_sim, n):
	alpha = bu.find_KS_distance(nominal_distribution, shifted_distribution)
	success = True
	if alpha > np.sqrt(-np.log(delta_sim) / (2*n)) - np.sqrt(np.log(2) / (2*n)):
		delta_true = 1
		success = False		
	else:
		delta_true = bs.get_delta_true_CVaR(n, delta_sim, alpha)
	return delta_true, success

def compute_exp_shifted_delta(nominal_distribution, shifted_distribution, delta_sim, n):
	"""A special case of CVAR shift."""
	return compute_cvar_shifted_delta(nominal_distribution, shifted_distribution, delta_sim, n)

def compute_pr_shifted_delta(nominal_distribution, shifted_distribution, delta_sim, n):
	# Compute the fraction of success for nominal and shifted distributions
	# Assumes J = 0 indicates success, J = 1 indicates failure
	p_sim = np.mean(nominal_distribution == 0)
	p_true = np.mean(shifted_distribution == 0)
	delta_true = bs.get_delta_true_Failure(n, delta_sim, p_sim, p_true)
	return delta_true

def compute_var_shifted_delta(nominal_distribution, shifted_distribution, delta_sim, n, tau):
	"""Given samples from nominal and shifted distribution and nominal var parameters
	find theoretically guaranteed coverage on the new distribution."""
	# Can find the precise tau_prime as the new quantile or bound it using KS distance 
	# tau_prime, _ = bu.find_shifted_quantile(nominal_distribution, shifted_distribution, tau)
	alpha = bu.find_KS_distance(nominal_distribution, shifted_distribution)
	tau_prime = tau + alpha
	delta_true = bs.get_delta_true_VaR(n, delta_sim, tau, tau_prime)
	return delta_true

def shifted_bounds(experiment_name, config):

	# simulation parameters
	task_name = config["simulation"]["task_name"]
	noise = config["simulation"]["noise"]
	num_reps = config["simulation"]["num_reps"]
	noise_low = config["simulation"]["noise_low"]
	noise_high = config["simulation"]["noise_high"]
	num_shifted = config["simulation"]["num_shifted"]

	noise_scales = np.linspace(noise_low, noise_high, num_shifted)

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
	if experiment_name in ["shift_fixed_var", "shift_fixed_pr"]:
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
	if experiment_name == "shift_fixed_exp":	
		# define functions for expected value	
		theory_func = lambda costs : np.mean(costs)
		bound_func = lambda costs : bu.expectation_bound(delta, b_fixed, costs)
		shift_func = lambda shifted_costs : 1-compute_exp_shifted_delta(nominal_costs, shifted_costs, delta, num_envs)[0]
	
	elif experiment_name == "shift_fixed_var":
		tau = config["bounds"]["tau"]
		theory_func = lambda costs : np.quantile(costs, tau)

		k, val, success = bu.k_miscoverage_prob_bin(num_envs, tau, delta)
		padded_tau = k / num_envs

		if not success:
			raise ValueError("Increase sampling budget to satisfy given tau, delta")

		bound_func = lambda costs : np.quantile(costs, padded_tau)
		shift_func = lambda shifted_costs : 1-compute_var_shifted_delta(nominal_costs, shifted_costs, delta, num_envs, tau)

	elif experiment_name == "shift_fixed_cvar":
		tau = config["bounds"]["tau"]
		
		theory_func = lambda costs : bu.MC_CVAR(costs, tau)
		bound_func = lambda costs : bu.CVAR_bound(tau, delta, b_fixed, costs)
		shift_func = lambda shifted_costs : 1-compute_cvar_shifted_delta(nominal_costs, shifted_costs, delta, num_envs)[0]
	
	elif experiment_name == "shift_fixed_pr":
		healthy_low = config["bounds"]["healthy_low"]
		healthy_high = config["bounds"]["healthy_high"]
		# In our case, we require that the ant be "extra" healthy i.e. have torso z value between [0.5, 1]
		# across the horizon (standard definition of healthy is between 0.2, 1)
		constr_func = lambda obs : np.all((obs[2,:] >= healthy_low) * (obs[2,:] <= healthy_high))
		shift_func = lambda shifted_costs : 1-compute_pr_shifted_delta(nominal_costs, shifted_costs, delta, num_envs)

	else:
		print("\nInvalid experiment name!")
		assert(False)

	# run simulations

	if experiment_name == "shift_fixed_pr":
		outputs = bu.analyze_pr_with_shift(num_reps, action_seq, task, noise_scales, init_q_pos, init_q_vel, num_theory, constr_func, num_envs, delta, verbose)			
	else:
		outputs = bu.analyze_bound_with_shift(num_reps, action_seq, task, noise_scales, init_q_pos, 
			init_q_vel, num_theory, theory_func, num_envs, bound_func, verbose)

	shifted_theory_costs, shifted_theory_vals, nominal_costs, nominal_val, fresh_bounds = outputs

	# For each true value, get the empirical coverage by computing fraction of generated bounds lying above the true value
	emp_coverage = np.array([np.mean(fresh_bounds >= true_val) for true_val in shifted_theory_vals])

	# Also, compute the empirical coverage under the nominal conditions
	emp_nominal_coverage = np.mean(fresh_bounds >= nominal_val)

	# Also, compute the theoretically guaranteed coverage under each distribution shift
	theory_coverage = np.array([shift_func(shifted_costs) for shifted_costs in shifted_theory_costs])

	theory_nominal_coverage = shift_func(nominal_costs)

	return shifted_theory_costs, shifted_theory_vals, nominal_costs, nominal_val, fresh_bounds, emp_coverage, theory_coverage, emp_nominal_coverage, theory_nominal_coverage