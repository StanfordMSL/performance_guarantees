import numpy as np
from task import Task
from cem import CEM
import bound_utils as bu
import pdb

def robust_bounds(experiment_name, config):

	# simulation parameters
	task_name = config["simulation"]["task_name"]
	noise = config["simulation"]["noise"]
	num_reps = config["simulation"]["num_reps"]
	noise_low = config["simulation"]["noise_low"]
	noise_high = config["simulation"]["noise_high"]
	num_shifted = config["simulation"]["num_shifted"]
	alphas = config["simulation"]["alphas"]

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
	if experiment_name in ["robust_fixed_var", "robust_fixed_pr"]:
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
	if experiment_name == "robust_fixed_exp":	
		# define functions for expected value	
		theory_func = lambda costs : np.mean(costs)
		bound_func = lambda costs, alpha : bu.robust_expectation_bound(delta, b_fixed, costs, alpha)

	elif experiment_name == "robust_fixed_var":
		tau = config["bounds"]["tau"]
		theory_func = lambda costs : np.quantile(costs, tau)
		bound_func = lambda costs, alpha : bu.robust_var_bound(tau, delta, costs, alpha)

	elif experiment_name == "robust_fixed_cvar":
		tau = config["bounds"]["tau"]
		
		theory_func = lambda costs : bu.MC_CVAR(costs, tau)
		bound_func = lambda costs, alpha : bu.robust_CVAR_bound(tau, delta, b_fixed, costs, alpha)
	
	elif experiment_name == "robust_fixed_pr":
		healthy_low = config["bounds"]["healthy_low"]
		healthy_high = config["bounds"]["healthy_high"]
		# In our case, we require that the ant be "extra" healthy i.e. have torso z value between [0.5, 1]
		# across the horizon (standard definition of healthy is between 0.2, 1)
		constr_func = lambda obs : np.all((obs[2,:] >= healthy_low) * (obs[2,:] <= healthy_high))
        
	else:
		print("\nInvalid experiment name!")
		assert(False)

	# run simulations

	if experiment_name == "robust_fixed_pr":
		outputs = bu.analyze_robust_pr(num_reps, action_seq, task, noise_scales, init_q_pos, init_q_vel, num_theory, constr_func, num_envs, delta, alphas, verbose)			
	else:
		outputs = bu.analyze_robust_bound(num_reps, action_seq, task, noise_scales, init_q_pos, 
			init_q_vel, num_theory, theory_func, num_envs, bound_func, alphas, verbose)

	shifted_theory_costs, shifted_theory_vals, nominal_costs, nominal_val, fresh_bounds, ks_dists, alphas = outputs

	# For each true value, get the empirical coverage by computing fraction of generated bounds lying above the true value
	# fresh_bounds is num_alpha, num_reps so take mean along axis=1
	emp_coverage = np.array([np.mean(fresh_bounds >= true_val, axis=1) for true_val in shifted_theory_vals]) # shape num_shifted, num_alpha

	# Also, compute the empirical coverage under the nominal conditions
	emp_nominal_coverage = np.mean(fresh_bounds >= nominal_val, axis=1) # shape num_alpha

	return shifted_theory_costs, shifted_theory_vals, nominal_costs, nominal_val, fresh_bounds, ks_dists, alphas, emp_coverage, emp_nominal_coverage