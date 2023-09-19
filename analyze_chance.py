from task import Task
import numpy as np
import bound_utils as bu
from scipy.stats import binom
from cem import CEM

# bisection search for p_
def inf_p(k, n, delta):
	p_min = 0
	p_max = 1
	while (p_max - p_min > 0.0001):
		p = (p_min + p_max) / 2
		val = binom.cdf(k-1, n, p)
		if val < 1 - delta: # val decreases as p increases
			p_max = p
		else:
			p_min = p
	return p


# get theoretical curve, return xs, ys
def get_curve(n, delta, epsilon, grid_density):
	eps = np.linspace(0.001, 0.999, num=grid_density)
	probs = np.zeros(len(eps))

	for i, eps_prime in enumerate(eps):
		cum_prob = 0
		for k in range(n, 0, -1):
			p = inf_p(k,n,delta)
			if p >= 1-epsilon:
				cum_prob += binom.pmf(k,n,1-eps_prime)
			else:
				probs[i] = cum_prob
				break
	return 1-eps, probs 



def valid_chance_theory(config):
	delta = config["bounds"]["delta"]
	epsilon = config["bounds"]["epsilon"]
	num_envs_range = config["simulation"]["num_envs_range_theory"]
	grid_density = config["bounds"]["grid_density"]

	theory_prob_success = np.zeros((len(num_envs_range), grid_density))
	theory_prob_accept = np.zeros((len(num_envs_range), grid_density))
	for i,n in enumerate(num_envs_range):
		print("\nOn theoretical num_envs: ", n)
		one_minus_eps, probs = get_curve(n, delta, epsilon, grid_density)
		theory_prob_success[i,:] = one_minus_eps
		theory_prob_accept[i,:] = probs

	return theory_prob_success, theory_prob_accept



def valid_chance_empirical(config):
	# simulation parameters
	task_name = config["simulation"]["task_name"]
	noise = config["simulation"]["noise"]
	num_reps = config["simulation"]["num_reps"]
	num_envs_range = config["simulation"]["num_envs_range_empirical"]

	# CEM parameters
	num_samples = config["plans"]["num_samples"]
	horizon = config["plans"]["horizon"]

	# bounds parameters
	num_theory = config["bounds"]["num_theory"]
	healthy_low = config["bounds"]["healthy_low"]
	healthy_high = config["bounds"]["healthy_high"]
	delta = config["bounds"]["delta"]
	epsilon = config["bounds"]["epsilon"]
	tau = 1-epsilon

	# make task
	task = Task(task_name, noise)

	# Get the noiseless initial position
	temp_task = Task(task_name, 0)
	temp_env = temp_task.create_envs(1)
	init_q_pos = temp_env.env_fns[0]().data.qpos # initial position 
	init_q_vel = temp_env.env_fns[0]().data.qvel # initial velocity
	temp_env.close()

	# Set the chance constraint function
	# Should take in a sequence of observations in shape (# observations/step, horizon)
	# In our case, we require that the ant be "extra" healthy i.e. have torso z value between [0.5, 1]
	# across the horizon (standard definition of healthy is between 0.2, 1)
	constr_func = lambda obs : np.all((obs[2,:] >= healthy_low) * (obs[2,:] <= healthy_high))

	# Generate the action sequences that will consider
	plans = task.sample_plans(num_samples, horizon)
		
	min_envs = bu.min_n_chance(delta, tau)
	if min(num_envs_range) < min_envs:
		raise ValueError("Not enough sim environments for certifying chance constraint!")

	# initialize arrays for (x,y) values when plotting: (empirical_prob_success, empirical_prob_accept)
	empirical_prob_success = np.zeros((len(num_envs_range), num_samples))
	empirical_prob_accept = np.zeros((len(num_envs_range), num_samples))

	for i, num_envs in enumerate(num_envs_range):
		print('\n\nOn num_envs ', num_envs)
		for j in range(num_samples):
			print('\nOn plan ' + str(j))

			action_seq = plans[j]
			
			theory_val, theory_holds, accept_stats, accept_frac = bu.analyze_chance_without_control(num_reps, 
				action_seq, task, init_q_pos, init_q_vel, num_theory, num_envs, constr_func, tau, delta, verbose=False)

			print('theory_val', theory_val)
			print('theory_holds', theory_holds)
			print('accept_frac', accept_frac)

			empirical_prob_success[i,j] = theory_val
			empirical_prob_accept[i,j] = accept_frac

	return empirical_prob_success, empirical_prob_accept

def valid_pr_empirical(config):
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

	# bounds parameters
	num_theory = config["bounds"]["num_theory"]
	num_envs = config["bounds"]["num_envs"]
	healthy_low = config["bounds"]["healthy_low"]
	healthy_high = config["bounds"]["healthy_high"]
	delta = config["bounds"]["delta"]
	feas_stat = None
	num_chance = None

	# make task
	task = Task(task_name, noise)

	# Get the noiseless initial position
	temp_task = Task(task_name, 0)
	temp_env = temp_task.create_envs(1)
	init_q_pos = temp_env.env_fns[0]().data.qpos # initial position 
	init_q_vel = temp_env.env_fns[0]().data.qvel # initial velocity
	temp_env.close()

	# Set the chance constraint function
	# Should take in a sequence of observations in shape (# observations/step, horizon)
	# In our case, we require that the ant be "extra" healthy i.e. have torso z value between [0.5, 1]
	# across the horizon (standard definition of healthy is between 0.2, 1)
	constr_func = lambda obs : np.all((obs[2,:] >= healthy_low) * (obs[2,:] <= healthy_high))

	# Solve for the fixed control sequence to use in later analysis
	solver = CEM(max_gen, num_samples, num_elite, 1, task, lambda costs : np.mean(costs), feas_stat, num_chance, rand_envs, processes, False)
	action_seq, _ = solver.solve(init_q_pos, init_q_vel, horizon)
	# action_seq = np.squeeze(task.sample_plans(1, horizon))

	cost_samples, true_pr, bound_samples = bu.analyze_pr_without_control(num_reps, action_seq, task, init_q_pos, init_q_vel, num_theory, num_envs, constr_func, delta, verbose)
	bound_quantile = np.quantile(bound_samples, delta)

	return cost_samples, bound_samples, true_pr, bound_quantile

