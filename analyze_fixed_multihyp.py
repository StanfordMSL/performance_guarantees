from task import Task
import traj_utils as trj
import numpy as np
import bound_utils as bu

def valid_fixed_multi_hyp(config, file_name):
	# this function is currently written only for VaR and for hand

    plans_dict = np.load(file_name)
    plans = plans_dict["plans"]
    num_plans = len(plans)

	# simulation parameters
    density_bounds = np.array(config["simulation"]["density_bounds"])
    friction_bounds = np.array(config["simulation"]["friction_bounds"])
    num_reps = config["simulation"]["num_reps"]
    processes = config["simulation"]["processes"]
    verbose = config["simulation"]["verbose"]

	# bound parameters
    delta = config["bounds"]["delta"]
    epsilon = config["bounds"]["epsilon"]
    tau = 1 - epsilon
    num_envs = config["bounds"]["num_envs"]
    num_theory = config["bounds"]["num_theory"]

    # cost bounds for task
    task_name = 'HandManipulateBlockEggPen_BooleanTouchSensors-v1'

    # Just egg for now
    task = Task(task_name, class_probs=[0, 1, 0], 
                    density_bounds=density_bounds, 
                    friction_bounds=friction_bounds)

	# Get the noiseless initial position
    temp_env = task.create_envs(1)
    init_q_pos = temp_env.env_fns[0]().data.qpos # initial position 
    init_q_vel = temp_env.env_fns[0]().data.qvel # initial velocity
    temp_env.close()

	# Account for multi-hypothesis
    delta_bar = bu.multi_hyp_correct(delta, num_plans)
    k, val, success = bu.k_miscoverage_prob_bin(num_envs, tau, delta_bar)
    ver_func = lambda costs : np.quantile(costs, k / num_envs)
    print('k verified: ', k)

    if not success:
        raise ValueError("Not enough samples to apply multi-hypothesis correction!")

	# naively not accounting for multi-hypothesis
    k_unver, val, success = bu.k_miscoverage_prob_bin(num_envs, tau, delta)
    unver_func = lambda costs : np.quantile(costs, k_unver / num_envs)
    print('k unverified: ', k_unver)

	# theoretical VaR
    theory_func = lambda costs : np.quantile(costs, tau)

    return analyze_fixed_correction(num_reps, task, plans, init_q_pos, 
                                    init_q_vel, unver_func, ver_func, 
                                    num_envs, num_theory, theory_func, 
                                    verbose=verbose, processes=processes)

def analyze_fixed_correction(num_reps, task, plans, init_q_pos, init_q_vel, 
                             unver_func, ver_func, num_val, num_theory, 
                             theory_func, verbose, processes=1):
    """Repeatedly generate candidate bounds for the given quantity to see if 
    holds, given several fixed control sequences."""
    unver_vals = np.zeros(num_reps)
    unver_theory_vals = np.zeros(num_reps)
    unver_chosen_plans = np.zeros(num_reps)
    ver_vals = np.zeros(num_reps)
    ver_theory_vals = np.zeros(num_reps)
    ver_chosen_plans = np.zeros(num_reps)

    num_plans = len(plans)

    # Since fixed plans, compute the theoretical value only once for each plan    
    theory_costs, _ = trj.new_env_eval_plans(num_theory, init_q_pos,
                                       init_q_vel, plans, task, processes)
    theory_vals = np.array([theory_func(theory_costs[i,:]) for i in 
                            range(num_plans)])
    
    if verbose:
        print('Finished theoretical evaluation')

    for rep in range(num_reps):
        
        if verbose:
            print('Starting rep ' + str(rep))

        # Compute new costs for each elite sample
        val_costs, _ = trj.new_env_eval_plans(num_val, init_q_pos, init_q_vel, 
                                        plans, task, processes)
		# val_costs shape (num_plans, num_val)

		# Evaluate each plan according to either the unverified or 
        # verified statistic
        unver_stats = np.array([unver_func(val_costs[i,:]) for i in range(len(val_costs))])
        ver_stats = np.array([ver_func(val_costs[i,:]) for i in range(len(val_costs))])

		# Choose the best plan according to each statistic
        unver_ind = np.argmin(unver_stats)
        ver_ind = np.argmin(ver_stats)
        unver_chosen_plans[rep] = unver_ind
        ver_chosen_plans[rep] = ver_ind

		# Retrieve the corresponding bound
        unver_vals[rep] = unver_stats[unver_ind]
        ver_vals[rep] = ver_stats[ver_ind]

        if verbose:
            print('Finished planning')

        # Retrieve the corresponding theoretical value for each chosen plan
        unver_theory_vals[rep] = theory_vals[unver_ind]
        ver_theory_vals[rep] = theory_vals[ver_ind]

    return unver_vals, unver_theory_vals, unver_chosen_plans, ver_vals, \
        ver_theory_vals, ver_chosen_plans