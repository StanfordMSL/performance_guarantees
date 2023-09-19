from multiprocess.pool import Pool
import multiprocess as mp
import numpy as np
from collections import OrderedDict
import gymnasium as gym

def get_env_friction(environments, name='object'):
	frictions = []
	for i, env_fn in enumerate(environments.env_fns):
		env = env_fn()
		frictions.append(env.model.geom(name).friction)

	return frictions

# TODO: Does not work but may not be needed
def set_env_states(environments, q_pos, q_vel):
	"""Set all environments to a desired state."""
	for env_fn in environments.env_fns:
		env = env_fn()
		env.data.qpos = q_pos
		env.data.qvel = q_vel

def rollout(environments, q_pos, q_vel, action_seq, task, vec=True):
	"""Evaluate one action sequence across several environments from starting state."""
	# action_seq should have shape=(# actions/env, horizon)
	horizon = action_seq.shape[1]
	environments.reset()

	# Add this to be able to handle the case when environments is actually a single, non-vector environment
	if vec:
		num_envs = environments.num_envs
	else:
		num_envs = 1
	
	tot_costs = np.zeros(num_envs)
	
	obs_type = "Dict" if isinstance(environments.observation_space, gym.spaces.dict.Dict) else "Vec"

	# observations should have shape (# environments, # observations, horizon)
	if obs_type == "Vec":
		observations = np.zeros((*environments.observation_space.shape, action_seq.shape[1])) # Ant
	elif obs_type == "Dict":
		observations = np.zeros((*environments.observation_space['observation'].shape, action_seq.shape[1])) # Hand

	if not vec:
		observations = np.expand_dims(observations, axis=0) # Add first dim num environments = 1 

	# TODO: Does not work, hence q_pos, q_vel unused for now
	# set_env_states(environments, q_pos, q_vel)

	# simulate planning environments
	for i in range(horizon):
		action = action_seq[:,i] # actions/env
		
		if vec:
			actions = np.repeat(action[np.newaxis, :], repeats=num_envs, axis=0)
		else:
			actions = action

		obs, rewards, termination, truncation, infos = environments.step(actions)
		tot_costs += rewards

		if obs_type == "Vec":
			observations[:,:,i] = obs
		elif obs_type == "Dict":
			observations[:,:,i] = obs['observation']

	# To go from rewards to costs
	tot_costs *= -1

	a = horizon * task.cost_low
	b = horizon * task.cost_high

	# Clip costs
	tot_costs = np.clip(tot_costs, a, b)

	# print('a', a)
	# print('b', b)
	# print('min, max tot_costs', np.min(tot_costs), np.max(tot_costs))
	# print('tot_costs.shape', tot_costs.shape)

	return tot_costs, observations

def new_env_rollout(num_envs, q_pos, q_vel, action_seq, task, lock=None, max_batch=150):
	"""Performs rollouts on newly generated environments."""
	# max_batch limits number of environments that can open at once
	# Hence may have to build up the fill results in iterations
	tot_costs_list = []
	observations_list = []

	name = mp.current_process().name

	count = 0
	while count < num_envs:
		num_create = min(num_envs - count, max_batch)

		if lock is not None:
				# print('ACQUIRING LOCK: ', lock, ' PROCESS: ', name)
				lock.acquire()
		environments = task.create_envs(num_create)
		environments.reset()
		if lock is not None:
				# print('RELEASING LOCK: ', lock, ' PROCESS: ', name)
				lock.release()

		tot_costs, observations = rollout(environments, q_pos, q_vel, action_seq, task)
		tot_costs_list.append(tot_costs)
		observations_list.append(observations)

		if lock is not None:
				# print('ACQUIRING LOCK: ', lock, ' PROCESS: ', name)
				lock.acquire()
		environments.close()
		if lock is not None:
				# print('RELEASING LOCK: ', lock, ' PROCESS: ', name)
				lock.release()

		count += num_create

	tot_costs = np.concatenate(tot_costs_list, axis=0)
	observations = np.concatenate(observations_list, axis=0)

	return tot_costs, observations

def new_env_eval_plans(num_envs, q_pos, q_vel, plans, task, processes=1):
	"""Evaluate each plan in plans on randomly drawn environments."""
	# plans should have shape=(# action_seqs, # actions/env, horizon) 
	num_plans = plans.shape[0]
	horizon = plans.shape[2]

	tot_costs = np.zeros((num_plans, num_envs))
	# For each environment and each plan have a cost
	obs_arr = [] # Ultimately shape=(num_plans, num_envs, # obs/env, horizon)

	lock = mp.Manager().Lock()

	if processes > 1:
		with Pool(processes) as outer_pool:
				res_iter = outer_pool.starmap(new_env_rollout, 
					[(num_envs, q_pos, q_vel, plans[i,:,:], task, lock) for i in range(plans.shape[0])])
				results = [res for res in res_iter]
	else:
		results = [new_env_rollout(num_envs, q_pos, q_vel, plans[i,:,:], task) for i in range(plans.shape[0])]

	for i in range(num_plans):
		tot_costs[i,:], obs = results[i]
		obs_arr.append(obs)
	
	obs_arr = np.array(obs_arr)

	return tot_costs, obs_arr

def eval_plans(environments, q_pos, q_vel, plans, task):	
	"""Evaluate each plan in plans on each of given environments."""
	# plans should have shape=(# action_seqs, # actions/env, horizon) 
	environments.reset()

	num_plans = plans.shape[0]
	horizon = plans.shape[2]

	tot_costs = np.zeros((num_plans, environments.num_envs))
	# For each environment and each plan have a cost
	# Compute a statistic across environments to evaluate each plan
	stats = np.zeros(num_plans)

	obs_type = "Dict" if isinstance(environments.observation_space, gym.spaces.dict.Dict) else "Vec"

	if obs_type == "Vec":
		obs_arr = np.zeros((num_plans, *environments.observation_space.shape, horizon)) # Ant
	elif obs_type == "Dict":
		obs_arr = np.zeros((num_plans, *environments.observation_space['observation'].shape, horizon)) # Hand
	
	for i in range(num_plans):
		action_seq = plans[i,:,:]
		tot_costs[i,:], obs_arr[i,:,:] = rollout(environments, q_pos, q_vel, action_seq, task)

	return tot_costs, obs_arr