from multiprocess.pool import Pool
import multiprocess as mp
import numpy as np
from collections import OrderedDict
import gymnasium as gym
import time

def set_env_states(environments, set_obs):
	obj_qpos = set_obs[54:61]
	obj_qvel = set_obs[48:54]

	mod_envs = []

	for j, env_fn in enumerate(environments.env_fns):
		# get and reset single env
		env = env_fn()
		env.reset()

		# access correct env level
		try:
		  nested_env = env.env.env.env
		except:
		  nested_env = env.env.env

		# reset hand state and object pose
		nested_env._robot_set_obs(set_obs)
		nested_env._object_set_obs(obj_qpos, obj_qvel)
		
		mod_envs.append(env)

	# return new, updated environments to replace the old ones
	environments = gym.vector.AsyncVectorEnv([lambda: mod_env for mod_env in mod_envs])
	return environments

def rollout(environments, set_obs, action_seq, task, lock=None, vec=True):
	"""Evaluate one action sequence across several environments from starting state."""
	# action_seq should have shape=(# actions/env, horizon)
	horizon = action_seq.shape[1]
	# environments.reset()

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

	name = mp.current_process().name

	if lock is not None:
				# print('ACQUIRING LOCK: ', lock, ' PROCESS: ', name)
				lock.acquire()
	# Our method for setting hand/object state
	environments = set_env_states(environments, set_obs)
	if lock is not None:
				# print('RELEASING LOCK: ', lock, ' PROCESS: ', name)
				lock.release()

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

	# Clip costs
	tot_costs = np.clip(tot_costs, horizon * task.cost_low, horizon * task.cost_high)

	return tot_costs, observations

def new_env_rollout(num_envs, set_obs, action_seq, task, lock=None, max_batch=150):
	"""Performs rollouts on newly generated environments."""
	# max_batch limits number of environments that can open at once
	# Hence may have to build up the fill results in iterations
	tot_costs_list = np.zeros(num_envs)
	observations_list = np.zeros((num_envs, len(set_obs), action_seq.shape[1]))

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

		tot_costs, observations = rollout(environments, set_obs, action_seq, task, lock=lock)
		tot_costs_list[count:count+num_create] = tot_costs
		observations_list[count:count+num_create,:,:] = observations

		if lock is not None:
				# print('ACQUIRING LOCK: ', lock, ' PROCESS: ', name)
				lock.acquire()
		environments.close()
		if lock is not None:
				# print('RELEASING LOCK: ', lock, ' PROCESS: ', name)
				lock.release()

		count += num_create

	return tot_costs_list, observations_list

def new_env_eval_plans(num_envs, set_obs, plans, task, processes=1):
	"""Evaluate each plan in plans on randomly drawn environments."""
	# plans should have shape=(# plans, # actions/env, horizon) 
	num_plans = plans.shape[0]
	horizon = plans.shape[2]

	tot_costs = np.zeros((num_plans, num_envs))
	# For each environment and each plan have a cost
	obs_arr = [] # Ultimately shape=(num_plans, num_envs, # obs/env, horizon)

	lock = mp.Manager().Lock()

	if processes > 1:
		# print("ABOUT TO POOL")

		with Pool(processes) as outer_pool:
				res_iter = outer_pool.starmap(new_env_rollout, 
					[(num_envs, set_obs, plans[i,:,:], task, lock) for i in range(plans.shape[0])])
				results = [res for res in res_iter]

		# print("FINISHED POOL")

	else:
		# print("DID NOT POOL")
		results = [new_env_rollout(num_envs, set_obs, plans[i,:,:], task) for i in range(plans.shape[0])]

	for i in range(num_plans):
		tot_costs[i,:], obs = results[i]
		obs_arr.append(obs)
	
	obs_arr = np.array(obs_arr)

	return tot_costs, obs_arr

def eval_plans(environments, set_obs, plans, task):	
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
		tot_costs[i,:], obs_arr[i,:,:] = rollout(environments, set_obs, action_seq, task)

	return tot_costs, obs_arr


def closed_loop_plan(exec_env, solver, horizon, sim_time, task, verbose=True):
	"""Solve for open-loop plans and apply in receding horizon fashion."""
	# get initial observation
	obs_type = "Dict" if isinstance(exec_env.observation_space, gym.spaces.dict.Dict) else "Vec"
	orig_obs, _ = exec_env.reset()

	if obs_type == "Vec":
		set_obs = orig_obs
	elif obs_type == "Dict":
		set_obs = orig_obs["observation"]

	tot_costs = 0

	# observations should have shape (# observations, sim_time)
	if obs_type == "Vec":
		observations = np.zeros((*exec_env.observation_space.shape, sim_time)) # Ant
	elif obs_type == "Dict":
		observations = np.zeros((*exec_env.observation_space['observation'].shape, sim_time)) # Hand

	for i in range(sim_time):
		if verbose:
			print(f'Starting closed-loop simulation time: {i}')

		# Open-loop plan ahead for # steps = horizon
		best_seq, _ = solver.solve(set_obs, horizon)

		# Apply first action to the execution environment
		obs, rewards, termination, truncation, infos = exec_env.step(best_seq[0])
		tot_costs += rewards

		# Record observation and update set_obs variable
		if obs_type == "Vec":
			observations[:,i] = obs
			set_obs = obs
		elif obs_type == "Dict":
			observations[:,i] = obs['observation']
			set_obs = obs["observation"]

	# To go from rewards to costs
	tot_costs *= -1

	return tot_costs, observations