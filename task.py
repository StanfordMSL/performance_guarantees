import gymnasium as gym
import numpy as np
import multiprocessing as mp
import random
import numpy as np
from traj_utils import get_env_friction
import os
import glfw

# os.environ["PYGLFW_LIBRARY"] = "venv/lib/python3.8/site-packages/glfw/x11/libglfw.so"
# os.environ["PYGLFW_LIBRARY_VARIANT"]= "x11"


# PYGLFW_LIBRARY = "venv/lib/python3.8/site-packages/glfw/x11/libglfw.so"

class Task():
	def __init__(self, task_name, noise=0, cost_low=-np.inf, cost_high=np.inf, **kwargs):
		self.task_name = task_name
		self.noise = noise # Noise about the initial state
		self.kwargs = kwargs
		temp = self.create_envs(1, **self.kwargs)
		self.action_space = temp.env_fns[0]().action_space
		temp.close()
		self.cost_low = cost_low
		self.cost_high = cost_high

	def sample_plans(self, num_samples, horizon):
		"""Sample plans fron env action space uniformly at random."""
		# Sample sequences to use for planning
		low = np.array(self.action_space.low)
		high = np.array(self.action_space.high)
		plans = np.stack([np.random.uniform(low=low, high=high, 
			size=(num_samples, *self.action_space.shape)) for i in range(horizon)], axis=-1)
		return plans

	# Note: Make sure to close these envs after use
	def create_envs(self, num_envs, render_mode=None, max_episode_steps=1000, **kwargs):
		"""Creates new environments in desired render mode."""
		# Allow you to pass in your own modified kwargs
		if not len(kwargs):
			kwargs = self.kwargs

		if self.task_name == 'HandManipulateBlockEggPen_BooleanTouchSensors-v1':
			environments = gym.vector.make(self.task_name, num_envs, asynchronous=True, render_mode=render_mode, max_episode_steps=max_episode_steps, **kwargs)

		elif 'Hand' in self.task_name:
			environments = gym.vector.make(self.task_name, num_envs=num_envs, asynchronous=True, render_mode=render_mode, max_episode_steps=max_episode_steps)
		
		elif self.task_name in ['Ant-v4', 'Humanoid-v4', 'Hopper-v4']:
			environments = gym.vector.make(self.task_name, num_envs=num_envs, reset_noise_scale=self.noise, 
					exclude_current_positions_from_observation=False, asynchronous=True, terminate_when_unhealthy=False, render_mode=render_mode, max_episode_steps=max_episode_steps)
		
		elif self.task_name in ['HalfCheetah-v4', 'Swimmer-v4']:
			environments = gym.vector.make(self.task_name, num_envs=num_envs, reset_noise_scale=self.noise, 
					exclude_current_positions_from_observation=False, asynchronous=True, render_mode=render_mode, max_episode_steps=max_episode_steps)

		return environments

	def create_one_env(self, render_mode=None, max_episode_steps=1000, **kwargs):
		"""Creates one non-vectorized environment in desired render mode."""
		# Allow you to pass in your own modified kwargs
		if not len(kwargs):
			kwargs = self.kwargs

		if self.task_name == 'HandManipulateBlockEggPen_BooleanTouchSensors-v1':
			environments = gym.make(self.task_name, render_mode=render_mode, max_episode_steps=max_episode_steps, **kwargs)

		elif 'Hand' in self.task_name:
			environments = gym.make(self.task_name, render_mode=render_mode, max_episode_steps=max_episode_steps)
		
		elif self.task_name in ['Ant-v4', 'Humanoid-v4', 'Hopper-v4']:
			environments = gym.make(self.task_name, reset_noise_scale=self.noise, 
					exclude_current_positions_from_observation=False, terminate_when_unhealthy=False, render_mode=render_mode, max_episode_steps=max_episode_steps)
		
		elif self.task_name in ['HalfCheetah-v4', 'Swimmer-v4']:
			environments = gym.make(self.task_name, reset_noise_scale=self.noise, 
					exclude_current_positions_from_observation=False, render_mode=render_mode, max_episode_steps=max_episode_steps)

		return environments


if __name__ == '__main__':
	# Demonstrate how to initialize a task
	# task = Task('Ant-v4', noise=3)
	density_bounds = np.array([600, 1100])
	friction_bounds = np.array([[0.9, 2], [0.004, 0.008], [0.00009, 0.00019]])
	task = Task('HandManipulateBlockEggPen_BooleanTouchSensors-v1', noise=0, 
		class_probs=[0.5, 0.5, 0.0], density_bounds = density_bounds, friction_bounds=friction_bounds)

	# make environment
	print("After Init")
	environment = task.create_one_env(render_mode="human")
	environment.reset()
	environment.render()

	horizon = 1000
	plan = np.squeeze(task.sample_plans(1, horizon))

	for i in range(horizon):
		actions = plan[:,i]
		obs, reward, termination, truncation, infos = environment.step(actions)

	environment.close()