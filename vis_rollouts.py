from task import Task
import numpy as np
import os
import yaml
import time
from gymnasium.utils.save_video import save_video
import cv2

def generate_one_rollout_video(task, plan, vid_name, verbose=False):
	t0 = time.time()

	env = task.create_one_env(render_mode="rgb_array_list") 
	obs, _ = env.reset()
	obs = obs['observation']
	action_dim = env.action_space.shape[0]

	T = plan.shape[1]

	for t in range(T):
	    if verbose:
	        print("t = ", t)
	    obs, rewards, termination, truncation, infos = env.step(plan[:,t])
	    obs = obs['observation']

	if verbose:
	    print('Time elapsed', time.time() - t0)

	# Save video
	save_video(env.render(), vid_name, fps=env.metadata["render_fps"])
	env.close()

def generate_rollout_videos(task, plans, num_envs, folder_name, verbose=True):

	for i, plan in enumerate(plans):
		for j in range(num_envs):
			
			if verbose:
				print(f'On plan {i}, rollout {j}')
			
			vid_folder = os.path.join(folder_name, f'plan_{i}', f'rollout_{j}')
			generate_one_rollout_video(task, plan, vid_folder, verbose)

			os.makedirs(os.path.join(vid_folder, 'frames'), exist_ok=True)
			split_one_vid(vid_folder, verbose)

def split_one_vid(vid_folder, verbose=False):
	vid_name = os.path.join(vid_folder, 'rl-video-episode-0.mp4')
	vidcap = cv2.VideoCapture(vid_name)
	success,image = vidcap.read()
	count = 0
	while success:
	  cv2.imwrite(os.path.join(vid_folder, 'frames', "frame%d.jpg" % count), image)     # save frame as JPEG file      
	  success,image = vidcap.read()
	  if verbose:
	  	print('Read a new frame: ', success)
	  count += 1

if __name__ == "__main__":
	plans_folder_name = 'experiments/config_files/valid_fixed_multi_hyp_plans/'
	plans_file_name =  plans_folder_name + '20230817-224846.npz'
	config_file = 'experiments/config_files/valid_fixed_multi_hyp.yaml'

	# 1. Load in the plans
	plans_dict = np.load(plans_file_name)
	plans = plans_dict["plans"]
	num_plans = len(plans)

	# 2. Load in the config
	with open(config_file, "r") as yamlfile:
		config = yaml.load(yamlfile, Loader=yaml.FullLoader)
		print("\nloaded config file")

	density_bounds = np.array(config["simulation"]["density_bounds"])
	friction_bounds = np.array(config["simulation"]["friction_bounds"])
	num_envs = config["bounds"]["num_envs"]

	# Just egg for now
	task_name = 'HandManipulateBlockEggPen_BooleanTouchSensors-v1'
	task = Task(task_name, class_probs=[0, 1, 0], 
					density_bounds=density_bounds, 
					friction_bounds=friction_bounds)

	# 3. For each plan, sample num_envs and rollout
	generate_rollout_videos(task, plans, num_envs, plans_folder_name, verbose=True)

