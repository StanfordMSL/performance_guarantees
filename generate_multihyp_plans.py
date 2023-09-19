from task import Task
import numpy as np
import os
import yaml
import time
from datetime import datetime
from spline_hand import Spline
from gymnasium.utils.save_video import save_video

# define cost functions
def cost_stat_deterministic(tot_costs):
    # tot_costs.shape = (num_rollouts, self.num_envs)
    return tot_costs[:,0]

# This function is currently written so that plans are generated each optimizing with a different assumed density and friction value
# However, in our experiments we actually first temporarily changed the config to make the density_bounds and friction_bounds have
# lower and upper bounds equal to the same single value to consider the case where all plans are optimizing when assuming the
# same nominal parameters.
def generate_cl_plans(config, vid_folder=''):
    density_bounds = np.array(config["simulation"]["density_bounds"])
    friction_bounds = np.array(config["simulation"]["friction_bounds"])
    num_plans = config["plans"]["num_plans"]
    verbose = config["simulation"]["verbose"]

    # Interpolate between the density and friction bounds
    densities = np.linspace(density_bounds[0], density_bounds[1], num_plans)
    frictions = np.linspace(friction_bounds[:,0], friction_bounds[:,1], num_plans)

    plans = []

    for i in range(num_plans):
        if verbose:
            print(f'Generating plan {i}')

        density_det = densities[i]
        friction_det = frictions[i]

        if vid_folder:
            vid_name = os.path.join(vid_folder, 'plan_'+str(i))
        else:
            vid_name = ''

        plan = generate_one_cl_plan(density_det, friction_det, config, vid_name)

        plans.append(plan)

    return densities, frictions, plans

def generate_one_cl_plan(density_det, friction_det, config, vid_name=''):
    noise_std = config["solver"]["noise_std"]
    num_spline = config["solver"]["num_spline"]
    num_rollouts = config["solver"]["num_rollouts"]
    horizon = config["solver"]["horizon"]
    T = config["plans"]["T"]
    verbose = config["simulation"]["verbose"]
    processes = config["solver"]["num_processes"]

    dens_bounds_det = np.array([density_det, density_det])
    fric_bounds_det = np.array([[friction_det[0],friction_det[0]], [friction_det[1],friction_det[1]], [friction_det[2],friction_det[2]]])
    det_task = Task('HandManipulateBlockEggPen_BooleanTouchSensors-v1', class_probs=[0, 1, 0], density_bounds=dens_bounds_det, friction_bounds=fric_bounds_det)
    solver = Spline(det_task, noise_std, True, 1, cost_stat_deterministic, feas_stat=None, processes=processes)

    # setup environment
    env = det_task.create_one_env(render_mode="rgb_array_list") 
    obs, _ = env.reset()
    obs = obs['observation']
    action_dim = env.action_space.shape[0]

    # data structures for seed plan and executed plan
    seed_plan = np.zeros((action_dim,horizon))
    executed_plan = np.zeros((action_dim,T))

    # Simulate
    if verbose:
        print("\nBeginning Simulation")
        print('density_det', density_det)
        print('friction_det', friction_det)
    
    t0 = time.time()

    for t in range(T):
        if verbose:
            print("t = ", t)
        action_seq = solver.solve(obs, seed_plan, num_spline, num_rollouts)
        executed_plan[:,t] = action_seq[:,0]
        obs, rewards, termination, truncation, infos = env.step(action_seq[:,0])
        obs = obs['observation']
        seed_plan = np.hstack((action_seq[:,1:], np.reshape(action_seq[:,-1], (action_dim,1))))

    if verbose:
        print('Time elapsed', time.time() - t0)

    # Save video
    if vid_name:
        save_video(env.render(), vid_name, fps=env.metadata["render_fps"])
    env.close()

    return executed_plan

def generate_random_plans(config):
    density_bounds = np.array(config["simulation"]["density_bounds"])
    friction_bounds = np.array(config["simulation"]["friction_bounds"])
    num_plans = config["plans"]["num_plans"] 
    T = config["plans"]["T"]

    task_name = 'HandManipulateBlockEggPen_BooleanTouchSensors-v1'

    # Just egg for now
    task = Task(task_name, class_probs=[0, 1, 0], 
                    density_bounds=density_bounds, 
                    friction_bounds=friction_bounds)
    
    plans = task.sample_plans(num_plans, T)
    
    return plans

if __name__ == '__main__':
    ## Load the configuration file
    config_file = "experiments/config_files/valid_fixed_multi_hyp.yaml"
    with open(config_file, "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        print("\nloaded config file")
    print(config)

    current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    str_current_datetime = str(current_datetime)

    folder_name = 'experiments/config_files/valid_fixed_multi_hyp_plans/'
    file_name =  os.path.join(folder_name, str_current_datetime + '.npz')

    ## Generate and save the plans
    if config["plans"]["random"]:
    	plans = generate_random_plans(config)
    	np.savez(file_name, plans=plans)
    else:
    	densities, frictions, plans = generate_cl_plans(config, vid_folder=folder_name)
    	np.savez(file_name, densities=densities, frictions=frictions, plans=plans)