# risk_sensitive_project
This repository accompanies the paper "Distribution-Free Risk Sensitive Control of Robotic Systems".

## Summary of Repository
 - ```experiments.py```: Each figure in the paper is associated with an experiment. We use this file to run each experiment and save the data necessary for plotting the results. Each experiment requires a ```.yaml``` config file in order to run. A config file for each experiment is found in the ```experiments/config_files``` directory. When an experiment is run the results as well as the associated config file are saved in a timestamped folder in the ```experiments/runs``` directory. 

 -  ```plot_experiments.py```: This file is used to plot the results of each experiment to generate the figures in the paper.

-  ```task.py```: This file defines the ```Task``` class which is a wrapper class providing basic simulation functionality such as environment creation and random action sequence sampling.

 - ```traj_utils.py```: This file provides functions for (i) rolling out a trajectory, (ii) evaluating multiple plans in parallel on a Gymnasium environment.

  - ```traj_utils_hand.py```: This file is similar to traj_utils but with functions specifically designed for use with shadow hand environment. Gymnasium handles the shadow hand environment slightly differently than it does the other MuJoCo environments. Ideally this will be merged with ```traj_utils.py```.

 - ```bound_utils.py```: This file contains functions for bounding various measures of risk, chance constraint hypothesis testing, and baseline code for assessing and visualizing bound validity.

 - ```analyze_bounds.py```: This file provides the function for the experiment that assesses validity of the risk bounds when considering a single plan.

 - ```analyze_chance.py```: This file provides a function for the experiment that assesses validity of the chance constraint hypothesis testing framework.

 - ```analyze_fixed_multihyp.py```: This file provides a function for the experiment that assesses validity and necessity of the multi-hypothesis correction given a set of candidate plans. 

 - ```cem.py```: This file defines the ```CEM``` class which implements the cross entropy sampling-based solver.

 - ```spline_hand.py```: This file defines the ```Spline``` class which implements the spline sampling-based solver from MJPC for the shadow hand.

 - ```generate_multihyp_plans.py```: This file is used for generating plans for the shadow hand using the spline solver. These plans are then used for testing the multi-hypothesis corrected bound.

 - ```vis_rollouts.py```: This file is used to visualize rollouts of the plans generated for the shadow hand.

 - ```dist_cartoon.py```: Used to generate a visualization of the different risk measures on an example distribution.

## Our Changes
In this section we detail the changes we made to either the ```gymnasium``` or the ```gymnasium_robotics``` python libraries. Most changes were made in order to enable (i) domain randomization or (ii) multi-processing (not just multi-threading).

In ```/gymnasium_robotics/envs/shadow_dexterous_hand/manipulate.py``` we made the following changes:
1. in ```__init__()``` for ```BaseManipulateEnv``` class, we set the following variables
	```
	randomize_initial_position=False
	randomize_initial_rotation=False
   ```

2. In the ```MujocoManipulateEnv``` class we redefined the method ```_sample_goal(self)``` as
	```
	def _sample_goal(self):
		goal = np.concatenate([[1, 0.87, 0.2], [1, 0, 0, 0]])
		return goal
    ```



In ```/gymnasium_robotics/envs/shadow_dexterous_hand/manipulate_touch_sensors.py``` we made the following changes:
1. in ```__init__()``` for ```MujocoManipulateTouchSensorsEnv``` class, we set the following variables
	```
	randomize_initial_position=False
	randomize_initial_rotation=False
   ```


In ```/gymnasium_robotics/__init__.py``` we registered a new environment ```HandManipulateBlockEggPen_BooleanTouchSensors-v1```.
This environment is defined in the file ```gymnasium_robotics\envs\shadow_dexterous_hand\manipulate_block_egg_pen_touch_sensors.py```.
This file merely randomizes which object ```.xml``` file is used in the ```model_path``` argument of the class initialization. 

To randomize friction and density we created the files

 - ```/gymnasium_robotics/envs/assets/hand/manipulate_block_touch_sensors_rand.xml```
 - ```/gymnasium_robotics/envs/assets/hand/manipulate_egg_touch_sensors_rand.xml```
 - ```/gymnasium_robotics/envs/assets/hand/manipulate_pen_touch_sensors_rand.xml```

The default values are ```Density=1000```, ```friction=“1 0.005 0.0001”```. More information about the MuJoCo .xml files is available [here](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom).


In ```/gymnasium_robotics/envs/shadow_dextrous_hand/manipulate_touch_sensors.py``` we changed the ```__init__()``` function for the class ```MujocoManipulateTouchSensorsEnv```.
We set ```randomize_initial_position=False``` and ```randomize_initial_rotation=False```.

In ```/gymnasium_robotics/utils/mujoco_utils.py``` we added the function ```robot_set_obs(model, data, joint_names, pos_values, vel_values)``` which is called by ```_robot_set_obs(self, obs)``` in the ```/gymnasium_robotics/envs/shadow_dexterous_hand/manipulate.py```file.

In ```manipulate.py``` we changed the ```compute_reward()``` function.

<!-- In ```/home/joe/Documents/risk_sensitive_project/venv/lib/python3.8/site-packages/gymnasium_robotics/envs/assets/hand/MainProcess.xml``` we changed the timestep from 0.002 to 0.01. Same for the other xml files there. Actually, was not necessary (just change the embedded xml strings in manipulate_block_egg_pen_touch_sensors.py) -->

<!-- In ```robot_env.py``` changed render_fps in metadata to 5 from 25 to account for changing timestep from 0.002 to 0.01. -->

