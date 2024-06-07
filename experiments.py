import os
import yaml
from datetime import datetime
import analyze_bounds as ab
import analyze_chance as ac
import analyze_fixed_multihyp as afm
import compare_bounds as bc
import sim_distribution_shift as sds
import analyze_robust_bounds as arb
import numpy as np


'''
This file is used to run the following experiments:

1. visualize the validity of the concentration bounds for a fixed plan

2. visualize the validity of the chance constraint method

3. visualize the validity and necessity of the multi-hypothesis correction

'''

# get timestamp for naming the experiment run
current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")

## Load the configuration file
config_file = "experiments/config_files/shift_fixed_var.yaml"
with open(config_file, "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print("\nloaded config file")
print(config)


# get experiment name from config filename
experiment_name = config_file.split("experiments/config_files/", 1)[1]
experiment_name = experiment_name.replace(".yaml", "")


# potentially get saved information if needed for experiment (multihyp case)
if experiment_name == 'valid_fixed_multi_hyp':
    plans_folder_name = 'experiments/config_files/valid_fixed_multi_hyp_plans/'
    plans_file_name =  plans_folder_name + '20230817-224846.npz'

# create folder to save run
str_current_datetime = str(current_datetime)
run_folder = "experiments/runs/" + experiment_name + "/" + str_current_datetime + "/"
os.mkdir(run_folder)


# save config file used
run_yaml = run_folder + "config.yaml"
with open(run_yaml, 'w') as yamlfile:
    yaml.dump(config, yamlfile)
    print("\nsaved config file")


## Run experiment and save results
run_data = run_folder + "results.npz"


if experiment_name == "valid_fixed_exp" or experiment_name == "valid_fixed_var" or experiment_name == "valid_fixed_cvar":
	cost_samples, bound_samples, true_stat, bound_quantile = ab.valid_fixed_bounds(experiment_name, config)
	np.savez(run_data, cost_samples=cost_samples, bound_samples=bound_samples, true_stat=true_stat, bound_quantile=bound_quantile)

elif experiment_name == "valid_fixed_pr":
	cost_samples, bound_samples, true_stat, bound_quantile = ac.valid_pr_empirical(config)
	np.savez(run_data, cost_samples=cost_samples, bound_samples=bound_samples, true_stat=true_stat, bound_quantile=bound_quantile)

elif experiment_name == "valid_chance":
	theory_prob_success, theory_prob_accept = ac.valid_chance_theory(config)
	empirical_prob_success, empirical_prob_accept = ac.valid_chance_empirical(config)
	np.savez(run_data, theory_prob_success=theory_prob_success, theory_prob_accept=theory_prob_accept, empirical_prob_success=empirical_prob_success, empirical_prob_accept=empirical_prob_accept)

elif experiment_name == 'valid_fixed_multi_hyp':
	unver_vals, unver_theory_vals, unver_chosen_plans, ver_vals, \
		ver_theory_vals, ver_chosen_plans = afm.valid_fixed_multi_hyp(config, plans_file_name)
	np.savez(run_data, unver_vals=unver_vals, unver_theory_vals=unver_theory_vals, unver_chosen_plans=unver_chosen_plans, \
		ver_vals=ver_vals, ver_theory_vals=ver_theory_vals, ver_chosen_plans=ver_chosen_plans)

elif experiment_name in ['compare_fixed_exp', 'compare_fixed_var', 'compare_fixed_cvar']:
	cost_samples, true_stat, bound_samples, bound_quantiles = bc.compare_fixed_bounds(experiment_name, config)
	np.savez(run_data, cost_samples=cost_samples, true_stat=true_stat, bound_samples=bound_samples, \
		bound_quantiles=bound_quantiles)

elif experiment_name in ['shift_fixed_exp', 'shift_fixed_var', 'shift_fixed_cvar', 'shift_fixed_pr']:
	shifted_theory_costs, shifted_theory_vals, nominal_costs, nominal_val, fresh_bounds, emp_coverage, theory_coverage, emp_nominal_coverage, theory_nominal_coverage = \
		sds.shifted_bounds(experiment_name, config)
	np.savez(run_data, shifted_theory_costs=shifted_theory_costs, shifted_theory_vals=shifted_theory_vals, nominal_costs=nominal_costs, \
		nominal_val=nominal_val, fresh_bounds=fresh_bounds, emp_coverage=emp_coverage, theory_coverage=theory_coverage, emp_nominal_coverage=emp_nominal_coverage, theory_nominal_coverage=theory_nominal_coverage)

elif experiment_name in ['robust_fixed_exp', 'robust_fixed_var', 'robust_fixed_cvar', 'robust_fixed_pr']:
	shifted_theory_costs, shifted_theory_vals, nominal_costs, nominal_val, fresh_bounds, ks_dists, alphas, emp_coverage, emp_nominal_coverage = \
		arb.robust_bounds(experiment_name, config)
	np.savez(run_data, shifted_theory_costs=shifted_theory_costs, shifted_theory_vals=shifted_theory_vals, nominal_costs=nominal_costs, \
		nominal_val=nominal_val, fresh_bounds=fresh_bounds, ks_dists=ks_dists, alphas=alphas, emp_coverage=emp_coverage, emp_nominal_coverage=emp_nominal_coverage)


else:
	raise ValueError("\nInvalid experiment name given! \nCheck config filename.")





