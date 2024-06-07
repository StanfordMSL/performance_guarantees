import matplotlib.pyplot as plt
import os
import numpy as np
import yaml
import pdb

'''

This file is used to plot the following experiments:

1. visualize the validity of the concentration bounds for a fixed plan

2. visualize the validity of the chance constraint method

3. visualize the validity and necessity of the multi-hypothesis correction

'''

# user input
results_folder = "experiments/runs/shift_fixed_var"
run_folder = "20240523-013000" 

# get filepaths from user input
run_data = os.path.join(results_folder, run_folder, "results.npz")
experiment_name = results_folder.split("experiments/runs/", 1)[1]
plot_folder = "experiments/figures/"

print("Results folder: ", results_folder)
print("Experiment name: ", experiment_name)

# load results
results = np.load(run_data)

# load config file used to generate results
config_file = os.path.join(results_folder, run_folder, "config.yaml")
with open(config_file, "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

# set default plotting parameters
plt.rcParams.update({
    'font.size': 12,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})




# load and plot experiment data
if experiment_name == "valid_fixed_exp" or experiment_name == "valid_fixed_var" or experiment_name == "valid_fixed_cvar" or experiment_name == "valid_fixed_pr":
	
	# Generate plot label
	if experiment_name == "valid_fixed_exp":
		label = r'$\overline{\mathbb{E}}$'
	elif experiment_name == "valid_fixed_var":
		label = r'$\overline{\textup{VaR}}_{\tau}$'
	elif experiment_name == "valid_fixed_cvar":
		label = r'$\overline{\textup{CVaR}}_{\tau}$'
	elif experiment_name == "valid_fixed_pr":
		label = r'$\overline{q}$'

	if experiment_name == "valid_fixed_pr":
		# Because for sparse want one bar centered at 0 and another at 1
		# Widths chosen as 1 so that height is probability of 0 or 1 resp.
		# bins = [-0.5, 0.5, 1.5]
		# Actually scale up by 10
		bins = [-0.1, 0.1, 0.9, 1.1]
	else:
		bins = "auto"

	# make and save figure
	fig = plt.figure(figsize=(5,3))
	plt.hist(results["cost_samples"], label=r'$J$', alpha=0.7, bins=bins, density=True, color='skyblue')
	plt.hist(results["bound_samples"], label=label, alpha=0.7, bins="auto", density=True, color='silver')
	plt.axvline(results["true_stat"], linestyle='dashed', color='steelblue')
	plt.axvline(results["bound_quantile"], linestyle='dashed', color='dimgray')
	plt.xlabel(r'Total Cost, $J$')
	plt.ylabel('Empirical Density')
	plt.legend(fontsize="13", loc="upper left")
	plt.show(block=False)

	fig.savefig(os.path.join(plot_folder, experiment_name + '.svg'), bbox_inches='tight')
	fig.savefig(os.path.join(plot_folder, experiment_name + '.png'), bbox_inches='tight')




elif experiment_name == "valid_chance":

	# results data
	empirical_prob_success = results["empirical_prob_success"]
	empirical_prob_accept = results["empirical_prob_accept"]
	theory_prob_success = results["theory_prob_success"]
	theory_prob_accept = results["theory_prob_accept"]

	# config parameters
	num_envs_range_empirical = config["simulation"]["num_envs_range_empirical"]
	num_envs_range_theory = config["simulation"]["num_envs_range_theory"]
	delta = config["bounds"]["delta"]
	epsilon = config["bounds"]["epsilon"]


	fig = plt.figure(figsize=(5,3))
	
	# Plot empirical
	# blues = ['#17a5d4', 'deepskyblue', 'skyblue', 'lightskyblue']
	colors = ['orange', 'red', 'green', 'blue']
	for i, num_envs in enumerate(num_envs_range_empirical):
		plt.scatter(empirical_prob_success[i,:], empirical_prob_accept[i,:], color=colors[i])
	
	# Plot theoretical
	for i,n in enumerate(num_envs_range_theory):
		plt.plot(theory_prob_success[i,:], theory_prob_accept[i,:], label='n = ' + str(n), color=colors[i])

	plt.xlabel(r'$\textup{Pr}[\textup{constraint satisfied}]$')
	plt.ylabel(r'$\textup{Pr}[\textup{concluding chance constraint holds}]$')
	plt.plot(np.linspace(0,1,50), delta * np.ones(50), linestyle='dashed', color='darkgray')
	plt.text(-0.03, delta-0.02, r'$\delta$', fontsize=13, color='black') # -0.02 to center
	plt.axvline(1-epsilon, linestyle='dotted', color='darkgray')
	plt.text(1-epsilon-0.015, -0.1, r'$\tau$', fontsize=13, color='black') # -0.015 to center
	plt.fill_between(np.linspace(0, 1-epsilon), y1=delta, y2=1, facecolor='red', alpha=0.2)
	plt.legend(fontsize="13", loc="upper left")
	plt.show(block=False)

	fig.savefig(os.path.join(plot_folder, experiment_name + '.svg'), bbox_inches='tight')
	fig.savefig(os.path.join(plot_folder, experiment_name + '.png'), bbox_inches='tight')




elif experiment_name == "valid_fixed_multi_hyp":

	# results data
	unver_vals = results["unver_vals"]
	unver_theory_vals = results["unver_theory_vals"]
	ver_vals = results["ver_vals"]
	ver_theory_vals = results["ver_theory_vals"]

	# config data
	delta = config["bounds"]["delta"]

	# Normalize by subtracting off theory_vals then look at the resulting offset distributions
	# bound >= theory when hold so >= 0 means bound holds
	unver_gaps = unver_vals - unver_theory_vals
	ver_gaps = ver_vals - ver_theory_vals

	# Get the delta quantile of the bounds, iff lies to right of 0 then coverage holds
	unver_quant = np.quantile(unver_gaps, delta)
	ver_quant = np.quantile(ver_gaps, delta)


	fig = plt.figure(figsize=(5,3))
	plt.hist(unver_gaps, label='Uncorrected', alpha=0.7, bins='auto', density=True, color='skyblue')
	plt.hist(ver_gaps, label='Corrected', alpha=0.7, bins='auto', density=True, color='silver')
	plt.axvline(unver_quant, linestyle='dashed', color='steelblue')
	plt.axvline(ver_quant, linestyle='dashed', color='dimgray')
	plt.legend(fontsize="13", loc="upper left")
	plt.xlabel(r'$\overline{\textup{VaR}}_{\tau}^* - \textup{VaR}_{\tau}^*$')
	plt.ylabel('Empirical Density')
	plt.show(block=False)

	fig.savefig(os.path.join(plot_folder, experiment_name + '.svg'), bbox_inches='tight')
	fig.savefig(os.path.join(plot_folder, experiment_name + '.png'), bbox_inches='tight')



elif experiment_name in  ['compare_fixed_exp', 'compare_fixed_var', 'compare_fixed_cvar']:
	
	# Generate plot label
	if experiment_name == "compare_fixed_exp":
		label = r'$\overline{\mathbb{E}} - \mathbb{E}$'
		names = ['Ours', '[9]']
	elif experiment_name == "compare_fixed_var":
		label = r'$\overline{\textup{VaR}}_{\tau} - \textup{VaR}_{\tau}$'
		names = ['Ours', '[32]', '[11]']
	elif experiment_name == "compare_fixed_cvar":
		label = r'$\overline{\textup{CVaR}}_{\tau} - \textup{CVaR}_{\tau}$'
		names = ['Ours', '[9]']

	colors = ['silver', 'red', 'blue']
	quantile_colors = ['dimgray', 'firebrick', 'mediumblue']
	bins = "auto"

	# make and save figure
	fig = plt.figure(figsize=(5,3))
	# plt.hist(results["cost_samples"], label=r'$J$', alpha=0.7, bins=bins, density=True, color='skyblue')
	# plt.axvline(results["true_stat"], linestyle='dashed', color='steelblue')

	bound_samples = results["bound_samples"] - results["true_stat"]
	quantiles = results["bound_quantiles"] - results["true_stat"]

	for i in range(bound_samples.shape[0]):
		name = names[i]
		color = colors[i]
		plt.hist(bound_samples[i], label=names[i], alpha=0.7, bins="auto", density=True, color=color)
		plt.axvline(quantiles[i], linestyle='dashed', color=quantile_colors[i])
	
	plt.xlabel(label)
	plt.ylabel('Empirical Density')
	plt.legend(fontsize="13", loc="upper left")
	plt.show(block=False)

	fig.savefig(os.path.join(plot_folder, experiment_name + '.svg'), bbox_inches='tight')
	fig.savefig(os.path.join(plot_folder, experiment_name + '.png'), bbox_inches='tight')

elif experiment_name in ['shift_fixed_exp', 'shift_fixed_var', 'shift_fixed_cvar', 'shift_fixed_pr']:
	# results data
	emp_coverage = results["emp_coverage"]
	theory_coverage = results["theory_coverage"]
	emp_nominal_coverage = results["emp_nominal_coverage"]
	theory_nominal_coverage = results["theory_nominal_coverage"]

	emp_coverage = np.append(emp_coverage, emp_nominal_coverage)
	theory_coverage = np.append(theory_coverage, theory_nominal_coverage)

	# config parameters
	nominal_noise = config["simulation"]["noise"]
	noise_low = config["simulation"]["noise_low"]
	noise_high = config["simulation"]["noise_high"]
	num_shifted = config["simulation"]["num_shifted"]
	delta = config["bounds"]["delta"]

	noise_scales = np.linspace(noise_low, noise_high, num_shifted)
	noise_scales = np.append(noise_scales, nominal_noise)

	# Generate plot label
	if experiment_name == "shift_fixed_exp":
		ylabel = r'$\mathbb{E}(J_{true}) \leq \overline{\mathbb{E}}$'
	elif experiment_name == "shift_fixed_var":
		ylabel = r'$\textup{VaR}_{\tau}(J_{true}) \leq \overline{\textup{VaR}}_{\tau}$'
	elif experiment_name == "shift_fixed_cvar":
		ylabel = r'$\textup{CVaR}_{\tau}(J_{true}) \leq \overline{\textup{CVaR}}_{\tau}$'
	elif experiment_name == "shift_fixed_pr":
		ylabel = r'$q_{true} \leq \overline{q}$'
	ylabel = r'$\textup{Pr}[$' + ylabel + r'$]$' # ' = 1 - \delta_{true}$'

	# make and save figure
	fig = plt.figure(figsize=(5,3))
	plt.scatter(noise_scales, emp_coverage, marker='o', label='Empirical', color='blue')
	# plt.scatter(nominal_noise, nominal_coverage, marker='*', label='Nominal Empirical', color='green')
	plt.scatter(noise_scales, theory_coverage, marker='x', label='Theoretical', color='orange')

	plt.plot(np.linspace(noise_low,noise_high,50), (1-delta) * np.ones(50), linestyle='dashed', color='darkgray')
	plt.text(noise_low+0.001, 1-delta-0.1, r'$1 - \delta_{sim}$', fontsize=13, color='black') # Adjust to center label
	plt.axvline(nominal_noise, linestyle='dotted', color='darkgray')
	plt.text(nominal_noise+0.0025, 0.01, r'$\sigma_{sim}$', fontsize=13, color='black') # Adjust to center label

	plt.xlabel(r'True Noise Parameter $\sigma$')
	plt.ylabel(ylabel)
	plt.legend(fontsize="13", loc="lower left")
	plt.show(block=False)

	fig.savefig(os.path.join(plot_folder, experiment_name + '.svg'), bbox_inches='tight')
	fig.savefig(os.path.join(plot_folder, experiment_name + '.png'), bbox_inches='tight')


elif experiment_name in ['robust_fixed_exp', 'robust_fixed_var', 'robust_fixed_cvar', 'robust_fixed_pr']:
	# results data
	emp_coverage = results["emp_coverage"] # shape num_shifted, num_alpha
	emp_nominal_coverage = np.expand_dims(results["emp_nominal_coverage"], axis=0) # shape 1, num_alpha
	emp_coverage = np.vstack([emp_coverage, emp_nominal_coverage]) # shape num_shifted + 1, num_alpha
	alphas = results["alphas"]
	ks_dists = results["ks_dists"]

	# Generate plot label
	if experiment_name == "robust_fixed_exp":
		ylabel = r'$\mathbb{E}(J_{true}) \leq \overline{\mathbb{E}}(\alpha)$'
	elif experiment_name == "robust_fixed_var":
		ylabel = r'$\textup{VaR}_{\tau}(J_{true}) \leq \overline{\textup{VaR}}_{\tau}(\alpha)$'
	elif experiment_name == "robust_fixed_cvar":
		ylabel = r'$\textup{CVaR}_{\tau}(J_{true}) \leq \overline{\textup{CVaR}}_{\tau}(\alpha)$'
	elif experiment_name == "robust_fixed_pr":
		ylabel = r'$q_{true} \leq \overline{q}(\alpha)$'
	ylabel = r'$\textup{Pr}[$' + ylabel + ']'

	# config parameters
	nominal_noise = config["simulation"]["noise"]
	noise_low = config["simulation"]["noise_low"]
	noise_high = config["simulation"]["noise_high"]
	num_shifted = config["simulation"]["num_shifted"]
	delta = config["bounds"]["delta"]

	noise_scales = np.linspace(noise_low, noise_high, num_shifted)
	noise_scales = np.append(noise_scales, nominal_noise)

	ks_dists = np.append(ks_dists, 0)

	# Re-order to insert nominal_noise in correct ascending position
	order = np.argsort(noise_scales)
	emp_coverage = emp_coverage[order, :]
	noise_scales = noise_scales[order]
	ks_dists = ks_dists[order]

	# make and save figure
	fig = plt.figure(figsize=(5,3))
	for i, alpha in enumerate(alphas):

		expect_no_hold = (ks_dists > alpha)
		h = plt.plot(noise_scales, emp_coverage[:, i], label=r'$\alpha = $' + str(np.round(alpha, 3)))
		# Plot with o where theory requires robust bound to hold, x where not required to hold
		plt.scatter(noise_scales[np.where(expect_no_hold)], emp_coverage[np.where(expect_no_hold), i], marker='x', color=h[0].get_color())
		plt.scatter(noise_scales[np.where(1-expect_no_hold)], emp_coverage[np.where(1-expect_no_hold), i], facecolors='none', marker='o', color=h[0].get_color())

	plt.plot(np.linspace(noise_low,noise_high,50), (1-delta) * np.ones(50), linestyle='dashed', color='darkgray')
	plt.text(noise_low+0.001, 1-delta+0.05, r'$1 - \delta$', fontsize=13, color='black') # Adjust to center label
	plt.axvline(nominal_noise, linestyle='dotted', color='darkgray')
	plt.text(nominal_noise+0.0025, 0.05, r'$\sigma_{sim}$', fontsize=13, color='black') # Adjust to center label
	plt.ylim(0,1.05)

	plt.xlabel(r'True Noise Parameter $\sigma$')
	plt.ylabel(ylabel)
	plt.legend(fontsize="13", loc="lower left")
	plt.show(block=False)

	fig.savefig(os.path.join(plot_folder, experiment_name + '.svg'), bbox_inches='tight')
	fig.savefig(os.path.join(plot_folder, experiment_name + '.png'), bbox_inches='tight')

else:
	raise ValueError("\nInvalid experiment name given! \nCheck config filename.")




