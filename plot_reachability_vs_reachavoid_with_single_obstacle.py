from simulators import(
    load_config,
    CarSingle5DEnv,
    BicycleReachAvoid5DMargin)
import jax
from matplotlib import pyplot as plt
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current_dir))

from cbf_ddp_softmax import run_ddp_cbf

os.environ["CUDA_VISIBLE_DEVICES"] = " "

jax.config.update('jax_platform_name', 'cpu')

fig = plt.figure(layout='constrained', figsize=(7.0, 3.4))
colorlist = [(0.0, 0, 1.0, 0.5), (0.0, 0.6, 0.3, 0.5), (1.0, 0.0, 0.0, 0.5), (0, 0, 0, 0.5), (0.45, 0.0, 1.0, 0.5)]
labellist = ['Reach-avoid (only obs)', 'Reach-avoid (reinitialize)', 'Reachability (only obs)', 'Reach-avoid (cons)', 'Reach-avoid (slow down)']
stylelist = ['solid', 'solid', 'dashed', 'solid', 'dashed']
legend_fontsize = 6.5

########## Reach avoid with only obstacle ############
config_file = './test_configs/reachavoid/test_config_cbf_reachavoid_single_obstacle_bic5D_singular.yaml'
road_boundary = 3.5

# Load the config to get key parameters needed for plot generation.
config = load_config(config_file)
config_env = config['environment']
config_agent = config['agent']
config_solver = config['solver']
config_cost = config['cost']
config_cost.N = config_solver.N
config_cost.V_MIN = config_agent.V_MIN
config_cost.DELTA_MIN = config_agent.DELTA_MIN
config_cost.V_MAX = config_agent.V_MAX
config_cost.DELTA_MAX = config_agent.DELTA_MAX
config_cost.TRACK_WIDTH_RIGHT = road_boundary
config_cost.TRACK_WIDTH_LEFT = road_boundary
config_env.TRACK_WIDTH_RIGHT = road_boundary
config_env.TRACK_WIDTH_LEFT = road_boundary
config_agent.FILTER_TYPE = 'SoftCBF'

env = CarSingle5DEnv(config_env, config_agent, config_cost)

subfigs = fig.subfigures(1, 2, wspace=0.05, width_ratios=[1.3, 1])
ax = subfigs[0].subplots(1, 1)
env.render_obs(ax=ax, c='k')

out_folder, plot_tag, config_agent, config_solver = run_ddp_cbf(config_file, road_boundary, filter_type='SoftCBF', is_task_ilqr=True, line_search='baseline')
plot_softcbf_data_reachavoid_only_obstacle = np.load(os.path.join(out_folder, f"road_boundary={road_boundary}/SoftCBF/figure/save_data.npy"), allow_pickle=True)
plot_softcbf_data_reachavoid_only_obstacle = plot_softcbf_data_reachavoid_only_obstacle.ravel()[0]


########## Reach avoid with only obstacle  and reiterate ############
config_file = './test_configs/reachavoid/test_config_cbf_reachavoid_single_obstacle_bic5D_singular_reiterate.yaml'
road_boundary = 3.5

# Load the config to get key parameters needed for plot generation.
config = load_config(config_file)
config_env = config['environment']
config_agent = config['agent']
config_solver = config['solver']
config_cost = config['cost']
config_cost.N = config_solver.N
config_cost.V_MIN = config_agent.V_MIN
config_cost.DELTA_MIN = config_agent.DELTA_MIN
config_cost.V_MAX = config_agent.V_MAX
config_cost.DELTA_MAX = config_agent.DELTA_MAX
config_cost.TRACK_WIDTH_RIGHT = road_boundary
config_cost.TRACK_WIDTH_LEFT = road_boundary
config_env.TRACK_WIDTH_RIGHT = road_boundary
config_env.TRACK_WIDTH_LEFT = road_boundary
config_agent.FILTER_TYPE = 'SoftCBF'

env = CarSingle5DEnv(config_env, config_agent, config_cost)
out_folder, plot_tag, config_agent, config_solver = run_ddp_cbf(config_file, road_boundary, filter_type='SoftCBF', is_task_ilqr=True, line_search='baseline')
plot_softcbf_data_reachavoid_only_obstacle_reiterate = np.load(os.path.join(out_folder, f"road_boundary={road_boundary}/SoftCBF/figure/save_data.npy"), allow_pickle=True)
plot_softcbf_data_reachavoid_only_obstacle_reiterate = plot_softcbf_data_reachavoid_only_obstacle_reiterate.ravel()[0]

########## Reachability with all constraints ############
config_file = './test_configs/reachability/test_config_cbf_reachability_single_obstacle_bic5D_singular.yaml'

# Load the config to get key parameters needed for plot generation.
config = load_config(config_file)
config_env = config['environment']
config_agent = config['agent']
config_solver = config['solver']
config_cost = config['cost']
config_cost.N = config_solver.N
config_cost.V_MIN = config_agent.V_MIN
config_cost.DELTA_MIN = config_agent.DELTA_MIN
config_cost.V_MAX = config_agent.V_MAX
config_cost.DELTA_MAX = config_agent.DELTA_MAX
config_cost.TRACK_WIDTH_RIGHT = road_boundary
config_cost.TRACK_WIDTH_LEFT = road_boundary
config_env.TRACK_WIDTH_RIGHT = road_boundary
config_env.TRACK_WIDTH_LEFT = road_boundary
config_agent.FILTER_TYPE = 'SoftCBF'

env = CarSingle5DEnv(config_env, config_agent, config_cost)
env.render_obs(ax=ax, c='k')

out_folder, plot_tag, config_agent, config_solver = run_ddp_cbf(config_file, road_boundary, filter_type='SoftCBF', is_task_ilqr=True, line_search='baseline')
plot_softcbf_data_reachability = np.load(os.path.join(out_folder, f"road_boundary={road_boundary}/SoftCBF/figure/save_data.npy"), allow_pickle=True)
plot_softcbf_data_reachability = plot_softcbf_data_reachability.ravel()[0]

########## Reach-avoid with road boundary and delta constraints ############
config_file = './test_configs/reachavoid/test_config_cbf_reachavoid_single_obstacle_bic5D_singular_constraints.yaml'

# Load the config to get key parameters needed for plot generation.
config = load_config(config_file)
config_env = config['environment']
config_agent = config['agent']
config_solver = config['solver']
config_cost = config['cost']
config_cost.N = config_solver.N
config_cost.V_MIN = config_agent.V_MIN
config_cost.DELTA_MIN = config_agent.DELTA_MIN
config_cost.V_MAX = config_agent.V_MAX
config_cost.DELTA_MAX = config_agent.DELTA_MAX
config_cost.TRACK_WIDTH_RIGHT = road_boundary
config_cost.TRACK_WIDTH_LEFT = road_boundary
config_env.TRACK_WIDTH_RIGHT = road_boundary
config_env.TRACK_WIDTH_LEFT = road_boundary
config_agent.FILTER_TYPE = 'SoftCBF'
action_space = np.array(config_agent.ACTION_RANGE, dtype=np.float32)

env = CarSingle5DEnv(config_env, config_agent, config_cost)
env.render_obs(ax=ax, c='k')

out_folder, plot_tag, config_agent, config_solver = run_ddp_cbf(config_file, road_boundary, filter_type='SoftCBF', is_task_ilqr=True, line_search='baseline')
plot_softcbf_data_reachavoid_constraints = np.load(os.path.join(out_folder, f"road_boundary={road_boundary}/SoftCBF/figure/save_data.npy"), allow_pickle=True)
plot_softcbf_data_reachavoid_constraints = plot_softcbf_data_reachavoid_constraints.ravel()[0]

########## Reach-avoid with reduced velocity ############
config_file = './test_configs/reachavoid/test_config_cbf_reachavoid_single_obstacle_bic5D_singular_reduced_velocity.yaml'

# Load the config to get key parameters needed for plot generation.
config = load_config(config_file)
config_env = config['environment']
config_agent = config['agent']
config_solver = config['solver']
config_cost = config['cost']
config_cost.N = config_solver.N
config_cost.V_MIN = config_agent.V_MIN
config_cost.DELTA_MIN = config_agent.DELTA_MIN
config_cost.V_MAX = config_agent.V_MAX
config_cost.DELTA_MAX = config_agent.DELTA_MAX
config_cost.TRACK_WIDTH_RIGHT = road_boundary
config_cost.TRACK_WIDTH_LEFT = road_boundary
config_env.TRACK_WIDTH_RIGHT = road_boundary
config_env.TRACK_WIDTH_LEFT = road_boundary
config_agent.FILTER_TYPE = 'SoftCBF'

env = CarSingle5DEnv(config_env, config_agent, config_cost)
env.render_obs(ax=ax, c='k')

out_folder, plot_tag, config_agent, config_solver = run_ddp_cbf(config_file, road_boundary, filter_type='SoftCBF', is_task_ilqr=True, line_search='baseline')
plot_softcbf_data_reachavoid_reducedvelocity = np.load(os.path.join(out_folder, f"road_boundary={road_boundary}/SoftCBF/figure/save_data.npy"), allow_pickle=True)
plot_softcbf_data_reachavoid_reducedvelocity = plot_softcbf_data_reachavoid_reducedvelocity.ravel()[0]

# Plot everything for visualizing in report.
plot_actions_list = []
plot_obses_list = []
plot_obses_complete_filter_list = []
plot_obses_barrier_filter_list = []
plot_values_list = []

plot_actions_list.append( np.array(plot_softcbf_data_reachavoid_only_obstacle['actions']) )
plot_obses_list.append( np.array(plot_softcbf_data_reachavoid_only_obstacle['obses'] ) )
plot_obses_complete_filter_list.append( np.array(plot_softcbf_data_reachavoid_only_obstacle['complete_indices'] ) )
plot_obses_barrier_filter_list.append( np.array(plot_softcbf_data_reachavoid_only_obstacle['barrier_indices'] ) )
plot_values_list.append( np.array(plot_softcbf_data_reachavoid_only_obstacle['values'] ) )

plot_actions_list.append( np.array(plot_softcbf_data_reachavoid_only_obstacle_reiterate['actions']) )
plot_obses_list.append( np.array(plot_softcbf_data_reachavoid_only_obstacle_reiterate['obses'] ) )
plot_obses_complete_filter_list.append( np.array(plot_softcbf_data_reachavoid_only_obstacle_reiterate['complete_indices'] ) )
plot_obses_barrier_filter_list.append( np.array(plot_softcbf_data_reachavoid_only_obstacle_reiterate['barrier_indices'] ) )
plot_values_list.append( np.array(plot_softcbf_data_reachavoid_only_obstacle_reiterate['values'] ) )

plot_actions_list.append( np.array(plot_softcbf_data_reachability['actions']) )
plot_obses_list.append( np.array(plot_softcbf_data_reachability['obses'] ) )
plot_obses_complete_filter_list.append( np.array(plot_softcbf_data_reachability['complete_indices'] ) )
plot_obses_barrier_filter_list.append( np.array(plot_softcbf_data_reachability['barrier_indices'] ) )
plot_values_list.append( np.array(plot_softcbf_data_reachability['values'] ) )

plot_actions_list.append( np.array(plot_softcbf_data_reachavoid_constraints['actions']) )
plot_obses_list.append( np.array(plot_softcbf_data_reachavoid_constraints['obses'] ) )
plot_obses_complete_filter_list.append( np.array(plot_softcbf_data_reachavoid_constraints['complete_indices'] ) )
plot_obses_barrier_filter_list.append( np.array(plot_softcbf_data_reachavoid_constraints['barrier_indices'] ) )
plot_values_list.append( np.array(plot_softcbf_data_reachavoid_constraints['values'] ) )

plot_actions_list.append( np.array(plot_softcbf_data_reachavoid_reducedvelocity['actions']) )
plot_obses_list.append( np.array(plot_softcbf_data_reachavoid_reducedvelocity['obses'] ) )
plot_obses_complete_filter_list.append( np.array(plot_softcbf_data_reachavoid_reducedvelocity['complete_indices'] ) )
plot_obses_barrier_filter_list.append( np.array(plot_softcbf_data_reachavoid_reducedvelocity['barrier_indices'] ) )
plot_values_list.append( np.array(plot_softcbf_data_reachavoid_reducedvelocity['values'] ) )

for idx, obs_data in enumerate(plot_obses_list):
    sc = ax.plot(
        obs_data[:, 0], obs_data[:, 1], color=colorlist[int(idx)], alpha = 1.0, 
        label=labellist[int(idx)], linewidth=1.0, linestyle=stylelist[int(idx)],
    )
    env.render_footprint(ax, obs=obs_data[-1], c='b', lw=0.5)
    complete_filter_indices = plot_obses_complete_filter_list[idx]
    barrier_filter_indices = plot_obses_barrier_filter_list[idx]
    if len(complete_filter_indices)>0:
        ax.plot(obs_data[complete_filter_indices, 0], 
                obs_data[complete_filter_indices, 1], 'o', 
                color=colorlist[int(idx)], alpha=0.65, markersize=1.0)

    if len(barrier_filter_indices)>0:
        ax.plot(obs_data[barrier_filter_indices, 0], 
                obs_data[barrier_filter_indices, 1], 'x', 
                color=colorlist[int(idx)], alpha=0.65, markersize=1.0, 
                label=labellist[int(idx)] + ' filter')

ax.legend(framealpha=0, fontsize=legend_fontsize, loc='upper left', 
            ncol=2, bbox_to_anchor=(-0.05, 1.4), fancybox=False, shadow=False)
ax.set_title(f'Trajectories with {config_agent.EGO_RADIUS} radius for ego vehicle', fontsize=8)
# ax.text(0.79, 0.79, f'Soft constraint with zero ego radius', color='black', ha='center', va='center', fontsize=5, fontweight='bold')
ax.set_xticks(ticks=[0, env.visual_extent[1]], labels=[0, env.visual_extent[1]], 
                fontsize=legend_fontsize)
ax.set_yticks(ticks=[env.visual_extent[2], env.visual_extent[3]], 
                labels=[env.visual_extent[2], env.visual_extent[3]], 
                fontsize=legend_fontsize)

ax.plot(np.linspace(0, env.visual_extent[1], 100), np.array([road_boundary]*100), 'k--')
ax.plot(np.linspace(0, env.visual_extent[1], 100), np.array([-1*road_boundary]*100), 'k--')
ax.set_xlabel('X position', fontsize=legend_fontsize)
ax.set_ylabel('Y position', fontsize=legend_fontsize)
ax.yaxis.set_label_coords(-0.03, 0.5)
ax.xaxis.set_label_coords(0.5, -0.04)

axes = subfigs[1].subplots(3, 1)

maxsteps = 0
dt = config_agent.DT
for idx, controls_data in enumerate(plot_actions_list):
    nsteps = controls_data.shape[0]
    maxsteps = np.maximum(maxsteps, nsteps)
    x_times = dt*np.arange(nsteps)
    fillarray = np.zeros(maxsteps)
    fillarray[np.array(plot_obses_barrier_filter_list[idx], dtype=np.int64)] = 1
    axes[0].plot(x_times, controls_data[:, 0], label=labellist[int(idx)], c=colorlist[int(idx)], 
                    alpha = 0.5, linewidth=1.0, linestyle=stylelist[idx])
    axes[1].plot(x_times, controls_data[:, 1], label=labellist[int(idx)], c=colorlist[int(idx)], 
                    alpha = 0.5, linewidth=1.0, linestyle=stylelist[idx])
    axes[2].plot(x_times, plot_values_list[idx], label=labellist[int(idx)], c=colorlist[int(idx)], 
                    alpha = 0.5, linewidth=1.0, linestyle=stylelist[idx])
    if idx==4:
        axes[0].fill_between(x_times, action_space[0, 0], action_space[0, 1], 
                                where=fillarray[0:nsteps], color=colorlist[int(idx)], alpha=0.15)
        axes[1].fill_between(x_times, action_space[1, 0], action_space[1, 1], 
                                where=fillarray[0:nsteps], color=colorlist[int(idx)], alpha=0.15)
        axes[2].fill_between(x_times, action_space[1, 0], action_space[1, 1], 
                                where=fillarray[0:nsteps], color=colorlist[int(idx)], alpha=0.15)

    #axes[0].set_xlabel('Time index', fontsize=legend_fontsize)
    axes[0].set_ylabel('Acceleration', fontsize=legend_fontsize)
    #axes[0].grid(True)
    axes[0].set_xticks(ticks=[], labels=[], fontsize=5, labelsize=5)
    axes[0].set_yticks(ticks=[action_space[0, 0], action_space[0, 1]], 
                        labels=[action_space[0, 0], action_space[0, 1]], 
                        fontsize=legend_fontsize)
    #axes[0].legend(framealpha=0, fontsize=legend_fontsize, loc='upper left', 
    #                ncol=2, bbox_to_anchor=(-0.05, 1.6), fancybox=False, shadow=False)
    axes[0].yaxis.set_label_coords(-0.04, 0.5)

    #axes[1].set_xlabel('Time (s)', fontsize=legend_fontsize)
    axes[1].set_ylabel('Steer control', fontsize=legend_fontsize)
    #axes[1].grid(True)
    axes[1].set_xticks(ticks=[], labels=[], fontsize=legend_fontsize)
    # axes[1].set_yticks(ticks=[action_space[1, 0], action_space[1, 1]], 
    #                     labels=[action_space[1, 0], action_space[1, 1]], 
    #                     fontsize=legend_fontsize)
    axes[1].set_yticks(ticks=[-1.0, 1.0], 
                        labels=[-1.0, 1.0], 
                        fontsize=legend_fontsize)
    #axes[1].set_ylim([action_space[1, 0], action_space[1, 1]])
    axes[1].set_ylim([-1.0, 1.0])
    #axes[1].legend(fontsize=legend_fontsize)
    axes[1].yaxis.set_label_coords(-0.04, 0.5)
    axes[1].xaxis.set_label_coords(0.5, -0.04)

    axes[2].set_xlabel('Time (s)', fontsize=legend_fontsize)
    axes[2].set_ylabel('Value function', fontsize=legend_fontsize)
    #axes[1].grid(True)
    axes[2].set_xticks(ticks=[0, round(dt*maxsteps, 2)], labels=[0, round(dt*maxsteps, 2)], fontsize=legend_fontsize)
    axes[2].set_yticks(ticks=[0.0, 3.0], 
                        labels=[0.0, 3.0], 
                        fontsize=legend_fontsize)
    axes[2].set_ylim([0.0, 3.0])
    #axes[1].legend(fontsize=legend_fontsize)
    axes[2].yaxis.set_label_coords(-0.04, 0.5)
    axes[2].xaxis.set_label_coords(0.5, -0.04)

fig.savefig(
        "./plots_summary/reachability_vs_reachavoid_comparison_bic5d.pdf", dpi=400, 
        bbox_inches='tight'
    )
fig.savefig(
        "./plots_summary/reachability_vs_reachavoid_comparison_bic5d.png", dpi=400, 
        bbox_inches='tight'
    )
