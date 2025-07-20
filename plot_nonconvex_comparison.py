from summary.utils import(
    make_animation_plots,
    make_bicycle_comparison_report,
    plot_run_summary)
from simulators import(
    load_config,
    CarSingle5DEnv,
    BicycleReachAvoid5DMargin,
    PrintLogger,
    Bicycle5DCost)
import jax
from shutil import copyfile
from matplotlib import pyplot as plt
import argparse
import imageio
import numpy as np
import copy
from typing import Dict
import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current_dir))

from cbf_ddp_softmax import run_ddp_cbf

os.environ["CUDA_VISIBLE_DEVICES"] = " "

jax.config.update('jax_platform_name', 'cpu')

config_file = './test_configs/reachability/test_config_cbf_reachability_non_convex_obstacle_bic5D.yaml'
road_boundary = 3.5

fig = plt.figure(layout='constrained', figsize=(7.5, 3.4))
colorlist = [(0, 0, 1, 1), (1, 0, 0, 1), (0, 0.3, 1.0, 0.8)]
labellist = ['CBFDDP-SM', 'CBFDDP-HM', 'CBFDDP-SM-Ellipse']
stylelist = ['solid', 'solid', 'solid']
legend_fontsize = 9

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

subfigs = fig.subfigures(1, 2, wspace=0.05, width_ratios=[1.6, 1])
ax = subfigs[0].subplots(1, 1)
env.render_state_cost_map(ax, nx=500, ny=500, vel=0.0, yaw=0.0, delta=0.0)
env.render_obs(ax=ax, c='#DCDCDC')

out_folder, plot_tag, config_agent = run_ddp_cbf(config_file, road_boundary, filter_type='SoftCBF', is_task_ilqr=True, line_search='baseline')
out_folder, plot_tag, config_agent = run_ddp_cbf(config_file, road_boundary, filter_type='CBF', is_task_ilqr=True, line_search='baseline')

plot_softcbf_data = np.load(os.path.join(out_folder, f"road_boundary={road_boundary}/SoftCBF/figure/save_data.npy"), allow_pickle=True)
plot_hardcbf_data = np.load(os.path.join(out_folder, f"road_boundary={road_boundary}/CBF/figure/save_data.npy"), allow_pickle=True)

plot_softcbf_data = plot_softcbf_data.ravel()[0]
plot_hardcbf_data = plot_hardcbf_data.ravel()[0]

ellipse_config_file = './test_configs/reachability/test_config_cbf_reachability_non_convex_obstacle_bic5D_ellipse.yaml'
ellipse_out_folder, _, _ = run_ddp_cbf(ellipse_config_file, road_boundary, filter_type='SoftCBF', is_task_ilqr=True, line_search='baseline')

ellipse_config = load_config(ellipse_config_file)
ellipse_config_env = ellipse_config['environment']
ellipse_config_agent = ellipse_config['agent']
ellipse_config_solver = ellipse_config['solver']
ellipse_config_cost = ellipse_config['cost']
ellipse_config_cost.N = ellipse_config_solver.N
ellipse_config_cost.V_MIN = ellipse_config_agent.V_MIN
ellipse_config_cost.DELTA_MIN = ellipse_config_agent.DELTA_MIN
ellipse_config_cost.V_MAX = ellipse_config_agent.V_MAX
ellipse_config_cost.DELTA_MAX = ellipse_config_agent.DELTA_MAX
ellipse_config_cost.TRACK_WIDTH_RIGHT = road_boundary
ellipse_config_cost.TRACK_WIDTH_LEFT = road_boundary
ellipse_config_env.TRACK_WIDTH_RIGHT = road_boundary
ellipse_config_env.TRACK_WIDTH_LEFT = road_boundary
ellipse_config_agent.FILTER_TYPE = 'SoftCBF'

ellipse_env = CarSingle5DEnv(ellipse_config_env, ellipse_config_agent, ellipse_config_cost)

subfigs = fig.subfigures(1, 2, wspace=0.05, width_ratios=[1.6, 1])
ax = subfigs[0].subplots(1, 1)
ellipse_env.render_obs(ax=ax, c='#F4F6F6')

plot_actions_list = []
plot_obses_list = []
plot_obses_complete_filter_list = []
plot_obses_barrier_filter_list = []

plot_actions_list.append( np.array(plot_softcbf_data['actions']) )
plot_obses_list.append( np.array(plot_softcbf_data['obses'] ) )
plot_obses_complete_filter_list.append( np.array(plot_softcbf_data['complete_indices'] ) )
plot_obses_barrier_filter_list.append( np.array(plot_softcbf_data['barrier_indices'] ) )

plot_actions_list.append( np.array(plot_hardcbf_data['actions']) )
plot_obses_list.append( np.array(plot_hardcbf_data['obses'] ) )
plot_obses_complete_filter_list.append( np.array(plot_hardcbf_data['complete_indices'] ) )
plot_obses_barrier_filter_list.append( np.array(plot_hardcbf_data['barrier_indices'] ) )

for idx, obs_data in enumerate(plot_obses_list):
    sc = ax.plot(
        obs_data[:, 0], obs_data[:, 1], color=colorlist[int(idx)], alpha = 1.0, 
        label=labellist[int(idx)], linewidth=1.5, linestyle=stylelist[int(idx)]
    )
    env.render_footprint(ax, obs=obs_data[-1], c='b', lw=0.5)
    complete_filter_indices = plot_obses_complete_filter_list[idx]
    barrier_filter_indices = plot_obses_barrier_filter_list[idx]
    if len(complete_filter_indices)>0:
        ax.plot(obs_data[complete_filter_indices, 0], 
                obs_data[complete_filter_indices, 1], 'o', 
                color=colorlist[int(idx)], alpha=0.65, markersize=3.0)

    if len(barrier_filter_indices)>0:
        ax.plot(obs_data[barrier_filter_indices, 0], 
                obs_data[barrier_filter_indices, 1], 'x', 
                color=colorlist[int(idx)], alpha=0.65, markersize=3.0, 
                label=labellist[int(idx)] + ' filter')

ax.legend(framealpha=0, fontsize=legend_fontsize, loc='upper left', 
            ncol=2, bbox_to_anchor=(-0.05, 1.35), fancybox=False, shadow=False)

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

fig.savefig(
        "nonconvex_comparison_bic5d.png", dpi=200, 
        bbox_inches='tight'
    )
