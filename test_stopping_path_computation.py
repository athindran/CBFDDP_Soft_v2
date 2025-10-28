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
from jax import numpy as jp
from matplotlib import cm
from scipy.integrate import solve_ivp
import argparse
import imageio
import numpy as np
import copy
from typing import Dict
from matplotlib import pyplot as plt
import os
import sys
sys.path.append(".")

config_file = './test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_1_bic5D.yaml'
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
config_cost.TRACK_WIDTH_RIGHT = 3.5
config_cost.TRACK_WIDTH_LEFT = 3.5
config_env.TRACK_WIDTH_RIGHT = 3.5
config_env.TRACK_WIDTH_LEFT = 3.5
config_agent.FILTER_TYPE = 'SoftCBF'

env = CarSingle5DEnv(config_env, config_agent, config_cost)

initial_state = jp.array([0.1, 0.2, 2.5, -0.96, 0.95])
stopping_states, stopping_ctrls = env.agent.dyn.compute_stopping_path(initial_state)

rollout_stopping_states = []
current_state = np.array(initial_state)
stopping_ctrl = np.array(env.agent.dyn.stopping_ctrl)
idx = 0
while current_state[2]>0:
    rollout_stopping_states.append(current_state)
    ode_out = solve_ivp(fun=env.agent.dyn.disc_deriv_numpy, y0=current_state.ravel(), args=(stopping_ctrl[np.newaxis, :]), t_span=(0, env.agent.dyn.dt))
    current_state = ode_out.y[:, -1]
    idx += 1

rollout_stopping_states = jp.array(rollout_stopping_states).T

fig = plt.figure()
sc = plt.scatter(
    rollout_stopping_states[0, :-1], rollout_stopping_states[1, :-1], s=10, c=rollout_stopping_states[2, :-1], cmap=cm.jet,
    vmin=0, vmax=1.5, edgecolor='none', marker='x', label='rollout'
)
plt.plot(stopping_states[0, :-1], stopping_states[1, :-1], 'k-', label='analytical')
plt.legend()
plt.grid()
plt.show()


config_file = './test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_1_bic4D.yaml'
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
config_cost.TRACK_WIDTH_RIGHT = 3.5
config_cost.TRACK_WIDTH_LEFT = 3.5
config_env.TRACK_WIDTH_RIGHT = 3.5
config_env.TRACK_WIDTH_LEFT = 3.5
config_agent.FILTER_TYPE = 'SoftCBF'

env = CarSingle5DEnv(config_env, config_agent, config_cost)

initial_state = jp.array([0.1, 0.2, 2.5, 1.3])
stopping_states, stopping_ctrls = env.agent.dyn.compute_stopping_path(initial_state)

rollout_stopping_states = []
current_state = jp.array(initial_state)
stopping_ctrl = env.agent.dyn.get_stopping_ctrl(current_state)
while env.agent.dyn.check_stopped(current_state):
    rollout_stopping_states.append(current_state)
    current_state, _ = env.agent.dyn.integrate_forward_jax(current_state, stopping_ctrl)

rollout_stopping_states = jp.array(rollout_stopping_states).T

fig = plt.figure()
sc = plt.scatter(
    rollout_stopping_states[0, :-1], rollout_stopping_states[1, :-1], s=10, c=rollout_stopping_states[2, :-1], cmap=cm.jet,
    vmin=0, vmax=1.5, edgecolor='none', marker='x', label='rollout'
)
plt.plot(stopping_states[0, :-1], stopping_states[1, :-1], 'k-', label='analytical')
plt.legend()
plt.grid()
plt.show()

config_file = './test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_pointmass4D.yaml'
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
config_cost.TRACK_WIDTH_RIGHT = 3.5
config_cost.TRACK_WIDTH_LEFT = 3.5
config_env.TRACK_WIDTH_RIGHT = 3.5
config_env.TRACK_WIDTH_LEFT = 3.5
config_agent.FILTER_TYPE = 'SoftCBF'

env = CarSingle5DEnv(config_env, config_agent, config_cost)

# The rollout stopping path will not stop at zero velocity perfectly but will overshoot.
initial_state = jp.array([0.1, 0.4, 2.0, -1.5])
stopping_states, stopping_ctrls = env.agent.dyn.compute_stopping_path(initial_state)

rollout_stopping_states = []
current_state = jp.array(initial_state)
stopping_ctrl = jp.zeros((2,))
while env.agent.dyn.check_stopped(current_state, stopping_ctrl):
    stopping_ctrl = env.agent.dyn.get_stopping_ctrl(current_state, jp.zeros(2,))
    rollout_stopping_states.append(current_state)
    current_state, _ = env.agent.dyn.integrate_forward_jax(current_state, stopping_ctrl)

rollout_stopping_states = jp.array(rollout_stopping_states).T

fig = plt.figure()
sc = plt.scatter(
    rollout_stopping_states[0, :-1], rollout_stopping_states[1, :-1], s=10, c=rollout_stopping_states[2, :-1], cmap=cm.jet,
    vmin=0, vmax=1.5, edgecolor='none', marker='x', label='rollout'
)
plt.plot(stopping_states[0, :-1], stopping_states[1, :-1], 'k-', label='analytical')
plt.legend()
plt.grid()
plt.show()