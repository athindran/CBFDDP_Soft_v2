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
from tqdm import tqdm
sys.path.append(".")

def verify_bic5d_rollouts():
    print("Rollout calibration for bic5D")
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
    config_cost.STOPPING_COMPUTATION_TYPE = 'rollout'
    cost_rollout = BicycleReachAvoid5DMargin(
                    config_cost, copy.deepcopy(env.agent.dyn), 'SoftCBF')
    config_cost.STOPPING_COMPUTATION_TYPE = 'analytic'
    cost_analytic = BicycleReachAvoid5DMargin(
                    config_cost, copy.deepcopy(env.agent.dyn), 'SoftCBF')
    plotting = False
    seed_state = jp.array([0.1, 0.2, 2.5, -0.96, 1e-12])
    noise_var = np.array([2.0, 5.0, 1.5, 2.0, 1.0])
    with tqdm(total=100, position=0, leave=True) as pbar:
        for iters in tqdm(range(100), position=0, leave=True):
            pbar.update()        
            offset = np.random.rand(5)*noise_var
            initial_state = seed_state + offset
            stopping_states, stopping_ctrls = env.agent.dyn.compute_stopping_path(initial_state)

            rollout_stopping_states = []
            current_state = np.array(initial_state)
            stopping_ctrl = np.array(env.agent.dyn.stopping_ctrl)
            stopping_ctrl_jp = jp.array(env.agent.dyn.stopping_ctrl)
            idx = 0
            while current_state[2]>0:
                rollout_stopping_states.append(current_state)
                ode_out = solve_ivp(fun=env.agent.dyn.disc_deriv_numpy, y0=current_state.ravel(), args=(stopping_ctrl[np.newaxis, :]), t_span=(0, env.agent.dyn.dt))
                current_state = ode_out.y[:, -1]
                idx += 1

            rollout_stopping_states = jp.array(rollout_stopping_states).T

            discrepancy_in_states = jp.max(jp.linalg.norm(rollout_stopping_states - stopping_states[:, 0:rollout_stopping_states.shape[1]], axis=0), axis=0)
            assert discrepancy_in_states<1e-4, f"discrepancy in states: {discrepancy_in_states}"

            target_margin_rollout = cost_rollout.constraint.get_target_stage_margin_rollout(initial_state, stopping_ctrl_jp)
            target_margin_analytic = cost_analytic.constraint.get_target_stage_margin_analytic(initial_state, stopping_ctrl_jp)

            discrepancy_in_target_margin = jp.abs(target_margin_analytic - target_margin_rollout)
            assert discrepancy_in_target_margin<5e-4, f"discrepancy in target margin: {discrepancy_in_target_margin}"

            # Leave out last state where where it has nearly stopped.
            # The gradient at the last state before stopping is expectedly significantly different due to the different mode of stopping.
            target_margin_rollout_derivatives = cost_rollout.constraint.get_derivatives_target(rollout_stopping_states[..., 0:-2], stopping_ctrls[..., 0:rollout_stopping_states.shape[1] - 2])
            target_margin_analytic_derivatives = cost_analytic.constraint.get_derivatives_target(rollout_stopping_states[..., 0:-2], stopping_ctrls[..., 0:rollout_stopping_states.shape[1] - 2])

            for iters in range(5):
                rollout_der = target_margin_rollout_derivatives[iters]
                analytic_der = target_margin_analytic_derivatives[iters]

                assert rollout_der.shape == analytic_der.shape, f"Shape mismatch: {rollout_der.shape}, {analytic_der.shape}"
                abs_diff = jp.abs(rollout_der - analytic_der)**2
                abs_diff_per_state = jp.sqrt(jp.mean(abs_diff, axis=tuple(range(abs_diff.ndim - 1))))
                abs_diff_acc = jp.amax(abs_diff_per_state, axis=0)
                index_abs_diff_acc = jp.argmax(abs_diff_per_state)
                ndim_to_abs_error_acc = [1e-2, 1.0]
                assert abs_diff_acc < ndim_to_abs_error_acc[rollout_der.ndim - 2], f"Derivative norm mismatch: {rollout_der.shape} {analytic_der.shape} {abs_diff_acc} {index_abs_diff_acc} {rollout_stopping_states[..., index_abs_diff_acc]}"            

            if plotting:
                fig = plt.figure()
                sc = plt.scatter(
                    rollout_stopping_states[0, :-1], rollout_stopping_states[1, :-1], s=10, c=rollout_stopping_states[2, :-1], cmap=cm.jet,
                    vmin=0, vmax=1.5, edgecolor='none', marker='x', label='rollout'
                )
                plt.plot(stopping_states[0, :-1], stopping_states[1, :-1], 'k-', label='analytical')
                plt.legend()
                plt.grid()
                plt.show()
    
    print("Bic5d completed.")


def verify_bic4d_rollouts():
    print("Rollout calibration for bic4D")
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
    config_cost.STOPPING_COMPUTATION_TYPE = 'rollout'
    cost_rollout = BicycleReachAvoid5DMargin(
                    config_cost, copy.deepcopy(env.agent.dyn), 'SoftCBF')
    config_cost.STOPPING_COMPUTATION_TYPE = 'analytic'
    cost_analytic = BicycleReachAvoid5DMargin(
                    config_cost, copy.deepcopy(env.agent.dyn), 'SoftCBF')
    plotting = False
    seed_state = jp.array([0.1, 0.2, 1.5, 0.0])
    noise_var = np.array([2.0, 5.0, 1.0, 1.5])

    with tqdm(total=100, position=0, leave=True) as pbar:
        for iters in tqdm(range(100), position=0, leave=True):
            pbar.update()
            offset = np.random.rand(4)*noise_var
            initial_state = seed_state + offset
            stopping_states, stopping_ctrls = env.agent.dyn.compute_stopping_path(initial_state)

            rollout_stopping_states = []
            current_state = jp.array(initial_state)
            while env.agent.dyn.check_stopped(current_state):
                stopping_ctrl = env.agent.dyn.get_stopping_ctrl(current_state)
                rollout_stopping_states.append(current_state)
                current_state, _ = env.agent.dyn.integrate_forward_jax(current_state, stopping_ctrl)

            rollout_stopping_states = jp.array(rollout_stopping_states).T
            discrepancy_in_states = jp.max(jp.linalg.norm(rollout_stopping_states - stopping_states[:, 0:rollout_stopping_states.shape[1]], axis=0), axis=0)
            assert discrepancy_in_states<1e-4, f"discrepancy in states: {discrepancy_in_states}"

            stopping_ctrl_jp = jp.array(env.agent.dyn.stopping_ctrl)
            target_margin_rollout = cost_rollout.constraint.get_target_stage_margin_rollout(initial_state, stopping_ctrl_jp)
            target_margin_analytic = cost_analytic.constraint.get_target_stage_margin_analytic(initial_state, stopping_ctrl_jp)

            discrepancy_in_target_margin = jp.abs(target_margin_analytic - target_margin_rollout)
            assert discrepancy_in_target_margin<1e-4, f"discrepancy in target margin: {discrepancy_in_target_margin}"

            # Leave out last state where where it has nearly stopped.
            # The gradient at the last state before stopping is expectedly significantly different due to the different mode of stopping.
            target_margin_rollout_derivatives = cost_rollout.constraint.get_derivatives_target(rollout_stopping_states[..., 0:-1], stopping_ctrls[..., 0:rollout_stopping_states.shape[1] - 1])
            target_margin_analytic_derivatives = cost_analytic.constraint.get_derivatives_target(rollout_stopping_states[..., 0:-1], stopping_ctrls[..., 0:rollout_stopping_states.shape[1] - 1])

            for iters in range(5):
                rollout_der = target_margin_rollout_derivatives[iters]
                analytic_der = target_margin_analytic_derivatives[iters]

                assert rollout_der.shape == analytic_der.shape, f"Shape mismatch: {rollout_der.shape}, {analytic_der.shape}"
                abs_diff = jp.abs(rollout_der - analytic_der)**2
                abs_diff_per_state = jp.sqrt(jp.mean(abs_diff, axis=tuple(range(abs_diff.ndim - 1))))
                abs_diff_acc = jp.amax(abs_diff_per_state, axis=0)
                index_abs_diff_acc = jp.argmax(abs_diff_per_state)
                ndim_to_abs_error_acc = [1e-2, 1.0]
                assert abs_diff_acc < ndim_to_abs_error_acc[rollout_der.ndim - 2], f"Derivative norm mismatch: {rollout_der.shape} {analytic_der.shape} {abs_diff_acc} {index_abs_diff_acc} {rollout_stopping_states[..., index_abs_diff_acc]}"
            

            if plotting:
                fig = plt.figure()
                sc = plt.scatter(
                    rollout_stopping_states[0, :-1], rollout_stopping_states[1, :-1], s=10, c=rollout_stopping_states[2, :-1], cmap=cm.jet,
                    vmin=0, vmax=1.5, edgecolor='none', marker='x', label='rollout'
                )
                plt.plot(stopping_states[0, :-1], stopping_states[1, :-1], 'k-', label='analytical')
                plt.legend()
                plt.grid()
                plt.show()

    print("Bic4d completed.")


def verify_pm4d_rollouts():
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

    plotting = True
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

    # The rollout stopping state is not the same as analytic stopping state for point mass 4d so we don't proceed further.
    rollout_stopping_states = jp.array(rollout_stopping_states).T

    if plotting:
        fig = plt.figure()
        sc = plt.scatter(
            rollout_stopping_states[0, :-1], rollout_stopping_states[1, :-1], s=10, c=rollout_stopping_states[2, :-1], cmap=cm.jet,
            vmin=0, vmax=1.5, edgecolor='none', marker='x', label='rollout'
        )
        plt.plot(stopping_states[0, :-1], stopping_states[1, :-1], 'k-', label='analytical')
        plt.legend()
        plt.grid()
        plt.show()


    print("Point mass 4d completed")

verify_bic5d_rollouts()
verify_bic4d_rollouts()
verify_pm4d_rollouts()
