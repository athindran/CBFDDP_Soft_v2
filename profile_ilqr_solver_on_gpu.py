from simulators import(
    load_config,
    CarSingle5DEnv,
    BicycleReachAvoid5DMargin)
import jax
import argparse
import imageio
from jax import numpy as jnp
import copy
import time
from matplotlib import pyplot as plt
import os
import sys
sys.path.append(".")


def main(config_file, road_boundary, filter_type, initializer_type):
    ## ------------------------------------- Warmup fields ------------------------------------------ ##
    config = load_config(config_file)
    config_env = config['environment']
    config_agent = config['agent']
    config_solver = config['solver']
    config_solver.FILTER_TYPE = filter_type
    config_agent.FILTER_TYPE = filter_type
    config_cost = config['cost']
    dyn_id = config_agent.DYN

    # Provide common fields to cost
    config_cost.N = config_solver.N
    config_cost.V_MIN = config_agent.V_MIN
    config_cost.DELTA_MIN = config_agent.DELTA_MIN
    config_cost.V_MAX = config_agent.V_MAX
    config_cost.DELTA_MAX = config_agent.DELTA_MAX

    config_cost.TRACK_WIDTH_RIGHT = road_boundary
    config_cost.TRACK_WIDTH_LEFT = road_boundary
    config_env.TRACK_WIDTH_RIGHT = road_boundary
    config_env.TRACK_WIDTH_LEFT = road_boundary
    plot_tag = config_env.tag + '-' + str(filter_type)

    env = CarSingle5DEnv(config_env, config_agent, config_cost)

    # region: Constructs placeholder and initializes iLQR
    config_ilqr_cost = copy.deepcopy(config_cost)

    policy_type = None
    cost = None
    config_solver.COST_TYPE = config_cost.COST_TYPE
    if config_cost.COST_TYPE == "Reachavoid":
        policy_type = "iLQRReachAvoid"
        cost = BicycleReachAvoid5DMargin(
            config_ilqr_cost, copy.deepcopy(env.agent.dyn), filter_type)
        env.cost = cost  # ! hacky
    # Not supported
    elif config_cost.COST_TYPE == "Reachability":
        policy_type = "iLQRReachability"
        cost = BicycleReachAvoid5DMargin(
            config_ilqr_cost, copy.deepcopy(env.agent.dyn), filter_type)
        env.cost = cost  # ! hacky

    env.agent.init_policy(
        policy_type=policy_type,
        config=config_solver,
        cost=cost,
        task_cost=None)
    max_iter_receding = config_solver.MAX_ITER_RECEDING

    x_cur = jnp.array([3.0, 0.0, 3.0, 0.05, 0.0])
    # region: Runs iLQR
    # Warms up jit
    env.agent.get_action(obs=x_cur, state=x_cur, warmup=True)
    env.report()

    start_time = time.time()
    control, solver_info = env.agent.get_action(obs=x_cur, state=x_cur, warmup=False)
    control = jax.block_until_ready(control)
    print("Process time : ", time.time() - start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf",
        "--config_file",
        help="Config file path",
        type=str,
        default=os.path.join(
            "./simulators/test_config_yamls",
            "test_config.yaml"))

    parser.add_argument(
        "-rb", "--road_boundary", help="Choose road width", type=float,
        default=2.0
    )

    parser.add_argument('--naive_task', dest='naive_task', action='store_true')
    parser.add_argument(
        '--no-naive_task',
        dest='naive_task',
        action='store_false')
    parser.set_defaults(naive_task=False)

    args = parser.parse_args()

    filters=['SoftCBF']
    initializer_type = 'scratch'
    out_folder, plot_tag, config_agent = None, None, None
    for filter_type in filters:
        with jax.default_device('gpu'):
            main(args.config_file, args.road_boundary, initializer_type=initializer_type, filter_type=filter_type)
