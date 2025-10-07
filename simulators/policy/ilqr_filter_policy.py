from typing import Optional, Dict
import time

from jax import numpy as jp
from jax import Array as DeviceArray
from jax import block_until_ready

import jax

from functools import partial
from .ilqr_reachavoid_policy import iLQRReachAvoid
from .ilqr_reachability_policy import iLQRReachability
from .base_policy import BasePolicy
from .solver_utils import barrier_filter_linear, barrier_filter_quadratic_two, barrier_filter_quadratic_eight
from simulators.dynamics.base_dynamics import BaseDynamics
from simulators.costs.base_margin import BaseMargin

class iLQRSafetyFilter(BasePolicy):

    def __init__(self, id: str, config, dyn: BaseDynamics,
                 cost: BaseMargin) -> None:
        super().__init__(id, config)
        self.filter_type = config.FILTER_TYPE
        self.constraint_type = config.CONSTRAINT_TYPE
        if self.filter_type == 'CBF':
            self.gamma = config.CBF_GAMMA
        elif self.filter_type == 'SoftCBF':
            self.gamma = config.SOFT_CBF_GAMMA
        else:
            self.gamma = None

        self.lr_threshold = config.LR_THRESHOLD
        self.filter_steps = 0
        self.barrier_filter_steps = 0
        self.dyn = dyn
        self.rollout_dyn_0 = dyn
        self.cost = cost
        self.dim_x = dyn.dim_x
        self.dim_u = dyn.dim_u
        self.N = config.N

        # Three ILQR solvers
        if config.COST_TYPE == "Reachavoid":
            self.solver_0 = iLQRReachAvoid(
                self.id, config, self.rollout_dyn_0, self.cost)
            # self.solver_0 = iLQRReachAvoid(
            #     self.id, config, self.rollout_dyn_1, self.cost)
            # self.solver_0 = iLQRReachAvoid(
            #     self.id, config, self.rollout_dyn_1, self.cost)
        elif config.COST_TYPE == "Reachability":
            self.solver_0 = iLQRReachability(
                self.id, config, self.rollout_dyn_0, self.cost)
            # self.solver_0 = iLQRReachability(
            #     self.id, config, self.rollout_dyn_1, self.cost)
            # self.solver_0 = iLQRReachability(
            #     self.id, config, self.rollout_dyn_1, self.cost)

    @partial(jax.jit, static_argnames='self')
    def run_ddpcbf_iteration(self, args):
        state, control_cbf_cand, grad_x, reinit_controls, _, scaled_c, num_iters, scaling_factor, cutoff, _, _, _, warmup = args
        num_iters = num_iters + 1
        # Extract information from solver for enforcing constraint
        _, B0 = self.dyn.get_jacobian(
            state[:, jp.newaxis], control_cbf_cand[:, jp.newaxis])
        control_correction = barrier_filter_linear(
            grad_x, B0[:, :, 0], scaled_c)

        control_cbf_cand_next = control_cbf_cand + control_correction
        state_imaginary, control_cbf_cand_next = self.dyn.integrate_forward_jax(
            state, control_cbf_cand_next
        )
        _, controls_next, statesopt_next, marginopt_next, Vopt_next, is_inside_target_next, V_x_next = self.solver_0.get_action_jitted(obs=state_imaginary,
                                                    controls=jp.array(reinit_controls),
                                                    state=state_imaginary, 
                                                    warmup=warmup)
        # CBF constraint violation
        constraint_violation_next = jp.minimum(Vopt_next - cutoff, 0.0)
        scaled_c_next = scaling_factor * constraint_violation_next

        return (state, control_cbf_cand_next, V_x_next, controls_next, statesopt_next, scaled_c_next, 
                num_iters, scaling_factor, cutoff,
                Vopt_next, marginopt_next, is_inside_target_next, warmup)

    @partial(jax.jit, static_argnames='self')
    def get_action_jitted(
        self, 
        obs: DeviceArray, 
        state: DeviceArray, 
        task_ctrl: DeviceArray,
        reinit_controls: DeviceArray,
        warmup=False,
    ):
        #start_time = time.time()
        task_ctrl = task_ctrl.ravel()
        control_0, controlsopt_0, statesopt_0, marginopt_0, Vopt_0, is_inside_target_0, _ = self.solver_0.get_action_jitted(
            obs=obs, controls=reinit_controls, state=state, warmup=warmup)
        # Find safe policy from step 1
        state_imaginary, task_ctrl = self.dyn.integrate_forward_jax(
            state, task_ctrl
        )
        boot_controls = jp.array(controlsopt_0)
        _,  controlsopt_next, statesopt_next, marginopt_next, Vopt_next, is_inside_target_next, V_x_next = self.solver_0.get_action_jitted(
            obs=state_imaginary, controls=boot_controls, state=state_imaginary, warmup=warmup)

        # Iterations
        cutoff = self.gamma * Vopt_0
        # # Define initial state and initial performance policy
        control_cbf_cand = jp.array(task_ctrl)
        # # Checking CBF constraint violation
        scaling_factor = 0.8
        mark_barrier_filter = (Vopt_next - cutoff < 0.0)
        constraint_violation = jp.minimum(Vopt_next - cutoff, 0.0)
        scaled_c = constraint_violation

        # # unroll two iterations
        num_iters = 0
        args = (state, control_cbf_cand, V_x_next, controlsopt_next, statesopt_next, scaled_c, num_iters, scaling_factor, 
                                                    cutoff, Vopt_next, marginopt_next, is_inside_target_next, warmup)
        args = self.run_ddpcbf_iteration(args)
        (state, control_cbf_cand, grad_x, reinit_controls, statesopt_next,  
            scaled_c, num_iters, 
                scaling_factor, cutoff, Vopt_next, marginopt_next, is_inside_target_next, warmup) = self.run_ddpcbf_iteration(args)

        # run_ddpcbf_iteration = lambda args: self.run_ddpcbf_iteration(args)
        # args = (state, control_cbf_cand, V_x_next, controlsopt_next, scaled_c, num_iters, constraint_violation, scaling_factor, 
        #                                                  cutoff, Vopt_next, marginopt_next, is_inside_target_next, warmup)
        # args = jax.lax.cond(jp.logical_or(constraint_violation<self.cbf_tol, warmup), run_ddpcbf_iteration, identity_fn, args)
        # (state, control_cbf_cand, grad_x, reinit_controls, 
        #     scaled_c, num_iters, constraint_violation, 
        #         scaling_factor, cutoff, Vopt_next, marginopt_next, is_inside_target_next, warmup) = args
        # args = jax.lax.cond(jp.logical_or(constraint_violation<self.cbf_tol, warmup), run_ddpcbf_iteration, identity_fn, args)
        # (state, control_cbf_cand, grad_x, reinit_controls, 
        #     scaled_c, num_iters, constraint_violation, 
        #         scaling_factor, cutoff, Vopt_next, marginopt_next, is_inside_target_next, warmup) = args

        # Exit loop once CBF constraint satisfied or maximum iterations
        # violated
        # check_ddpcbf_iteration_continue = lambda args: self.check_ddpcbf_iteration_continue(args)
        # run_ddpcbf_iteration = lambda args: self.run_ddpcbf_iteration(args)
        # args = (state, control_cbf_cand, V_x_next, controlsopt_next, scaled_c, num_iters, constraint_violation, scaling_factor, 
        #                                                  cutoff, Vopt_next, marginopt_next, is_inside_target_next, warmup) 
        # (state, control_cbf_cand, grad_x, reinit_controls, 
        #     scaled_c, num_iters, constraint_violation, 
        #         scaling_factor, cutoff, Vopt_next, marginopt_next, is_inside_target_next, warmup) = jax.lax.while_loop(check_ddpcbf_iteration_continue, 
        #                         run_ddpcbf_iteration, args)
        #control_cbf_cand = jax.block_until_ready(control_cbf_cand)
        #self.barrier_filter_steps += mark_barrier_filter
        solver_info = {
            'states': statesopt_next,
            'controls': reinit_controls,
            'reinit_controls': reinit_controls,
            'Vopt': Vopt_0,
            'marginopt': marginopt_0,
            'num_iters': num_iters,
            'Vopt_next': Vopt_next,
            'marginopt_next': marginopt_next,
            'is_inside_target_next': is_inside_target_next,
            'safe_opt_ctrl': control_0,
            'task_ctrl': task_ctrl,
            'mark_barrier_filter': mark_barrier_filter,
            'mark_complete_filter': False,
            'grad_x': grad_x,
            #'process_time': time.time() - start_time,
            'resolve': False,
            'barrier_filter_steps': 0,
            'filter_steps': 0,
        }

        return control_cbf_cand.ravel(), solver_info
