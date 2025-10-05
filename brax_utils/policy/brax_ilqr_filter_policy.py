import jax
import copy

from jax import numpy as jp
from jax import Array as DeviceArray
from functools import partial
from typing import Optional, Dict, List
from .brax_ilqr_reachability_policy import iLQRBraxReachability
from simulators import BasePolicy
from simulators import barrier_filter_linear, barrier_filter_quadratic_two, barrier_filter_quadratic_eight
from brax_utils import WrappedBraxEnv
from simulators.costs.base_margin import BaseMargin

class iLQRBraxSafetyFilter(BasePolicy):

    def __init__(self, id: str, config, brax_envs: List[WrappedBraxEnv],
                 cost: BaseMargin) -> None:
        super().__init__(id, config)
        self.config = config

        self.filter_type = config.FILTER_TYPE
        self.constraint_type = config.CONSTRAINT_TYPE
        self.cbf_tol = config.CBF_DDP_TOLERANCE
        if self.filter_type == 'CBF':
            self.gamma = config.CBF_GAMMA
        elif self.filter_type == 'SoftCBF':
            self.gamma = config.SOFT_CBF_GAMMA
        else:
            self.gamma = None

        self.lr_threshold = config.LR_THRESHOLD

        self.filter_steps = 0
        self.barrier_filter_steps = 0

        self.brax_env = brax_envs[0]
        self.cost = cost

        self.dim_x = self.brax_env.dim_x
        self.dim_u = self.brax_env.dim_u
        self.N = config.N

        # Two ILQR solvers
        if self.config.COST_TYPE == "Reachability":
            self.solver_0 = iLQRBraxReachability(
                self.id, self.config, brax_envs[1], self.cost)
            # self.solver_1 = iLQRBraxReachability(
            #     self.id, self.config, brax_envs[2], self.cost)
            # self.solver_2 = iLQRBraxReachability(
            #     self.id, self.config, brax_envs[3], self.cost)

    @partial(jax.jit, static_argnames='self')
    def run_ddpcbf_iteration(self, args):
        state, control_cbf_cand, grad_x, reinit_controls, scaled_c, num_iters, constraint_violation, scaling_factor, cutoff, _, _, _, warmup = args
        num_iters = num_iters + 1
        # Extract information from solver for enforcing constraint
        _, B0u = self.brax_env.get_generalized_coordinates_grad(
            state, control_cbf_cand)
        control_correction = barrier_filter_linear(
            grad_x, B0u, scaled_c)

        control_cbf_cand_new = control_cbf_cand + control_correction
        state_imaginary = self.brax_env.step(
            state, control_cbf_cand_new
        )
        _, controls_new, marginopt_new, Vopt_new, is_inside_target_new, V_x_new = self.solver_0.get_action_jitted(obs=state_imaginary,
                                                    controls=jp.array(reinit_controls),
                                                    state=state_imaginary, 
                                                    warmup=warmup)
        # CBF constraint violation
        constraint_violation_new = Vopt_new - cutoff
        scaled_c_new = scaling_factor * constraint_violation_new

        return (state, control_cbf_cand_new, V_x_new, controls_new, scaled_c_new, 
                num_iters, constraint_violation_new, scaling_factor, cutoff,
                Vopt_new, marginopt_new, is_inside_target_new, warmup)

    @partial(jax.jit, static_argnames='self')
    def check_ddpcbf_iteration_continue(self, args):
        _, _, _, _, _, num_iters, constraint_violation, _, _, _, _, _, warmup = args
        return jp.logical_and(jp.logical_or(constraint_violation<self.cbf_tol, warmup), num_iters<2)

    def get_action_jitted(
        self, 
        obs: DeviceArray, 
        state: DeviceArray, 
        task_ctrl: DeviceArray,
        reinit_controls: DeviceArray,
        warmup: bool = False,
    ):
        @jax.jit
        def identity_fn(args):
            return args

        task_ctrl_jp = jp.array(task_ctrl)
        controls_initialize = jp.array(reinit_controls)
        control_0, controlsopt_0, marginopt_0, Vopt_0, is_inside_target_0, V_x_0 = self.solver_0.get_action_jitted(
                obs=obs, controls=controls_initialize, state=state, warmup=warmup)

        # Find safe policy from step 1
        state_imaginary = self.brax_env.step(
            state, task_ctrl_jp
        )
        boot_controls = jp.array(controlsopt_0)
        _,  controlsopt_next, marginopt_next, Vopt_next, is_inside_target_next, V_x_next = self.solver_0.get_action_jitted(
            obs=state_imaginary, controls=boot_controls, state=state_imaginary, warmup=warmup)

        # Iterations
        cutoff = self.gamma * Vopt_0

        # # Define initial state and initial performance policy
        control_cbf_cand = jp.array(task_ctrl)
        num_iters = 0
        # Checking CBF constraint violation
        constraint_violation = Vopt_next - cutoff
        scaled_c = constraint_violation
        # Scaling parameter
        scaling_factor = 1.2

        # unroll two iterations
        num_iters = 0
        if constraint_violation<self.cbf_tol or warmup:
            args = (state, control_cbf_cand, V_x_next, controlsopt_next, scaled_c, num_iters, constraint_violation, scaling_factor, 
                                                        cutoff, Vopt_next, marginopt_next, is_inside_target_next, warmup) 
            (state, control_cbf_cand, grad_x, reinit_controls,
                scaled_c, num_iters, constraint_violation, 
                    scaling_factor, cutoff, Vopt_next, marginopt_next, is_inside_target_next, warmup) = self.run_ddpcbf_iteration(args)

        if constraint_violation<self.cbf_tol or warmup:
            args = (state, control_cbf_cand, grad_x, reinit_controls, scaled_c, num_iters, constraint_violation, scaling_factor, 
                                                        cutoff, Vopt_next, marginopt_next, is_inside_target_next, warmup)
            (state, control_cbf_cand, grad_x, reinit_controls, 
                scaled_c, num_iters, constraint_violation, 
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

        solver_info = {
            'controls': reinit_controls,
            'reinit_controls': reinit_controls,
            'Vopt': Vopt_0,
            'marginopt': marginopt_0,
            'is_inside_target': is_inside_target_0,
            'num_iters': num_iters,
            'Vopt_next': Vopt_next,
            'marginopt_next': marginopt_next,
            'is_inside_target_next': is_inside_target_next,
            'safe_opt_control': jp.array(control_0),
            'mark_barrier_filter': num_iters>0,
            'mark_complete_filter': False,
        }

        return control_cbf_cand.ravel(), solver_info     

    def get_action(
        self, 
        obs: DeviceArray, 
        state: DeviceArray, 
        task_ctrl: DeviceArray,
        prev_sol: Optional[Dict] = None, 
        prev_ctrl = None, 
        warmup=False,
    ):

        task_ctrl_jp = jp.array(task_ctrl)

        # Find safe policy from step 0
        if prev_sol is not None:
            controls_initialize = jp.array(prev_sol['reinit_controls'])
        else:
            controls_initialize = jp.zeros((self.dim_u, self.N))

        if prev_sol is None or prev_sol['resolve']:
            control_0, solver_info_0 = self.solver_0.get_action(
                obs=obs, controls=controls_initialize, state=state)
        else:
            # Potential source of acceleration. We don't need to resolve both ILQs as we can reuse
            # solution from previous time. - Unused currently.
            solver_info_0 = prev_sol['bootstrap_next_solution']
            control_0 = solver_info_0['controls'][:, 0]
            # Closed loop solution
            #solver_info_0['controls'] = jp.array( solver_info_0['reinit_controls'] )
            #solver_info_0['states'] = jp.array( solver_info_0['reinit_states'] )
            #solver_info_0['Vopt'] = solver_info_0['Vopt_next']
            #solver_info_0['marginopt'] = solver_info_0['marginopt_next']
            #solver_info_0['is_inside_target'] = solver_info_0['is_inside_target_next']

        solver_info_0['safe_opt_ctrl'] =  jp.array(control_0)
        solver_info_0['task_ctrl'] = jp.array(task_ctrl)

        solver_info_0['mark_barrier_filter'] = False
        solver_info_0['mark_complete_filter'] = False
        # Find safe policy from step 1
        state_imaginary = self.brax_env.step(
            state, task_ctrl_jp
        )
        boot_controls = jp.array(solver_info_0['controls'])

        _, solver_info_1 = self.solver_1.get_action(
            obs=state_imaginary, controls=boot_controls, state=state_imaginary)

        solver_info_0['Vopt_next'] = solver_info_1['Vopt']
        solver_info_0['marginopt_next'] = solver_info_1['marginopt']
        solver_info_0['is_inside_target_next'] = solver_info_1['is_inside_target']

        cutoff = self.gamma * solver_info_0['Vopt']

        control_cbf_cand = task_ctrl

        solver_initial = jp.zeros((self.dim_u,))
        if prev_sol is not None:
            solver_initial = (prev_ctrl - control_cbf_cand)

        # Define initial state and initial performance policy
        control_cbf_cand_jp = jp.array(control_cbf_cand)
        num_iters = 0

        # Setting tolerance to zero does not cause big improvements at the
        # cost of more unnecessary looping
        cbf_tol = self.cbf_tol
        # Conditioning parameter out of abundance of caution
        eps_reg = 1e-8

        # Checking CBF constraint violation
        constraint_violation = solver_info_1['Vopt'] - cutoff
        scaled_c = constraint_violation

        # Scaling parameter
        scaling_factor = 1.2

        # Exit loop once CBF constraint satisfied or maximum iterations
        # violated
        control_bias_term = jp.zeros((self.dim_u,))
        while((constraint_violation < cbf_tol or warmup) and num_iters < 2):
            num_iters = num_iters + 1

            # Extract information from solver for enforcing constraint
            grad_x = solver_info_1['grad_x']
            _, B0u = self.brax_env.get_generalized_coordinates_grad(
                state, control_cbf_cand_jp)

            if self.constraint_type == 'quadratic':
                grad_xx = solver_info_1['grad_xx']

                # Compute P, p
                P = B0u.T @ grad_xx @ B0u - eps_reg * jp.eye(self.dim_u)

                # For some reason, hessian from jax is only approxiamtely
                # symmetric
                P = 0.5 * (P + P.T)
                p = grad_x.T @ B0u
                # Controls improvement direction
                # limits = np.array( [[self.dyn.ctrl_space[0, 0] - control_cbf_cand[0], self.dyn.ctrl_space[0, 1] - control_cbf_cand[0]],
                #          [self.dyn.ctrl_space[1, 0] - control_cbf_cand[1], self.dyn.ctrl_space[1, 1] - control_cbf_cand[1]]] )
                if self.dim_u==2:
                    control_correction = barrier_filter_quadratic_two(
                        P, p, scaled_c, initialize=solver_initial, control_bias_term=control_bias_term)
                elif self.dim_u==8:
                    control_correction = barrier_filter_quadratic_eight(
                        P, p, scaled_c, initialize=solver_initial, control_bias_term=control_bias_term)                
            elif self.constraint_type == 'linear':
                control_correction = barrier_filter_linear(
                    grad_x, B0u, scaled_c)

            control_bias_term = control_bias_term + control_correction
            control_cbf_cand_jp = control_cbf_cand_jp + control_correction
            # control_cbf_cand_jp = jp.clip(control_cbf_cand_jp, self.brax_env.action_limits[0], self.brax_env.action_limits[1])

            # Restart from current point and run again
            solver_initial = (prev_ctrl - control_cbf_cand)

            state_imaginary = self.brax_env.step(
                state, control_cbf_cand_jp
            )
            _, solver_info_1 = self.solver_2.get_action(obs=state_imaginary,
                                                        controls=jp.array(
                                                            solver_info_1['controls']),
                                                        state=state_imaginary)
            solver_info_0['Vopt_next'] = solver_info_1['Vopt']
            solver_info_0['marginopt_next'] = solver_info_1['marginopt']
            solver_info_0['is_inside_target_next'] = solver_info_1['is_inside_target']

            # CBF constraint violation
            constraint_violation = solver_info_1['Vopt'] - cutoff
            scaled_c = scaling_factor * constraint_violation

        if solver_info_1['Vopt'] > 0 or warmup:
            if num_iters > 0:
                self.barrier_filter_steps += 1
                solver_info_0['mark_barrier_filter'] = True
            solver_info_0['resolve'] = False
            solver_info_0['num_iters'] = num_iters
            solver_info_0['bootstrap_next_solution'] = solver_info_1
            solver_info_0['reinit_controls'] = jp.array(
                solver_info_1['controls'])

            return control_cbf_cand_jp.ravel(), solver_info_0

        self.filter_steps += 1
        # Safe policy
        solver_info_0['resolve'] = True
        solver_info_0['num_iters'] = num_iters
        solver_info_0['reinit_controls'] = jp.zeros((self.dim_u, self.N))
        solver_info_0['reinit_controls'] = solver_info_0['reinit_controls'].at[:, 0:self.N - 1].set(
            solver_info_0['controls'][:, 1:self.N])
        solver_info_0['mark_complete_filter'] = True
        safety_control = solver_info_0['controls'][:, 0]

        return safety_control, solver_info_0
