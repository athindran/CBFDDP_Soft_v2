from typing import Tuple, Optional, Dict
import time
import copy
import jax
from jax import numpy as jp
from jax import Array as DeviceArray
from functools import partial

from .base_policy import BasePolicy
from simulators.dynamics.base_dynamics import BaseDynamics
from simulators.costs.base_margin import BaseMargin


class iLQR(BasePolicy):

    def __init__(
        self, id: str, config, dyn: BaseDynamics, cost: BaseMargin
    ) -> None:
        super().__init__(id, config)
        self.policy_type = "iLQR"
        self.dyn = copy.deepcopy(dyn)
        self.cost = copy.deepcopy(cost)
        self.line_search = getattr(config, "LINE_SEARCH", 'baseline')

        # iLQR parameters
        self.dim_x = dyn.dim_x
        self.dim_u = dyn.dim_u
        self.N = config.N
        self.order = config.ORDER
        self.max_iter = config.MAX_ITER
        self.tol = 1e-5  # ILQR update tolerance.
        self.eps = getattr(config, "EPS", 1e-6)
        self.min_alpha = 1e-12

    @partial(jax.jit, static_argnames='self')
    def run_ddp_iteration(self, args):
        states, controls, J, _, _, _, num_iters, warmup = args
        # We need cost derivatives from 0 to N-1, but we only need dynamics
        # jacobian from 0 to N-2.
        c_x, c_u, c_xx, c_uu, c_ux = self.cost.get_derivatives(
            states, controls)
        fx, fu = self.dyn.get_jacobian(states[:, :-1], controls[:, :-1])
        K_closed_loop, k_open_loop = self.backward_pass(
            c_x=c_x, c_u=c_u, c_xx=c_xx, c_uu=c_uu, c_ux=c_ux, fx=fx, fu=fu
        )
        # Choose the best alpha scaling using appropriate line search methods
        alpha_chosen = 1.0
        alpha_chosen = self.baseline_line_search(states, controls, K_closed_loop, k_open_loop, J)

        states, controls, J_new = self.forward_pass(states, controls, K_closed_loop, k_open_loop, alpha_chosen)
        cvg_tolerance = jp.abs((J - J_new) / J)
        status = 0
        status = jax.lax.cond((cvg_tolerance<self.tol) & (J_new<=J), lambda: 1, lambda: status)
        status = jax.lax.cond((status!=1) & (alpha_chosen<self.min_alpha), lambda: 2, lambda: status)

        num_iters += 1
        return (states, controls, J_new,
                alpha_chosen, cvg_tolerance, status, num_iters, warmup)

    @partial(jax.jit, static_argnames='self')
    def check_ddp_iteration_continue(self, args):
        _,  _,  J_new,  _,  _, status, num_iters, warmup = args
        return jp.logical_and(jp.logical_or(status==0, warmup), num_iters<self.max_iter)

    @partial(jax.jit, static_argnames='self')
    def get_action_jitted(
        self, obs: DeviceArray, state: DeviceArray, controls: DeviceArray, warmup: bool = False
    ):
        status = 0
        self.min_alpha = 9e-10
        states, controls = self.rollout_nominal(
            state, jp.array(controls)
        )
        J = self.cost.get_traj_cost(states, controls)

        run_ddp_iteration = lambda args: self.run_ddp_iteration(args)
        check_ddp_iteration_continue = lambda args: self.check_ddp_iteration_continue(args)

        num_iters = 0
        (states, controls, J, alpha_chosen, 
                cvg_tolerance, status, num_iters, warmup) = jax.lax.while_loop(check_ddp_iteration_continue, run_ddp_iteration, 
                                            (states, controls, J, 1.0, 1.0, 0, num_iters, warmup))

        solver_info = dict(
            states=states, controls=controls, t_process=0.0, status=status, J=J
        )
        return controls[:, 0], solver_info

    @partial(jax.jit, static_argnames='self')
    def baseline_line_search(self, states, controls, K_closed_loop, k_open_loop, J, beta=0.5, alpha_initial=1.0):
        alpha = alpha_initial
        J_new = -jp.inf

        @jax.jit
        def run_forward_pass(args):
            states, controls, K_closed_loop, k_open_loop, alpha, J, J_new = args
            alpha = beta*alpha
            _, _, J_new = self.forward_pass(states, controls, K_closed_loop, k_open_loop, alpha)
            return states, controls, K_closed_loop, k_open_loop, alpha, J, J_new

        @jax.jit
        def check_terminated(args):
            _, _, _, _, alpha, J, J_new = args
            return jp.logical_and( alpha>self.min_alpha, J_new>J )
        
        states, controls, K_closed_loop, k_open_loop, alpha, J, J_new = jax.lax.while_loop(check_terminated, run_forward_pass, (states, controls, K_closed_loop, k_open_loop, alpha, J, J_new))

        return alpha

    @partial(jax.jit, static_argnames='self')
    def forward_pass(
        self, nominal_states: DeviceArray, nominal_controls: DeviceArray,
        K_closed_loop: DeviceArray, k_open_loop: DeviceArray, alpha: float
    ) -> Tuple[DeviceArray, DeviceArray, float]:
        # We seperate the rollout and cost explicitly since get_cost might rely on
        # other information, such as env parameters (track), and is difficult for
        # jax to differentiate.
        X, U = self.rollout(
            nominal_states, nominal_controls, K_closed_loop, k_open_loop, alpha
        )
        J = self.cost.get_traj_cost(X, U)
        return X, U, J

    @partial(jax.jit, static_argnames='self')
    def rollout(
        self, nominal_states: DeviceArray, nominal_controls: DeviceArray,
        K_closed_loop: DeviceArray, k_open_loop: DeviceArray, alpha: float
    ) -> Tuple[DeviceArray, DeviceArray]:

        @jax.jit
        def _rollout_step(i, args):
            X, U = args
            u_fb = jp.einsum(
                "ik,k->i", K_closed_loop[:, :, i], (X[:, i] - nominal_states[:, i])
            )
            u = nominal_controls[:, i] + alpha * k_open_loop[:, i] + u_fb
            x_nxt, u_clip = self.dyn.integrate_forward_jax(X[:, i], u)
            X = X.at[:, i + 1].set(x_nxt)
            U = U.at[:, i].set(u_clip)
            return X, U

        X = jp.zeros((self.dim_x, self.N))
        U = jp.zeros((self.dim_u, self.N))  # Assumes the last ctrl are zeros.
        X = X.at[:, 0].set(nominal_states[:, 0])

        X, U = jax.lax.fori_loop(0, self.N - 1, _rollout_step, (X, U))
        # Last control is only used for control cost - relevant for PVTOL with setpoint.
        U = U.at[:, self.N - 1].set(jp.asarray(U[:, self.N - 2]))

        return X, U

    @partial(jax.jit, static_argnames='self')
    def rollout_nominal(
        self, initial_state: DeviceArray, controls: DeviceArray
    ) -> Tuple[DeviceArray, DeviceArray]:

        @jax.jit
        def _rollout_nominal_step(i, args):
            X, U = args
            x_nxt, u_clip = self.dyn.integrate_forward_jax(X[:, i], U[:, i])
            X = X.at[:, i + 1].set(x_nxt)
            U = U.at[:, i].set(u_clip)
            return X, U

        X = jp.zeros((self.dim_x, self.N))
        X = X.at[:, 0].set(initial_state)
        X, U = jax.lax.fori_loop(
            0, self.N - 1, _rollout_nominal_step, (X, controls)
        )
        return X, U

    @partial(jax.jit, static_argnames='self')
    def backward_pass(
        self, c_x: DeviceArray, c_u: DeviceArray, c_xx: DeviceArray,
        c_uu: DeviceArray, c_ux: DeviceArray, fx: DeviceArray, fu: DeviceArray
    ) -> Tuple[DeviceArray, DeviceArray]:
        """
        Jitted backward pass looped computation.

        Args:
            c_x (DeviceArray): (dim_x, N)
            c_u (DeviceArray): (dim_u, N)
            c_xx (DeviceArray): (dim_x, dim_x, N)
            c_uu (DeviceArray): (dim_u, dim_u, N)
            c_ux (DeviceArray): (dim_u, dim_x, N)
            fx (DeviceArray): (dim_x, dim_x, N-1)
            fu (DeviceArray): (dim_x, dim_u, N-1)

        Returns:
            Ks (DeviceArray): gain matrices (dim_u, dim_x, N - 1)
            ks (DeviceArray): gain vectors (dim_u, N - 1)
        """

        @jax.jit
        def backward_pass_looper(i, _carry):
            V_x, V_xx, ks, Ks = _carry
            n = self.N - 2 - i

            Q_x = c_x[:, n] + fx[:, :, n].T @ V_x
            Q_u = c_u[:, n] + fu[:, :, n].T @ V_x
            Q_xx = c_xx[:, :, n] + fx[:, :, n].T @ V_xx @ fx[:, :, n]
            Q_ux = c_ux[:, :, n] + fu[:, :, n].T @ V_xx @ fx[:, :, n]
            Q_uu = c_uu[:, :, n] + fu[:, :, n].T @ V_xx @ fu[:, :, n]

            Q_uu_inv = jp.linalg.inv(Q_uu + reg_mat)

            Ks = Ks.at[:, :, n].set(-Q_uu_inv @ Q_ux)
            ks = ks.at[:, n].set(-Q_uu_inv @ Q_u)

            # The terms will cancel out but for the regularization added. 
            # See https://studywolf.wordpress.com/2016/02/03/the-iterative-linear-quadratic-regulator-method/ and references therein.
            # V_x = Q_x + Q_ux.T @ ks[:, idx]
            # V_xx = Q_xx + Q_ux.T @ Ks[:, :, idx]

            V_x = Q_x + Ks[:, :, n].T @ Q_u + Q_ux.T @ ks[:, n] + Ks[:, :, n].T @ Q_uu @ ks[:, n]
            V_xx = (Q_xx + Ks[:, :, n].T @ Q_ux + Q_ux.T @ Ks[:, :, n]
                    + Ks[:, :, n].T @ Q_uu @ Ks[:, :, n])

            return V_x, V_xx, ks, Ks

        # Initializes.
        Ks = jp.zeros((self.dim_u, self.dim_x, self.N - 1))
        ks = jp.zeros((self.dim_u, self.N - 1))
        V_x = c_x[:, -1]
        V_xx = c_xx[:, :, -1]
        reg_mat = self.eps * jp.eye(self.dim_u)

        V_x, V_xx, ks, Ks = jax.lax.fori_loop(
            0, self.N - 1, backward_pass_looper, (V_x, V_xx, ks, Ks)
        )
        return Ks, ks
