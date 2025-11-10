# CBF-DDP V2 with Soft Operators

This is a repository for using CBF-DDP with soft minimum and maximum operators in place of the hard operators that were originally used. All changes are contained in the main branch. The branch `singular_point_reproduction` is being used for exploration for local minima and singular phenomena for reach-avoid DDP.

## Usage instructions

There are two minor variations of the same methods, one intended to work with our own environments and one intended to work with the Brax-MJX interface. We rely on Anaconda for the Python environment setup. The `bicycle_jax_supported_env.yml` is best suited for our own simulators. The `brax_env.yml` is best suited for the MJX simulations. The user is free to mix the versions, but the code may not be reproducible perfectly.

### RaceCar 

The race car is based on the setup in `simulators/car/*.py`, with the dynamics implemented in `simulators/dynamics/*.py`. We provide the option for `Bicycle4D`, `Bicycle5D`, and `PointMass4D` dynamics for the racecar setup. The point mass dynamics repurposes the same configurations and costs for the point mass, with a mixing of the terms used to refer to the state variables. 

To run the race-car setup, examples are provided in `test_scripts_ilqr_task.sh` and `test_scripts_naive_task.sh`.

```
# Bicycle 4D reach-avoid - baseline line search, road boundary 2.5 in each direction, rollout stopping path.
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_1_bic4D.yaml -rb 2.5 -ls 'baseline' -sp 'rollout'

# Bicycle 5D reach-avoid - baseline line search, road boundary 3.0 in each direction, analytic stopping path.
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_1_bic5D.yaml -rb 3.0 -ls 'baseline' -sp 'analytic'

# Bicycle 5D reachability - trust region line search, road boundary 3.5 in each direction.
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D.yaml -rb 3.5 -ls 'trust_region_tune_margin'

# Bicycle 5D reachability - trust region line search, road boundary 3.5 in each direction, use naive task instead of default ILQR task policy.
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D.yaml -rb 3.5 -ls 'trust_region_tune_margin' --naive_task
```
#### Reachability Rollout with CBFDDP-SM on the Bicycle 5D dynamics.
<p align="center">
<img src="./videos/bic5d_reachability_softddpcbf.gif" width="480" height="400" />
</p>

#### Reach-avoid Rollout with CBFDDP-SM on the Bicycle 5D dynamics.
<p align="center">
<img src="./videos/bic5d_reachavoid_softddpcbf.gif" width="480" height="400" />
</p>

Provided are:

* four options for line search - `baseline`, `armijo`, `trust_region_constant_margin`, `trust_region_tune_margin`. We did not need anything more than the baseline method, as we initialized the solver with the solutions from the previous time.

* Naive task or ILQR task policy

* `analytic` stopping path or `rollout` stopping path based on reach-avoid

The test configs in `./test_configs/` provide the config options needed for tuning the filters. Tuning needs to be performed as explained in the supporting document to achieve the desired results. The environment offers the option to provide `Circle`, `Box` and `Ellipse` obstacles.

The choice of safety filter is hard-coded in `evaluate_soft_ddpcbf.py` with the options [`SoftCBF`, `CBF`, `LR`, `SoftLR`]—more instructions on how to setup each safety filter will be provided. The safety filter option may be provided as a command-line argument in a future release.

### 2D Planar vertical takeoff and landing

The PVTOL6D is based on the setup in `simulators/aerialV/*.py` with the dynamics implemented in `simulators/dynamics/*.py`. We perform the following test for the PVTOL

#### Reachability Rollout with CBFDDP-SM on the PVTOL 6D dynamics.
<p align="center">
<img src="./videos/pvtol_softddpcbf.gif" width="480" height="300" />
</p>

```
python evaluate_soft_ddpcbf_pvtol.py -cf ./test_configs/pvtol/test_config_circle_reach_obs1_pvtol6D.yaml
```

### Brax Reacher and MJX Barkour

In order to run reacher, use `python run_mjx_brax_simulations.py --env 'reacher'`. 

The seed is hardwired inside the code. Use it to your convenience to test solutions and compare with ours. The reacher setup is dependent on whether the 'linear' mode QP solver or the 'quadratic' mode QCQP constraint solver is used. There may be hardwired changes needed in `./brax_utils/configs/reacher.yaml`. The $\gamma$ factor is tuned to convenience and may beed to be re-tuned. The margin functions used are described in `brax_utils/costs/reacher_margin.py` 

In order to run barkour, use `python run_mjx_brax_simulations.py --env 'barkour'`. 

There may be hardwired changes needed in `./brax_utils/configs/barkour.yaml`. The margin functions used are described in `brax_utils/costs/barkour_margin.py` 

#### MJX-Brax Barkour with CBF-DDP
<p align="center">
<img src="./videos/barkour_reachability_ddpcbf_policy.gif" width="380" height="200" />
</p>

#### MJX-Brax Barkour with LR-DDP
<p align="center">
<img src="./videos/barkour_reachability_ddplr_policy.gif" width="380" height="200" />
</p>

#### Remark

The brax setup was not entirely conducive to retrieving a sufficient state and the derivative flow along this sufficient state. The author is projecting down the generalized pipeline state to generalized coordinates and performing uneasy maneuvers to retrieve the derivative flow of this reduced state.

# Citation

TBD
