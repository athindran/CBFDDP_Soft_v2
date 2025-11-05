# Soft-DDPCBF

This is a repository for using CBF-DDP with soft minimum and maximum operators in place of the hard operators that were originally used. All changes are contained in the main branch. The branch `singular_point_reproduction` is being used for exploration for local minima and singular phenomena for reach-avoid DDP.

### Reachability Rollout with CBFDDP-SM on the Bicycle 5D dynamics.
<p align="center">
<img src="./videos/bic5d_reachability_softddpcbf.gif" width="480" height="400" />
</p>

### Reach-avoid Rollout with CBFDDP-SM on the Bicycle 5D dynamics.
<p align="center">
<img src="./videos/bic5d_reachavoid_softddpcbf.gif" width="480" height="400" />
</p>

### Reachability Rollout with CBFDDP-SM on the PVTOL 6D dynamics.
<p align="center">
<img src="./videos/pvtol_softddpcbf.gif" width="480" height="300" />
</p>

### MJX-Brax Barkour with CBF-DDP
<p align="center">
<img src="./videos/barkour_reachability_ddpcbf_policy.gif" width="380" height="200" />
</p>

### MJX-Brax Barkour with LR-DDP
<p align="center">
<img src="./videos/barkour_reachability_ddplr_policy.gif" width="380" height="200" />
</p>

## Usage intructions

There are two minor variations, one intended to work with our own environments and one intended to work with the Brax-MJX interface. We rely on Anaconda for the python environment setup. The `bicycle_jax_supported_env.yml` is best suited for our own simulators. The `brax_env.yml` is best uited for the MJX simulations. The user is free to mix the versions but the code may not be reproducible perfectly.

### RaceCar 

The race-car is based on the setup in `simulators/car/*.py` with the dynamics implemented in `simulators/dynamics/*.py`. We provide the option for `Bicycle4D`, `Bicycle5D` and `PointMass4D` dynamics for the racecar setup. The point mass dynamics re-purposes the same configs and costs for the pointmass with mixing of the terms used to refer to the state variables. 

In order to run the race-car setup, examples are provided in `test_scripts_ilqr_task.sh` and `test_scripts_naive_task.sh`.

```
# Bicycle 4D reachavoid - baseline line search, road boundary 2.5 in each direction, rollout stopping path.
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_1_bic4D.yaml -rb 2.5 -ls 'baseline' -sp 'rollout'

# Bicycle 5D reachavoid - baseline line search, road boundary 3.0 in each direction, analytic stopping path.
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_1_bic5D.yaml -rb 3.0 -ls 'baseline' -sp 'analytic'

# Bicycle 5D reachability - trust region line search, road boundary 3.5 in each direction.
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D.yaml -rb 3.5 -ls 'trust_region_tune_margin'

# Bicycle 5D reachability - trust region line search, road boundary 3.5 in each direction, use naive task instead of default ILQR task policy.
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D.yaml -rb 3.5 -ls 'trust_region_tune_margin' --naive_task
```

Provided are:

* four options for line search - baseline, armijo, trust_region_constant_margin, trust_region_tune_margin. We did not need anything more than the baseline method as we initialize the solver with the solutions from the previous time at each time.

* Naive task or ILQR task policy

* `analytic` stopping path or `rollout` stopping path based on reach-avoid

The test configs in `./test_configs/` provide the config options needed for tuning the filters. Tuning needs to done as explained in the supporting document for obtaining desired results. The environment provides the option to provide `Circle`, `Box` and `Ellipse`.

The choice of safety filter is hard-coded in `evaluate_soft_ddpcbf.py` with the options [`SoftCBF`, `CBF`, `LR`, `SoftLR`]. More instructions on how to setup each safety filter will be provided. The safety filter option may be provided as a command line argument later.

### 2D Planar vertical takeoff and landing

The PVTOL6D is based on the setup in `simulators/aerialV/*.py` with the dynamics implemented in `simulators/dynamics/*.py`. We perform the following test for the PVTOL


```
python evaluate_soft_ddpcbf_pvtol.py -cf ./test_configs/pvtol/test_config_circle_reach_obs1_pvtol6D.yaml
```

### Brax Reacher and MJX Barkour

TBD
