# Quality Diversity for Grasping in Robotics


This code allows the generation of repertoires of diverse and high performing grasping trajectories, with Quality-Diversity methods.

It also allow to reproduce results from the paper : "Quality Diversity for Grasping in Robotics" [*under review*]

**Status:** *refactoring in progress*

## Before starting

### Recommandations

* use python 3.8.x or 3.9.x: versions >3.10.x cause problems with scoop
* scoop parallelization requires multiple cpu cores; lower number of threads => slower exploration

## Install

```
python3 -m venv qdg
source qdg/bin/activate
```
```
pip3 install -r requirements.txt
pip3 install -e gym_envs
pip3 install -e .
```

## Examples

### Trajectory generation

Debug mode, to visualize each evaluation: 
```
python3 apply_qd_grasp.py -a me_scs -r kuka_ik -o ycb_power_drill -nbr 2000 -d
```

Longer run:
```
python3 -m scoop apply_qd_grasp.py -a me_scs -r kuka_ik -o ycb_power_drill -nbr 25000
```


### Visualizing output

To replay successful trajectories from a completed run:
```
python3 visualization/replay_trajectories_grasp_at_touch.py -r path_to_run_folder/
```
Visualise the approach trajectories:
```
...
```
Visualise the success archive as fitness heatmap:
```
...
```


## Ressources: 

### Quality diversity
https://quality-diversity.github.io/

### Compared methods
...

## TODO :
* clean legacy files
* clean up the pipeline
* refactore gym_grab
* ...






