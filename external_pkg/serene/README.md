# SERENE

This is the code for SERENE from the [SparsE Reward Exploration via Novelty search and Emitters](https://arxiv.org/abs/2102.03140) paper published at GECCO 2021.

The code is based on the [Novelty Search Gym repo](https://github.com/GPaolo/novelty_search_gym).

The environments are included in the repo in the `environments/assets` folder.

---
To install run:
```
pipenv shell --three
python setup.py install
```

## Performing an experiment
To run the algorithm you just need to launch:
```bash
ipython script/run_experiment.py
```

If you want to change the experiment parameters, go to: `parameters.py`

You can also specify some of the parameters on the command line. If you want to check which ones, launch:
```bash
ipython script/run_experiment.py -- -h
```
---

For each generation the script saves: `population`, `offsprings`, `archive` as pkl files in
folders whose name is formatted as:
`<env_name>_<exp_type>/<year>_<month>_<day>_<hour>:<minute>_<random_seed_used>`.
The time corresponds to when the experiment has been launched.

In this folder you find the parameters, in a file called: `_params.json`,
the archive for each generation, in files called: `archive_gen_<generation>.pkl`,
the population for each generation, in files called: `population_gen_<generation>.pkl`,
and the offsprings for each generation, in files called: `offsprings_gen_<generation>.pkl`.

The final  generation is also save as `population_final.pkl`, etc.

## Evaluating the archive
Once the experiment is finished, if you want to study the behavior descriptors of the
agents in the archive you have to evaluate the archive first by running it in the
environment and saving the trajectories of images and observations.

You can do that by launching:
```bash
ipython scripts/evaluate_archive.py -- -e EXPERIMENT_PATH
```

This script will evaluate the archive of the given generation (The default is the last).

As for the `run_experiment.py` script, you can change the parameters or provide some on the command line.

Once the evaluation is done, the trajectories will be saved in the experiment folder, inside a folder called `analyzed_data`.

If you use the `-abd` flag when launching the `evaluate_archive.py` script, the code will generate a collection of the behavior descriptors of all the agents.
This collection can be used through the `analysis/search_video.py` script to generate a video of the exploration performed by the algorithm.

### Plotting the results
Finally you can plot your results by using the jupyter notebook `archive_analysis` located in the `analysis` folder.

# Extending it
Being based on the Novelty Search Gym repo, this code is fairly modular and can be extended easily, both by adding new kind of experiments/metrics or by adding new gym environments.

### Adding environments
If you want to add an environment you have to do:
1. Add the gym environment in the `environments/assets` folder
2. Register the environment in the `environments/environments.py` file as an entry in the `registered_envs` dictionary
3. Add an input formatter (and in case also an output formatter) in the `environments/io_formatters.py` files.
These formatters are used to interface the environment with the controllers.
    * The input formatters prepares the observation to be fed to the controller.
    * The output formatters takes the controller output and formats it as an action for the environment.
4. Add the ground truth behavior descriptor in the `analysis/gt_behavior_descriptors.py` and in the `get_metrics` function in `analysis/evaluate_archive.py`.
5. Add the observations extraction function from the trajectory in the `core/behavior_descriptors/trajectory_to_observations.py`.

### Adding experiments
If you want to add an experiment type you have to:
1. The behavior descriptor in the `core/behavior_descriptors/behavior_descriptor.py`. You have to both add the
option in the `__init__` of the class and the actual descriptor function as a member function of the class.
2. If you want to define an evolution algorithm you can do it in `core/evolvers`
   Then you have to add it in the `core/evolvers/__init__.py` and in the `core/searcher.py`
4. Add the name you chose for the experiment in the list of possible choices in the parser in `scripts/run_experiment.py`

---

If you find this code useful and you use it in a project, please cite it:
```
@article{paolo2021sparse,
  title={Sparse Reward Exploration via Novelty Search and Emitters},
  author={Paolo, Giuseppe and Coninx, Alexandre and Doncieux, Stephane and Laflaqui{\`e}re, Alban},
  journal={arXiv preprint arXiv:2102.03140},
  year={2021},
  howpublished = {\url{https://github.com/GPaolo/novelty_search_gym}}
}
```
