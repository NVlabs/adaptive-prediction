# Expanding the Deployment Envelope of Behavior Prediction via Adaptive Meta-Learning #
This repository contains the code for [Expanding the Deployment Envelope of Behavior Prediction via Adaptive Meta-Learning](https://arxiv.org/abs/2209.11820) by Boris Ivanovic, James Harrison, and Marco Pavone as well as an updated version of [Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data](https://arxiv.org/abs/2001.03093) by Tim Salzmann\*, Boris Ivanovic\*, Punarjay Chakravarty, and Marco Pavone (\* denotes equal contribution).

## Installation ##

### Cloning ###
When cloning this repository, make sure you clone the submodules as well, with the following command:
```sh
git clone --recurse-submodules <repository cloning URL>
```
Alternatively, you can clone the repository as normal and then load submodules later with:
```sh
git submodule init # Initializing our local configuration file
git submodule update # Fetching all of the data from the submodules at the specified commits
```

### Environment Setup ###
First, we'll create a conda environment to hold the dependencies.
```sh
conda create --name adaptive python=3.9 -y
source activate adaptive
pip install -r requirements.txt

# See note in requirements.txt for why we do this.
pip install --no-dependencies l5kit==1.5.0
```

Lastly, install `trajdata` and `trajectron` (this repository) with
```sh
cd unified-av-data-loader
pip install -e .

cd ..
pip install -e .
```

If you now check `pip list`, you should see the above dependencies installed as well as `trajdata` and `trajectron`.

### Data Setup ###

We use [trajdata](https://github.com/NVlabs/trajdata) to manage and access data in this project, please follow the [dataset setup instructions](https://github.com/NVlabs/trajdata/blob/main/DATASETS.md) linked in its README (particularly for ETH/UCY Pedestrians, nuScenes, and Lyft Level 5).

After, please execute `preprocess_challenge_splits.py` (make sure to change Lines 16 and 17 for your environment) within `experiments/nuScenes/` (this will make note of the available scenes within the nuScenes dataset, according to the nuScenes prediction challenge, for later use in training).

## Model Training ##

This repository makes use of [Weights & Biases](https://wandb.ai) for logging training information. Before running any of the following commands, please edit Lines 106 and 107 of `train_unified.py` to specify your desired W&B project and entity names.

### Pedestrian Dataset ###
To train a model on the ETH and UCY Pedestrian datasets, you can execute any of the commands in `experiments/pedestrians/train_1mode_models.sh` from the `experiments/pedestrians/` folder (i.e., run the commands from that folder).

**NOTE:** Make sure that you specify the correct directories for `--data_loc_dict` (where the raw, original dataset is located for each specified dataset) and `--trajdata_cache_dir` (where trajdata's cache is located). The provided values in the shell script are examples, but you can choose whatever suits your computing environment best.

Our codebase is set up such that hyperparameters are saved in a json file every time a model is trained, so that you don't have to remember what particular settings you used when you end up training multiple models.

### nuScenes Dataset ###
To train a model on the nuScenes dataset, you can execute a command similar to the following from within the `experiments/nuScenes/` directory, depending on the model version you desire and other hyperparameters.

For example, running this command from the project's root directory will use the same config as the model saved in `experiments/nuScenes/models/nusc_mm_sec4_tpp-13_Sep_2022_11_06_01/` while overwriting any parameters specified as commandline arguments (e.g., if the `nusc_mm_sec4_tpp-13_Sep_2022_11_06_01` model used a different `train_data`, then this training run would use `nusc_trainval-train` as it was specified as a commandline argument).
```sh
torchrun --nproc_per_node=1 train_unified.py --eval_every=1 --vis_every=1 --batch_size=256 --eval_batch_size=256 --preprocess_workers=16 --log_dir=experiments/nuScenes/models --log_tag=nusc_adaptive_tpp --train_epochs=20 --conf=experiments/nuScenes/models/nusc_mm_sec4_tpp-13_Sep_2022_11_06_01/config.json --trajdata_cache_dir=<TRAJDATA_CACHE_PATH> --data_loc_dict=\{\"nusc_trainval\":\ \"<PATH_TO_NUSC_DATA>\"\} --train_data=nusc_trainval-train --eval_data=nusc_trainval-train_val --history_sec=2.0 --prediction_sec=6.0
```

## Model Evaluation ##
### Pedestrian Datasets ###
To evaluate trained models as in our paper, please run the `experiments/pedestrians/Peds Adaptive.ipynb` notebook.

After that, please use the `experiments/Plots.ipynb` notebook to generate the figures/tables in our paper.

### nuScenes Dataset ###
If you only wish to use a trained model to generate trajectories and plot them, you can do this in the `experiments/nuScenes/nuScenes-Lyft Qualitative.ipynb` notebook.

To quantitatively evaluate a trained model's adaptive prediction performance in the online setting, you can execute the `full_per_agent_eval.py` script within `experiments/nuScenes/`. Similarly, `full_uncorrelated_eval.py` evaluates models' adaptive prediction performance in the offline setting. Finally, the `experiments/nuScenes/nuScenes-Lyft Quantitative.ipynb` notebook contains code to evaluate model calibration as well as base model performance (see notebook for more details). These scripts and notebook will produce csv files in the `experiments/nuScenes/results/` directory which can then be loaded in the `experiments/Plots.ipynb` notebook to recreate the figures from our paper.

**NOTE:** These evaluations can take a long time to run, on the order of 4+ hours for `full_per_agent_eval.py` with a powerful Ryzen CPU and RTX 3090. To make them quicker (albeit at the cost of evaluation accuracy), you can reduce `N_SAMPLES` in Line 517 of `full_per_agent_eval.py` and reduce the frequency of evaluation in Line 610 of `full_uncorrelated_eval.py` (similar advice applies for evaluating calibration across models in `nuScenes-Lyft Quantitative.ipynb`).

## Citation ##
If you use this work in your own research or wish to refer to the paper's results, please use the following BibTeX entries.
```bibtex
@inproceedings{IvanovicHarrisonEtAl2023,
  author       = {Ivanovic, Boris and Harrison, James and Pavone, Marco},
  title        = {Expanding the Deployment Envelope of Behavior Prediction via Adaptive Meta-Learning},
  year         = {2023},
  booktitle    = {{IEEE International Conference on Robotics and Automation (ICRA)}},
  url          = {https://arxiv.org/abs/2209.11820}
}

@inproceedings{SalzmannIvanovicEtAl2020,
  author       = {Salzmann, Tim and Ivanovic, Boris and Chakravarty, Punarjay and Pavone, Marco},
  title        = {{Trajectron++}: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data},
  year         = {2020},
  booktitle    = {{European Conference on Computer Vision (ECCV)}},
  url          = {https://arxiv.org/abs/2001.03093}
}
```
