#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

### NOTE: These commands are meant to serve as templates, meaning some of the values are truncated for readability.
### To exactly reproduce model training from our paper, please see kf_models/**/config.json for full-fidelity values.
### To use these commands on datasets other than eupeds_eth, make sure to change the `train_data`, `eval_data`,
### and `data_loc_dict` arguments.

### Q: Why 2.8s of history length instead of 3.2 s as used in other works?
### A: While it is oft-said in many papers (e.g., S-GAN, Trajectron++, and nearly all following works)
###    that 3.2 s = 8 timesteps of observation (or history) are used as input,
###    in reality these approaches use 8 _observations_ (which naively comes out to 3.2s via 8*0.4s).
###    However, this neglects the fact that there are only 7 time _steps_ between those 8 points,
###    yielding only 2.8 s of elapsed time. Thus, in this script, you will see us using 2.8s of
###    history to predict 4.8s of future motion, which comes out to using 8 observed timesteps
###    (7 history + 1 current) to predict 12 future timesteps.
###    To summarize, it is because 8 observations = 7 historical timesteps + the current timestep.

# Ours
torchrun --nproc_per_node=1 --master_port=29500 ../../train_unified.py --eval_every=1 --vis_every=1 --batch_size=256 --eval_batch_size=256 --preprocess_workers=16 --log_dir=kf_models --log_tag=eth_1mode_adaptive_tpp --train_epochs=5 --conf=../../config/pedestrians.json --trajdata_cache_dir=~/.unified_data_cache --data_loc_dict=\{\"eupeds_eth\":\ \"~/datasets/eth_ucy_peds\"\} --train_data=eupeds_eth-train --eval_data=eupeds_eth-val --history_sec=2.8 --prediction_sec=4.8 --K=1 --adaptive=True --alpha_init=0.00043 --augment_input_noise=0.005 --grad_clip=5.37 --learning_rate=0.00038 --sigma_eps_init=0.293

# K0
torchrun --nproc_per_node=1 --master_port=29500 ../../train_unified.py --eval_every=1 --vis_every=1 --batch_size=256 --eval_batch_size=256 --preprocess_workers=16 --log_dir=kf_models --log_tag=eth_1mode_k0_tpp --train_epochs=5 --conf=../../config/pedestrians.json --trajdata_cache_dir=~/.unified_data_cache --data_loc_dict=\{\"eupeds_eth\":\ \"~/datasets/eth_ucy_peds\"\} --train_data=eupeds_eth-train --eval_data=eupeds_eth-val --history_sec=2.8 --prediction_sec=4.8 --K=1 --adaptive=True --only_k0=True --alpha_init=2.97e-05 --augment_input_noise=0.86 --grad_clip=87.66 --learning_rate=0.0023 --sigma_eps_init=0.476

# Base
torchrun --nproc_per_node=1 --master_port=29500 ../../train_unified.py --eval_every=1 --vis_every=1 --batch_size=256 --eval_batch_size=256 --preprocess_workers=16 --log_dir=kf_models --log_tag=eth_1mode_base_tpp --train_epochs=5 --conf=../../config/pedestrians.json --trajdata_cache_dir=~/.unified_data_cache --data_loc_dict=\{\"eupeds_eth\":\ \"~/datasets/eth_ucy_peds\"\} --train_data=eupeds_eth-train --eval_data=eupeds_eth-val --history_sec=2.8 --prediction_sec=4.8 --K=1 --alpha_init=1.54e-05 --augment_input_noise=0.57 --grad_clip=0.909 --learning_rate=0.016 --sigma_eps_init=0.0002

# Oracle is just Base on the target split
