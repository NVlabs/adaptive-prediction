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

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Final, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import trajdata.visualization.vis as trajdata_vis
from torch import nn, optim
from torch.utils import data
from tqdm import tqdm
from trajdata import AgentBatch, AgentType, UnifiedDataset

import trajectron.visualization as visualization
from trajectron.model.model_registrar import ModelRegistrar
from trajectron.model.model_utils import UpdateMode
from trajectron.model.trajectron import Trajectron

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


###########################################################################
# Change this to match your computing environment!
LYFT_SAMPLE_RAW_DATA_DIR: Final[
    str
] = "/home/bivanovic/datasets/lyft/scenes/sample.zarr"
###########################################################################

base_model = "models/nusc_mm_base_tpp-11_Sep_2022_19_15_45"
k0_model = "models/nusc_mm_k0_tpp-12_Sep_2022_00_40_16"
adaptive_model = "models/nusc_mm_sec4_tpp-13_Sep_2022_11_06_01"
oracle_model = "models/lyft_mm_base_tpp-11_Sep_2022_18_56_49"

base_checkpoint = 20
k0_checkpoint = 20
adaptive_checkpoint = 20
oracle_checkpoint = 1

eval_data = "lyft_sample-mini_val"

history_sec = 2.0
prediction_sec = 6.0


AXHLINE_COLORS = {"Base": "#DD9787", "K0": "#A6C48A", "Oracle": "#BCB6FF"}

SEABORN_PALETTE = {
    "Finetune": "#AA7C85",
    "K0+Finetune": "#2D93AD",
    "Ours": "#67934D",
    "Ours+Finetune": "#FF8E61",
    "Base": "#DD9787",
    "K0": "#A6C48A",
    "Oracle": "#BCB6FF",
}


def load_model(
    model_dir: str,
    device: str,
    epoch: int = 10,
    custom_hyperparams: Optional[Dict] = None,
):
    save_path = Path(model_dir) / f"model_registrar-{epoch}.pt"

    model_registrar = ModelRegistrar(model_dir, device)
    with open(os.path.join(model_dir, "config.json"), "r") as config_json:
        hyperparams = json.load(config_json)

    if custom_hyperparams is not None:
        hyperparams.update(custom_hyperparams)

    trajectron = Trajectron(model_registrar, hyperparams, None, device)
    trajectron.set_environment()
    trajectron.set_annealing_params()

    checkpoint = torch.load(save_path, map_location=device)
    trajectron.load_state_dict(checkpoint["model_state_dict"], strict=False)

    return trajectron, hyperparams


if torch.cuda.is_available():
    device = "cuda:0"
    torch.cuda.set_device(0)
else:
    device = "cpu"


def finetune_update(
    model: Trajectron,
    batch: AgentBatch = None,
    dataloader: data.DataLoader = None,
    num_epochs: int = None,
    update_mode: UpdateMode = UpdateMode.NO_UPDATE,
) -> float:
    if batch is None and dataloader is None:
        raise ValueError("Only one of batch or dataloader can be passed in.")

    if dataloader is not None and num_epochs is None:
        raise ValueError("num_epochs must not be None if dataloader is not None.")

    lr_scheduler = None
    optimizer = optim.Adam(
        [
            {
                "params": model.model_registrar.get_all_but_name_match(
                    "map_encoder"
                ).parameters()
            },
            {
                "params": model.model_registrar.get_name_match(
                    "map_encoder"
                ).parameters(),
                "lr": model.hyperparams["map_enc_learning_rate"] / 10,
            },
        ],
        lr=model.hyperparams["learning_rate"] / 10,
    )
    # Set Learning Rate
    if model.hyperparams["learning_rate_style"] == "const":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
    elif model.hyperparams["learning_rate_style"] == "exp":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=model.hyperparams["learning_decay_rate"]
        )

    if batch is not None:
        model.step_annealers()
        optimizer.zero_grad(set_to_none=True)

        train_loss = model(batch, update_mode=update_mode)
        train_loss.backward()

        # Clipping gradients.
        if model.hyperparams["grad_clip"] is not None:
            nn.utils.clip_grad_value_(
                model.model_registrar.parameters(), model.hyperparams["grad_clip"]
            )

        optimizer.step()

        # Stepping forward the learning rate scheduler and annealers.
        lr_scheduler.step()

    elif dataloader is not None:
        batch: AgentBatch
        for batch_idx, batch in enumerate(dataloader):
            model.step_annealers()

            optimizer.zero_grad(set_to_none=True)

            train_loss = model(batch)

            train_loss.backward()

            # Clipping gradients.
            if model.hyperparams["grad_clip"] is not None:
                nn.utils.clip_grad_value_(
                    model.model_registrar.parameters(), model.hyperparams["grad_clip"]
                )

            optimizer.step()

            # Stepping forward the learning rate scheduler and annealers.
            lr_scheduler.step()


def finetune_last_layer_update(
    model: Trajectron,
    batch: AgentBatch = None,
    dataloader: data.DataLoader = None,
    num_epochs: int = None,
) -> float:
    if batch is None and dataloader is None:
        raise ValueError("Only one of batch or dataloader can be passed in.")

    if dataloader is not None and num_epochs is None:
        raise ValueError("num_epochs must not be None if dataloader is not None.")

    lr_scheduler = None
    optimizer = optim.Adam(
        [
            {
                "params": model.model_registrar.get_all_but_name_match(
                    "last_layer"
                ).parameters()
            },
            {
                "params": model.model_registrar.get_name_match(
                    "last_layer"
                ).parameters(),
                "lr": model.hyperparams["learning_rate"] / 10,
            },
        ],
        lr=0,
    )
    # Set Learning Rate
    if model.hyperparams["learning_rate_style"] == "const":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
    elif model.hyperparams["learning_rate_style"] == "exp":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=model.hyperparams["learning_decay_rate"]
        )

    if batch is not None:
        model.step_annealers()
        optimizer.zero_grad(set_to_none=True)

        train_loss = model(batch)
        train_loss.backward()

        # Clipping gradients.
        if model.hyperparams["grad_clip"] is not None:
            nn.utils.clip_grad_value_(
                model.model_registrar.parameters(), model.hyperparams["grad_clip"]
            )

        optimizer.step()

        # Stepping forward the learning rate scheduler and annealers.
        lr_scheduler.step()

    elif dataloader is not None:
        batch: AgentBatch
        for batch_idx, batch in enumerate(dataloader):
            model.step_annealers()

            optimizer.zero_grad(set_to_none=True)

            train_loss = model(batch)

            train_loss.backward()

            # Clipping gradients.
            if model.hyperparams["grad_clip"] is not None:
                nn.utils.clip_grad_value_(
                    model.model_registrar.parameters(), model.hyperparams["grad_clip"]
                )

            optimizer.step()

            # Stepping forward the learning rate scheduler and annealers.
            lr_scheduler.step()


adaptive_trajectron, hyperparams = load_model(
    adaptive_model,
    device,
    epoch=adaptive_checkpoint,
    custom_hyperparams={
        "trajdata_cache_dir": "/home/bivanovic/.unified_data_cache",
        "single_mode_multi_sample": True,
    },
)
# For offline test
adaptive_finetune_trajectron, _ = load_model(
    adaptive_model,
    device,
    epoch=adaptive_checkpoint,
    custom_hyperparams={
        "trajdata_cache_dir": "/home/bivanovic/.unified_data_cache",
        "single_mode_multi_sample": True,
    },
)

k0_trajectron, _ = load_model(
    k0_model,
    device,
    epoch=k0_checkpoint,
    custom_hyperparams={
        "trajdata_cache_dir": "/home/bivanovic/.unified_data_cache",
        "single_mode_multi_sample": False,
    },
)
k0_finetune_trajectron, _ = load_model(
    k0_model,
    device,
    epoch=k0_checkpoint,
    custom_hyperparams={
        "trajdata_cache_dir": "/home/bivanovic/.unified_data_cache",
        "single_mode_multi_sample": False,
    },
)

base_trajectron, _ = load_model(
    base_model,
    device,
    epoch=base_checkpoint,
    custom_hyperparams={
        "trajdata_cache_dir": "/home/bivanovic/.unified_data_cache",
        "single_mode_multi_sample": False,
    },
)
finetune_trajectron, _ = load_model(
    base_model,
    device,
    epoch=base_checkpoint,
    custom_hyperparams={
        "trajdata_cache_dir": "/home/bivanovic/.unified_data_cache",
        "single_mode_multi_sample": False,
    },
)

oracle_trajectron, _ = load_model(
    oracle_model,
    device,
    epoch=oracle_checkpoint,
    custom_hyperparams={
        "trajdata_cache_dir": "/home/bivanovic/.unified_data_cache",
        "single_mode_multi_sample": False,
    },
)

# Load training and evaluation environments and scenes
attention_radius = defaultdict(
    lambda: 20.0
)  # Default range is 20m unless otherwise specified.
attention_radius[(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)] = 10.0
attention_radius[(AgentType.PEDESTRIAN, AgentType.VEHICLE)] = 20.0
attention_radius[(AgentType.VEHICLE, AgentType.PEDESTRIAN)] = 20.0
attention_radius[(AgentType.VEHICLE, AgentType.VEHICLE)] = 30.0

map_params = {"px_per_m": 2, "map_size_px": 100, "offset_frac_xy": (-0.75, 0.0)}

online_eval_dataset = UnifiedDataset(
    desired_data=[eval_data],
    history_sec=(0.1, history_sec),
    future_sec=(prediction_sec, prediction_sec),
    agent_interaction_distances=attention_radius,
    incl_robot_future=hyperparams["incl_robot_node"],
    incl_raster_map=hyperparams["map_encoding"],
    raster_map_params=map_params,
    only_predict=[AgentType.VEHICLE],
    no_types=[AgentType.UNKNOWN],
    num_workers=0,
    cache_location=hyperparams["trajdata_cache_dir"],
    data_dirs={
        "lyft_sample": LYFT_SAMPLE_RAW_DATA_DIR,
    },
    verbose=True,
)

batch_eval_dataset = UnifiedDataset(
    desired_data=[eval_data],
    history_sec=(history_sec, history_sec),
    future_sec=(prediction_sec, prediction_sec),
    agent_interaction_distances=attention_radius,
    incl_robot_future=hyperparams["incl_robot_node"],
    incl_raster_map=hyperparams["map_encoding"],
    raster_map_params=map_params,
    only_predict=[AgentType.VEHICLE],
    no_types=[AgentType.UNKNOWN],
    num_workers=0,
    cache_location=hyperparams["trajdata_cache_dir"],
    data_dirs={
        "lyft_sample": LYFT_SAMPLE_RAW_DATA_DIR,
    },
    verbose=True,
)


prog = re.compile("(.*)/(?P<scene_name>.*)/(.*)$")


def plot_outputs(
    eval_dataset: UnifiedDataset,
    dataset_idx: int,
    model: Trajectron,
    model_name: str,
    agent_ts: int,
    save=True,
    extra_str=None,
    subfolder="",
    filetype="png",
):
    batch: AgentBatch = eval_dataset.get_collate_fn(pad_format="right")(
        [eval_dataset[dataset_idx]]
    )
    
    fig, ax = plt.subplots()
    trajdata_vis.plot_agent_batch(batch, batch_idx=0, ax=ax, show=False, close=False)

    with torch.no_grad():
        # predictions = model.predict(batch,
        #                             z_mode=True,
        #                             gmm_mode=True,
        #                             full_dist=False,
        #                             output_dists=False)
        # prediction = next(iter(predictions.values()))

        pred_dists, _ = model.predict(
            batch, z_mode=False, gmm_mode=False, full_dist=True, output_dists=True
        )
        # pred_dist = next(iter(pred_dists.values()))

    batch.to("cpu")

    visualization.visualize_distribution(ax, pred_dists, batch_idx=0)

    # batch_eval: Dict[str, torch.Tensor] = evaluation.compute_batch_statistics_pt(
    #     batch.agent_fut[..., :2],
    #     prediction_output_dict=torch.from_numpy(prediction),
    #     y_dists=pred_dist
    # )

    scene_info_path, _, scene_ts = eval_dataset._data_index[dataset_idx]
    scene_name = prog.match(scene_info_path).group("scene_name")

    agent_name = batch.agent_name[0]
    agent_type_name = f"{str(AgentType(batch.agent_type[0].item()))}/{agent_name}"

    ax.set_title(f"{scene_name}/t={scene_ts} {agent_type_name}")
    # print(model_name, extra_str, batch_eval)

    if save:
        fname = f"plots/{subfolder}{model_name}_{scene_name}_{agent_name}_t{agent_ts}"
        if extra_str:
            fname += "_" + extra_str
        fig.savefig(fname + f".{filetype}")

        plt.close(fig)


def get_dataloader(
    eval_dataset: UnifiedDataset,
    batch_size: int = 128,
    num_workers: int = 0,
    shuffle: bool = False,
):
    return data.DataLoader(
        eval_dataset,
        collate_fn=eval_dataset.get_collate_fn(pad_format="right"),
        pin_memory=False if device == "cpu" else True,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


metrics_list = ["ml_ade", "ml_fde", "nll_mean", "min_ade_5", "min_ade_10"]


import pickle

with open("base_mm_performance.pkl", "rb") as f:
    eval_dict = pickle.load(f)


def dataset_eval(
    batch_eval_dataloader: data.DataLoader,
    model: Trajectron,
    eval_type: str,
    model_name: str,
    num_updates: int,
    model_eval_dict: DefaultDict[str, Union[List[int], List[float]]],
    plot=False,
):
    with torch.no_grad():
        if plot:
            plot_outputs(
                batch_eval_dataset,
                dataset_idx=200,
                model=model,
                model_name=model_name,
                agent_ts=num_updates,
                subfolder=eval_type,
            )

        model_perf = defaultdict(lambda: defaultdict(list))
        for eval_batch in tqdm(
            batch_eval_dataloader,
            desc=f"{model_name} Eval {num_updates}",
            leave=False,
            position=1,
        ):
            eval_results: Dict[
                AgentType, Dict[str, torch.Tensor]
            ] = model.predict_and_evaluate_batch(eval_batch)
            for agent_type, metric_dict in eval_results.items():
                for metric, values in metric_dict.items():
                    model_perf[agent_type][metric].append(values.cpu().numpy())

        for idx, metric in enumerate(metrics_list):
            metric_values = np.concatenate(
                model_perf[AgentType.VEHICLE][metric]
            ).tolist()
            if idx == 0:
                model_eval_dict["num_updates"].extend(
                    [num_updates] * len(metric_values)
                )

            model_eval_dict[metric].extend(metric_values)


adaptive_dict = defaultdict(list)
adaptive_finetune_dict = defaultdict(list)
finetune_dict = defaultdict(list)
k0_finetune_dict = defaultdict(list)

online_eval_dataloader = get_dataloader(online_eval_dataset, batch_size=1, shuffle=True)
batch_eval_dataloader = get_dataloader(
    batch_eval_dataset, batch_size=128, num_workers=16
)

adaptive_trajectron.reset_adaptive_info()
adaptive_finetune_trajectron.reset_adaptive_info()

# Resetting the finetune baseline to its base.
finetune_trajectron, _ = load_model(
    base_model,
    device,
    epoch=base_checkpoint,
    custom_hyperparams={
        "trajdata_cache_dir": "/home/bivanovic/.unified_data_cache",
        "single_mode_multi_sample": False,
    },
)
k0_finetune_trajectron, _ = load_model(
    k0_model,
    device,
    epoch=k0_checkpoint,
    custom_hyperparams={
        "trajdata_cache_dir": "/home/bivanovic/.unified_data_cache",
        "single_mode_multi_sample": False,
    },
)

N_SAMPLES = 1001

outer_pbar = tqdm(
    online_eval_dataloader,
    total=min(N_SAMPLES, len(online_eval_dataloader)),
    desc=f"Adaptive Eval PH={prediction_sec}",
    position=0,
)

plot_per_eval = False

online_batch: AgentBatch
for data_sample, online_batch in enumerate(outer_pbar):
    if data_sample >= N_SAMPLES:
        outer_pbar.close()
        break

    if data_sample == 0:
        dataset_eval(
            batch_eval_dataloader,
            adaptive_trajectron,
            "uncorrelated_lyft/",
            "Ours",
            data_sample,
            adaptive_dict,
            plot_per_eval,
        )
        dataset_eval(
            batch_eval_dataloader,
            adaptive_finetune_trajectron,
            "uncorrelated_lyft/",
            "Ours+Finetune",
            data_sample,
            adaptive_finetune_dict,
            plot_per_eval,
        )
        dataset_eval(
            batch_eval_dataloader,
            finetune_trajectron,
            "uncorrelated_lyft/",
            "Finetune",
            data_sample,
            finetune_dict,
            plot_per_eval,
        )
        dataset_eval(
            batch_eval_dataloader,
            k0_finetune_trajectron,
            "uncorrelated_lyft/",
            "K0_Finetune",
            data_sample,
            k0_finetune_dict,
            plot_per_eval,
        )

    with torch.no_grad():
        # This is the inference call that internally updates L_n and K_n.
        adaptive_trajectron.adaptive_predict(
            online_batch, update_mode=UpdateMode.ITERATIVE
        )

    if data_sample < 101:
        with torch.no_grad():
            adaptive_finetune_trajectron.adaptive_predict(
                online_batch, update_mode=UpdateMode.ITERATIVE
            )
    else:
        finetune_update(
            adaptive_finetune_trajectron, online_batch, update_mode=UpdateMode.NO_UPDATE
        )

    finetune_update(finetune_trajectron, online_batch)
    finetune_last_layer_update(k0_finetune_trajectron, online_batch)

    if (data_sample + 1) % 50 == 0:
        dataset_eval(
            batch_eval_dataloader,
            adaptive_trajectron,
            "uncorrelated_lyft/",
            "Ours",
            data_sample + 1,
            adaptive_dict,
            plot_per_eval,
        )
        dataset_eval(
            batch_eval_dataloader,
            adaptive_finetune_trajectron,
            "uncorrelated_lyft/",
            "Ours+Finetune",
            data_sample + 1,
            adaptive_finetune_dict,
            plot_per_eval,
        )
        dataset_eval(
            batch_eval_dataloader,
            finetune_trajectron,
            "uncorrelated_lyft/",
            "Finetune",
            data_sample + 1,
            finetune_dict,
            plot_per_eval,
        )
        dataset_eval(
            batch_eval_dataloader,
            k0_finetune_trajectron,
            "uncorrelated_lyft/",
            "K0_Finetune",
            data_sample + 1,
            k0_finetune_dict,
            plot_per_eval,
        )


adaptive_eval_df = pd.DataFrame.from_dict(adaptive_dict)
adaptive_eval_df["method"] = "Ours"
adaptive_finetune_eval_df = pd.DataFrame.from_dict(adaptive_finetune_dict)
adaptive_finetune_eval_df["method"] = "Ours+Finetune"
finetune_eval_df = pd.DataFrame.from_dict(finetune_dict)
finetune_eval_df["method"] = "Finetune"
k0_finetune_eval_df = pd.DataFrame.from_dict(k0_finetune_dict)
k0_finetune_eval_df["method"] = "K0+Finetune"


def relativize_df(input_df, reference_values: np.ndarray):
    output_df = input_df.copy()
    output_df.loc[:, metrics_list] = (
        100 * (output_df.loc[:, metrics_list] - reference_values) / reference_values
    )
    return output_df


horizontal_lines_eval_df = pd.DataFrame.from_dict(eval_dict)
relative_baseline = horizontal_lines_eval_df.loc[0, metrics_list]

combined_df = pd.concat(
    (
        relativize_df(adaptive_eval_df, relative_baseline),
        relativize_df(adaptive_finetune_eval_df, relative_baseline),
        relativize_df(finetune_eval_df, relative_baseline),
        relativize_df(k0_finetune_eval_df, relative_baseline),
    ),
    ignore_index=True,
)

combined_df.to_csv(f"results/{eval_data}_uncorrelated_offline_rel.csv", index=False)
