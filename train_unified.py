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
import pathlib
import pickle
import random
import time
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils import data
from tqdm import tqdm, trange
from trajdata import AgentType, UnifiedDataset
from trajdata.augmentation import NoiseHistories
from trajdata.data_structures.batch import AgentBatch
from trajdata.data_structures.data_index import AgentDataIndex
from trajdata.visualization import vis as trajdata_vis

import trajectron.evaluation as evaluation
import trajectron.visualization as visualization
from trajectron.argument_parser import args
from trajectron.model.model_registrar import ModelRegistrar
from trajectron.model.model_utils import UpdateMode
from trajectron.model.trajectron import Trajectron
from trajectron.utils.comm import all_gather

# torch.autograd.set_detect_anomaly(True)


def restrict_to_predchal(
    dataset: UnifiedDataset,
    split: str,
    city: str = "",
) -> None:
    curr_dir = pathlib.Path(__file__).parent.resolve()
    with open(
        curr_dir / f"experiments/nuScenes/predchal{city}_{split}_index.pkl", "rb"
    ) as f:
        within_challenge_split = pickle.load(f)

    within_challenge_split = [
        (dataset.cache_path / scene_info_path, num_elems, elems)
        for scene_info_path, num_elems, elems in within_challenge_split
    ]

    dataset._scene_index = [orig_path for orig_path, _, _ in within_challenge_split]

    # The data index is effectively a big list of tuples taking the form:
    # (scene_path: str, index_len: int, valid_timesteps: np.ndarray[, agent_name: str])
    dataset._data_index = AgentDataIndex(within_challenge_split, dataset.verbose)
    dataset._data_len: int = len(dataset._data_index)


def train(rank, args):
    if torch.cuda.is_available():
        args.device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
    else:
        args.device = f"cpu"

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Load hyperparameters from json
    if not os.path.exists(args.conf):
        raise ValueError(f"Config json at {args.conf} not found!")
    with open(args.conf, "r", encoding="utf-8") as conf_json:
        hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams.update({k: v for k, v in vars(args).items() if v is not None})
    hyperparams["edge_encoding"] = not args.no_edge_encoding
    if args.learning_rate is not None:
        hyperparams["learning_rate"] = args.learning_rate

    # Distributed LR Scaling
    hyperparams["learning_rate"] *= dist.get_world_size()

    if rank == 0 and not hyperparams["debug"]:
        if "k0" in hyperparams["log_tag"]:
            model_name = "K0"
        elif "adaptive" in hyperparams["log_tag"]:
            model_name = "Adaptive"
        elif "base" in hyperparams["log_tag"]:
            model_name = "Base"
        else:
            model_name = "Unknown"

        if "eupeds" in hyperparams["train_data"]:
            train_scene = hyperparams["train_data"].split("-")[0][len("eupeds_") :]
        else:
            train_scene = hyperparams["train_data"][:4]

        #######################################################################
        # Make sure to specify your desired project and entity names if needed!
        run = wandb.init(
            project=None,
            entity=None,
            name=hyperparams["log_tag"],
            notes=f"{model_name}, {train_scene}",
            job_type="train",
            group=hyperparams["train_data"],
            config=hyperparams,
        )
        #######################################################################

        hyperparams = run.config

    print("-----------------------")
    print("| TRAINING PARAMETERS |")
    print("-----------------------")
    print("| Max History: %ss" % hyperparams["history_sec"])
    print("| Max Future: %ss" % hyperparams["prediction_sec"])
    print("| Batch Size: %d" % hyperparams["batch_size"])
    print("| Eval Batch Size: %d" % hyperparams["eval_batch_size"])
    print("| Device: %s" % hyperparams["device"])
    print("| Learning Rate: %s" % hyperparams["learning_rate"])
    print("| Learning Rate Step Every: %s" % hyperparams["lr_step"])
    print("| Preprocess Workers: %s" % hyperparams["preprocess_workers"])
    print("| Robot Future: %s" % hyperparams["incl_robot_node"])
    print("| Map Encoding: %s" % hyperparams["map_encoding"])
    print("| Added Input Noise: %.2f" % hyperparams["augment_input_noise"])
    print("| Overall GMM Components: %d" % hyperparams["K"])
    if hyperparams["adaptive"] and not hyperparams["only_k0"]:
        print("| [Adaptive] sigma_eps_init: %f" % hyperparams["sigma_eps_init"])
        print("| [Adaptive] alpha_init: %f" % hyperparams["alpha_init"])
    print("-----------------------")

    log_writer = None
    model_dir = None
    if not hyperparams["debug"]:
        # Create the log and model directory if they're not present.
        model_dir_subfolder = hyperparams["log_tag"] + time.strftime(
            "-%d_%b_%Y_%H_%M_%S", time.localtime()
        )
        model_dir = os.path.join(hyperparams["log_dir"], model_dir_subfolder)

        if rank == 0:
            pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

            # Save config to model directory
            with open(os.path.join(model_dir, "config.json"), "w") as conf_json:
                json.dump(hyperparams.as_dict(), conf_json)

            log_writer = run

        print("model_dir:", model_dir_subfolder)

    # Load training and evaluation environments and scenes
    attention_radius = defaultdict(
        lambda: 20.0
    )  # Default range is 20m unless otherwise specified.
    attention_radius[(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)] = 10.0
    attention_radius[(AgentType.PEDESTRIAN, AgentType.VEHICLE)] = 20.0
    attention_radius[(AgentType.VEHICLE, AgentType.PEDESTRIAN)] = 20.0
    attention_radius[(AgentType.VEHICLE, AgentType.VEHICLE)] = 30.0

    data_dirs: Dict[str, str] = json.loads(hyperparams["data_loc_dict"])

    augmentations = list()
    if hyperparams["augment_input_noise"] > 0.0:
        augmentations.append(NoiseHistories(stddev=hyperparams["augment_input_noise"]))

    map_params = {"px_per_m": 2, "map_size_px": 100, "offset_frac_xy": (-0.75, 0.0)}

    train_dataset = UnifiedDataset(
        desired_data=[hyperparams["train_data"]],
        history_sec=(0.1, hyperparams["history_sec"]),
        future_sec=(0.1, hyperparams["prediction_sec"]),
        agent_interaction_distances=attention_radius,
        incl_robot_future=hyperparams["incl_robot_node"],
        incl_raster_map=hyperparams["map_encoding"],
        raster_map_params=map_params,
        only_predict=[AgentType.VEHICLE, AgentType.PEDESTRIAN],
        no_types=[AgentType.UNKNOWN],
        augmentations=augmentations if len(augmentations) > 0 else None,
        num_workers=hyperparams["preprocess_workers"],
        cache_location=hyperparams["trajdata_cache_dir"],
        data_dirs=data_dirs,
        verbose=True,
    )

    if hyperparams["train_data"] == "nusc_trainval-train":
        restrict_to_predchal(train_dataset, "train")

    train_sampler = data.distributed.DistributedSampler(
        train_dataset, num_replicas=dist.get_world_size(), rank=rank
    )

    train_dataloader = data.DataLoader(
        train_dataset,
        collate_fn=train_dataset.get_collate_fn(pad_format="right"),
        pin_memory=False if hyperparams["device"] == "cpu" else True,
        batch_size=hyperparams["batch_size"],
        shuffle=False,
        num_workers=hyperparams["preprocess_workers"],
        sampler=train_sampler,
    )

    eval_dataset = UnifiedDataset(
        desired_data=[hyperparams["eval_data"]],
        history_sec=(hyperparams["history_sec"], hyperparams["history_sec"]),
        future_sec=(hyperparams["prediction_sec"], hyperparams["prediction_sec"]),
        agent_interaction_distances=attention_radius,
        incl_robot_future=hyperparams["incl_robot_node"],
        incl_raster_map=hyperparams["map_encoding"],
        raster_map_params=map_params,
        only_predict=[AgentType.VEHICLE, AgentType.PEDESTRIAN],
        no_types=[AgentType.UNKNOWN],
        num_workers=hyperparams["preprocess_workers"],
        cache_location=hyperparams["trajdata_cache_dir"],
        data_dirs=data_dirs,
        verbose=True,
    )

    if hyperparams["eval_data"] == "nusc_trainval-train_val":
        restrict_to_predchal(eval_dataset, "train_val")

    eval_sampler = data.distributed.DistributedSampler(
        eval_dataset, num_replicas=dist.get_world_size(), rank=rank
    )

    eval_dataloader = data.DataLoader(
        eval_dataset,
        collate_fn=eval_dataset.get_collate_fn(pad_format="right"),
        pin_memory=False if hyperparams["device"] == "cpu" else True,
        batch_size=hyperparams["eval_batch_size"],
        shuffle=False,
        num_workers=hyperparams["preprocess_workers"],
        sampler=eval_sampler,
    )

    model_registrar = ModelRegistrar(model_dir, hyperparams["device"])

    trajectron = Trajectron(
        model_registrar, hyperparams, log_writer, hyperparams["device"]
    )

    trajectron.set_environment()
    trajectron.set_annealing_params()

    if torch.cuda.is_available():
        trajectron = DDP(
            trajectron,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,
        )
        trajectron_module = trajectron.module
    else:
        trajectron_module = trajectron

    lr_scheduler = None
    step_scheduler = None
    optimizer = optim.Adam(
        [
            {
                "params": model_registrar.get_all_but_name_match(
                    "map_encoder"
                ).parameters()
            },
            {
                "params": model_registrar.get_name_match("map_encoder").parameters(),
                "lr": hyperparams["map_enc_learning_rate"],
            },
        ],
        lr=hyperparams["learning_rate"],
    )
    # Set Learning Rate
    if hyperparams["learning_rate_style"] == "const":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
    elif hyperparams["learning_rate_style"] == "exp":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=hyperparams["learning_decay_rate"]
        )

    if hyperparams["lr_step"] != 0:
        step_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=hyperparams["lr_step"], gamma=0.1
        )

    # if rank == 0 and not hyperparams['debug']:
    #     log_writer.watch(trajectron, log="all", log_freq=500)

    #################################
    #           TRAINING            #
    #################################
    curr_iter: int = 0
    for epoch in range(1, hyperparams["train_epochs"] + 1):
        train_sampler.set_epoch(epoch)
        pbar = tqdm(
            train_dataloader,
            ncols=80,
            unit_scale=dist.get_world_size(),
            disable=(rank > 0),
        )

        # prof = torch.profiler.profile(
        #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler/tpp_unified'),
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True
        # )
        # prof.start()

        # initialize the timer for the 1st iteration
        step_timer_start = time.time()

        batch: AgentBatch
        for batch_idx, batch in enumerate(pbar):
            # if batch_idx >= (1 + 1 + 3) * 2:
            #     break

            trajectron_module.set_curr_iter(curr_iter)
            trajectron_module.step_annealers()

            optimizer.zero_grad(set_to_none=True)

            train_loss = trajectron(batch)

            pbar.set_description(f"Epoch {epoch} L: {train_loss.detach().item():.2f}")

            train_loss.backward()

            # Clipping gradients.
            if hyperparams["grad_clip"] is not None:
                nn.utils.clip_grad_value_(
                    model_registrar.parameters(), hyperparams["grad_clip"]
                )

            optimizer.step()

            # Stepping forward the learning rate scheduler and annealers.
            lr_scheduler.step()
            if rank == 0 and not hyperparams["debug"]:
                step_timer_stop = time.time()
                elapsed = step_timer_stop - step_timer_start

                log_writer.log(
                    {
                        "train/learning_rate": lr_scheduler.get_last_lr()[0],
                        "train/loss": train_loss.detach().item(),
                        "steps_per_sec": 1 / elapsed,
                        "epoch": epoch,
                        "batch": batch_idx,
                    },
                    step=curr_iter,
                )

            curr_iter += 1

            # initialize the timer for the following iteration
            step_timer_start = time.time()

            # prof.step()

        # Resetting adaptive information after training.
        if hyperparams["adaptive"]:
            trajectron_module.reset_adaptive_info()

        # prof.stop()
        # raise
        if hyperparams["lr_step"] != 0:
            step_scheduler.step()

        #################################
        #           EVALUATION          #
        #################################
        if (
            hyperparams["eval_every"] is not None
            and not hyperparams["debug"]
            and epoch % hyperparams["eval_every"] == 0
            and epoch > 0
        ):
            with torch.no_grad():
                # Calculate evaluation loss
                eval_perf = defaultdict(lambda: defaultdict(list))

                batch: AgentBatch
                for batch in tqdm(
                    eval_dataloader,
                    ncols=80,
                    unit_scale=dist.get_world_size(),
                    disable=(rank > 0),
                    desc=f"Epoch {epoch} Eval",
                ):
                    eval_results: Dict[
                        AgentType, Dict[str, torch.Tensor]
                    ] = trajectron_module.predict_and_evaluate_batch(
                        batch, update_mode=UpdateMode.BATCH_FROM_PRIOR
                    )
                    for agent_type, metric_dict in eval_results.items():
                        for metric, values in metric_dict.items():
                            eval_perf[agent_type][metric].append(values.cpu().numpy())

                if torch.cuda.is_available() and dist.get_world_size() > 1:
                    gathered_values = all_gather(eval_perf)
                    if rank == 0:
                        eval_perf = []
                        for eval_dicts in gathered_values:
                            eval_perf.extend(eval_dicts)

                if rank == 0:
                    evaluation.log_batch_errors(
                        eval_perf,
                        [
                            "ml_ade",
                            "ml_fde",
                            "min_ade_5",
                            "min_ade_10",
                            "nll_mean",
                            "nll_final",
                        ],
                        log_writer,
                        "eval",
                        epoch,
                        curr_iter,
                    )

                    ####################################
                    #           VIZUALIZATION          #
                    ####################################
                    # import matplotlib.pyplot as plt

                    # batch_idxs = random.sample(range(len(eval_dataset)), 5)
                    # batch: AgentBatch = eval_dataset.get_collate_fn(pad_format="right")(
                    #     [eval_dataset[i] for i in batch_idxs]
                    # )
                    # pred_dists, _ = trajectron_module.predict(
                    #     batch,
                    #     update_mode=UpdateMode.BATCH_FROM_PRIOR
                    # )
                    # batch.to("cpu")

                    # images = list()
                    # for i in trange(len(batch_idxs), desc="Visualizing Random Predictions"):
                    #     try:
                    #         fig, ax = plt.subplots()
                    #         trajdata_vis.plot_agent_batch(batch, batch_idx=i, ax=ax, show=False, close=False)
                    #         visualization.visualize_distribution(ax, pred_dists, batch_idx=i)

                    #         images.append(wandb.Image(
                    #             fig,
                    #             caption=f"{str(AgentType(batch.agent_type[i].item()))}/{batch.agent_name[i]}"
                    #         ))
                    #     except:
                    #         continue

                    # log_writer.log({f"eval/predictions_viz": images}, step=curr_iter)
                    # plt.close("all")

        if rank == 0 and (
            hyperparams["save_every"] is not None
            and hyperparams["debug"] is False
            and epoch % hyperparams["save_every"] == 0
        ):
            save_checkpoint(
                model_registrar.model_dir,
                trajectron_module,
                optimizer,
                lr_scheduler,
                step_scheduler,
                epoch,
            )

        # Waiting for process 0 to be done its evaluation and visualization.
        if torch.cuda.is_available():
            dist.barrier()


def save_checkpoint(
    save_dir: str,
    model: Trajectron,
    optimizer: optim.Optimizer,
    lr_scheduler: Optional[optim.lr_scheduler._LRScheduler],
    step_scheduler: Optional[optim.lr_scheduler.StepLR],
    epoch: int,
) -> None:
    save_path = pathlib.Path(save_dir) / f"model_registrar-{epoch}.pt"

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict()
            if lr_scheduler is not None
            else None,
            "step_scheduler_state_dict": step_scheduler.state_dict()
            if step_scheduler is not None
            else None,
        },
        save_path,
    )


def spmd_main(local_rank):
    if torch.cuda.is_available():
        backend = "nccl"
    else:
        backend = "gloo"

    dist.init_process_group(backend=backend)

    print(
        f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}, "
        + f"port = {os.environ['MASTER_PORT']} \n",
        end="",
    )

    train(local_rank, args)


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    spmd_main(local_rank)
