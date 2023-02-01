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

from collections import defaultdict

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation
from scipy.stats import gaussian_kde

import trajectron.visualization as visualization
from trajectron.utils import prediction_output_to_trajectories


def compute_ade_pt(predicted_trajs, gt_traj):
    error = torch.linalg.norm(predicted_trajs - gt_traj, dim=-1)
    ade = torch.mean(error, axis=-1)
    return ade.flatten()


def compute_ade(predicted_trajs, gt_traj):
    error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    ade = np.mean(error, axis=-1)
    return ade.flatten()


def compute_fde_pt(predicted_trajs, gt_traj):
    final_error = torch.linalg.norm(
        predicted_trajs[..., -1, :] - gt_traj[..., -1, :], dim=-1
    )
    return final_error.flatten()


def compute_fde(predicted_trajs, gt_traj):
    final_error = np.linalg.norm(
        predicted_trajs[..., -1, :] - gt_traj[..., -1, :], axis=-1
    )
    return final_error.flatten()


def compute_kde_nll(predicted_trajs, gt_traj):
    kde_ll = 0.0
    log_pdf_lower_bound = -20
    num_timesteps = gt_traj.shape[0]
    num_batches = predicted_trajs.shape[0]

    for batch_num in range(num_batches):
        for timestep in range(num_timesteps):
            try:
                kde = gaussian_kde(predicted_trajs[batch_num, :, timestep].T)
                pdf = np.clip(
                    kde.logpdf(gt_traj[timestep].T),
                    a_min=log_pdf_lower_bound,
                    a_max=None,
                )[0]
                kde_ll += pdf / (num_timesteps * num_batches)
            except np.linalg.LinAlgError:
                kde_ll = np.nan

    return -kde_ll


def compute_obs_violations(predicted_trajs, map):
    obs_map = map.data

    interp_obs_map = RectBivariateSpline(
        range(obs_map.shape[1]),
        range(obs_map.shape[0]),
        binary_dilation(obs_map.T, iterations=4),
        kx=1,
        ky=1,
    )

    old_shape = predicted_trajs.shape
    pred_trajs_map = map.to_map_points(predicted_trajs.reshape((-1, 2)))

    traj_obs_values = interp_obs_map(
        pred_trajs_map[:, 0], pred_trajs_map[:, 1], grid=False
    )
    traj_obs_values = traj_obs_values.reshape((old_shape[0], old_shape[1]))
    num_viol_trajs = np.sum(traj_obs_values.max(axis=1) > 0, dtype=float)

    return num_viol_trajs


def compute_nll_pt(predicted_dist, gt_traj):
    log_p_yt_xz = torch.clamp(predicted_dist.log_prob(gt_traj), min=-20.0)
    log_p_y_xz_final = log_p_yt_xz[..., -1]
    log_p_y_xz = log_p_yt_xz.mean(dim=-1)
    return -log_p_y_xz[0], -log_p_y_xz_final[0]


def compute_min_afde_k_pt(predicted_dist, gt_traj, k):
    means = predicted_dist.mus
    probs = predicted_dist.pis_cat_dist.probs
    sorted_idxs = torch.argsort(probs[..., [0], :], dim=-1, descending=True)
    sorted_means = torch.gather(
        means,
        dim=-2,
        index=sorted_idxs.unsqueeze(-1).expand(
            (-1, -1, means.shape[2], -1, means.shape[4])
        ),
    )

    errors = torch.linalg.norm(sorted_means - gt_traj.unsqueeze(-2), dim=-1)
    top_k_errors = errors[..., :k]

    ade_k = torch.mean(top_k_errors, dim=-2)
    min_ade_k = torch.min(ade_k, dim=-1).values

    fde_k = top_k_errors[..., -1, :]
    min_fde_k = torch.min(fde_k, dim=-1).values

    return min_ade_k.flatten(), min_fde_k.flatten()


def compute_nll(predicted_dist, gt_traj):
    log_p_yt_xz = torch.clamp(
        predicted_dist.log_prob(torch.as_tensor(gt_traj)), min=-20.0
    )
    log_p_y_xz_final = log_p_yt_xz[..., -1]
    log_p_y_xz = log_p_yt_xz.mean(dim=-1)
    return -log_p_y_xz[0].numpy(), -log_p_y_xz_final[0].numpy()


def compute_batch_statistics_pt(futures, prediction_output_dict=None, y_dists=None):
    eval_ret = dict()
    if prediction_output_dict is not None:
        ade_errors = compute_ade_pt(prediction_output_dict, futures)
        fde_errors = compute_fde_pt(prediction_output_dict, futures)

        eval_ret["ml_ade"] = ade_errors
        eval_ret["ml_fde"] = fde_errors

    if y_dists is not None:
        nll_means, nll_finals = compute_nll_pt(y_dists, futures)
        min_ade_5, min_fde_5 = compute_min_afde_k_pt(y_dists, futures, k=5)
        min_ade_10, min_fde_10 = compute_min_afde_k_pt(y_dists, futures, k=10)

        eval_ret["min_ade_5"] = min_ade_5
        eval_ret["min_fde_5"] = min_fde_5
        eval_ret["min_ade_10"] = min_ade_10
        eval_ret["min_fde_10"] = min_fde_10
        eval_ret["nll_mean"] = nll_means
        eval_ret["nll_final"] = nll_finals

    return eval_ret


def compute_batch_statistics(
    prediction_output_dict,
    dt,
    max_hl,
    ph,
    kde=False,
    obs=False,
    map=None,
    prune_ph_to_future=False,
    best_of=False,
    y_dists=None,
):
    (prediction_dict, histories_dict, futures_dict) = prediction_output_to_trajectories(
        prediction_output_dict, dt, max_hl, ph, prune_ph_to_future=prune_ph_to_future
    )

    batch_error_dict = defaultdict(lambda: defaultdict(list))
    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():
            ade_errors = compute_ade(prediction_dict[t][node], futures_dict[t][node])
            fde_errors = compute_fde(prediction_dict[t][node], futures_dict[t][node])
            if y_dists:
                nll_values, nll_final = compute_nll(
                    y_dists[t][node], futures_dict[t][node]
                )

            if kde:
                kde_ll = compute_kde_nll(
                    prediction_dict[t][node], futures_dict[t][node]
                )
            else:
                kde_ll = 0
            if obs:
                obs_viols = compute_obs_violations(prediction_dict[t][node], map)
            else:
                obs_viols = 0
            if best_of:
                ade_errors = np.min(ade_errors, keepdims=True)
                fde_errors = np.min(fde_errors, keepdims=True)
                kde_ll = np.min(kde_ll)

            if hasattr(node, "detailed_type"):
                batch_error_dict[node.detailed_type]["ade"].extend(list(ade_errors))
                batch_error_dict[node.detailed_type]["fde"].extend(list(fde_errors))
                # batch_error_dict[node.detailed_type]['kde'].extend([kde_ll])
                # batch_error_dict[node.detailed_type]['obs_viols'].extend([obs_viols])
                if y_dists:
                    batch_error_dict[node.detailed_type]["nll_mean"].extend(
                        [nll_values]
                    )
                    batch_error_dict[node.detailed_type]["nll_final"].extend(
                        [nll_final]
                    )
            else:
                batch_error_dict[node.type]["ade"].extend(list(ade_errors))
                batch_error_dict[node.type]["fde"].extend(list(fde_errors))
                # batch_error_dict[node.type]['kde'].extend([kde_ll])
                # batch_error_dict[node.type]['obs_viols'].extend([obs_viols])
                if y_dists:
                    batch_error_dict[node.type]["nll_mean"].extend([nll_values])
                    batch_error_dict[node.type]["nll_final"].extend([nll_final])

    return batch_error_dict


def log_batch_errors(
    batch_errors,
    eval_metrics,
    log_writer,
    namespace,
    epoch,
    curr_iter,
    bar_plot=[],
    box_plot=[],
):
    for node_type in batch_errors.keys():
        for metric in eval_metrics:
            metric_batch_error: np.ndarray = np.concatenate(
                batch_errors[node_type][metric]
            )

            if len(metric_batch_error) > 0:
                log_writer.log(
                    {
                        f"{node_type.name}/{namespace}/{metric}": wandb.Histogram(
                            metric_batch_error
                        ),
                        # f"{node_type.name}/{namespace}/{metric}_min": np.min(metric_batch_error),
                        f"{node_type.name}/{namespace}/{metric}": np.mean(
                            metric_batch_error
                        ),
                        # f"{node_type.name}/{namespace}/{metric}_median": np.median(metric_batch_error),
                        # f"{node_type.name}/{namespace}/{metric}_max": np.max(metric_batch_error),
                        "epoch": epoch,
                    },
                    step=curr_iter,
                )

                if metric in bar_plot:
                    pd = {
                        "dataset": [namespace] * len(metric_batch_error),
                        metric: metric_batch_error,
                    }
                    kde_barplot_fig, ax = plt.subplots(figsize=(5, 5))
                    visualization.visualization_utils.plot_barplots(
                        ax, pd, "dataset", metric
                    )
                    log_writer.add_figure(
                        f"{node_type}/{namespace}/{metric}_bar_plot",
                        kde_barplot_fig,
                        curr_iter,
                    )

                if metric in box_plot:
                    mse_fde_pd = {
                        "dataset": [namespace] * len(metric_batch_error),
                        metric: metric_batch_error,
                    }
                    fig, ax = plt.subplots(figsize=(5, 5))
                    visualization.visualization_utils.plot_boxplots(
                        ax, mse_fde_pd, "dataset", metric
                    )
                    log_writer.add_figure(
                        f"{node_type}/{namespace}/{metric}_box_plot", fig, curr_iter
                    )


def print_batch_errors(batch_errors_list, namespace, curr_iter):
    for node_type in batch_errors_list[0].keys():
        for metric in batch_errors_list[0][node_type].keys():
            metric_batch_error = []
            for batch_errors in batch_errors_list:
                metric_batch_error.extend(batch_errors[node_type][metric])

            if len(metric_batch_error) > 0:
                print(
                    f"{curr_iter}: {node_type.name}/{namespace}/{metric}_mean",
                    np.mean(metric_batch_error),
                )
                # print(f"{curr_iter}: {node_type.name}/{namespace}/{metric}_median", np.median(metric_batch_error))
