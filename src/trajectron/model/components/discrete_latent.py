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

import numpy as np
import torch
import torch.distributions as td
import wandb

from trajectron.model.model_utils import ModeKeys


class DiscreteLatent(object):
    def __init__(self, hyperparams, device):
        self.hyperparams = hyperparams
        self.z_dim = hyperparams["N"] * hyperparams["K"]
        self.N = hyperparams["N"]
        self.K = hyperparams["K"]
        self.kl_min = hyperparams["kl_min"]
        self.device = device
        self.temp = None  # filled in by MultimodalGenerativeCVAE.set_annealing_params
        self.z_logit_clip = (
            None  # filled in by MultimodalGenerativeCVAE.set_annealing_params
        )
        self.p_dist = None  # filled in by MultimodalGenerativeCVAE.encoder
        self.q_dist = None  # filled in by MultimodalGenerativeCVAE.encoder

    def dist_from_h(self, h: torch.Tensor, mode: ModeKeys):
        logits_separated = torch.reshape(
            h,
            (*h.shape[:2], self.N, self.K)
            if self.hyperparams["adaptive"]
            else (h.shape[0], self.N, self.K),
        )

        if self.N == 1:
            logits_separated.squeeze_(dim=-2)

        logits_separated_mean_zero = logits_separated - torch.mean(
            logits_separated, dim=-1, keepdim=True
        )
        if self.z_logit_clip is not None and mode == ModeKeys.TRAIN:
            c = self.z_logit_clip
            logits = torch.clamp(logits_separated_mean_zero, min=-c, max=c)
        else:
            logits = logits_separated_mean_zero

        return td.OneHotCategorical(logits=logits)

    def sample_q(self, num_samples, mode):
        bs = self.p_dist.probs.size()[0]
        num_components = self.N * self.K
        z_NK = (
            torch.from_numpy(self.all_one_hot_combinations(self.N, self.K))
            .float()
            .to(self.device)
            .repeat(num_samples, bs)
        )
        return torch.reshape(z_NK, (num_samples * num_components, -1, self.z_dim))

    def sample_p(
        self, num_samples, mode, most_likely_z=False, full_dist=True, all_z_sep=False
    ):
        num_components = 1
        bs = self.p_dist.probs.shape[0]
        if full_dist:
            z_NK = (
                torch.from_numpy(self.all_one_hot_combinations(self.N, self.K))
                .float()
                .to(self.device)
                .repeat(num_samples, bs)
            )
            num_components = self.K**self.N
            k = num_samples * num_components
        elif all_z_sep:
            z_NK = (
                torch.from_numpy(self.all_one_hot_combinations(self.N, self.K))
                .float()
                .to(self.device)
                .repeat(1, bs)
            )
            k = self.K**self.N
            num_samples = k
        elif most_likely_z:
            # Sampling the most likely z from p(z|x).
            eye_mat = torch.eye(self.p_dist.event_shape[-1], device=self.device)
            argmax_idxs = torch.argmax(self.p_dist.probs, dim=-1)
            z_NK = torch.unsqueeze(eye_mat[argmax_idxs], dim=0).expand(
                num_samples, -1, -1, -1
            )
            k = num_samples
        else:
            z_NK = self.p_dist.sample((num_samples,))
            k = num_samples

        ret_shape = (
            (k, bs, -1, self.N * self.K)
            if self.hyperparams["adaptive"]
            else (k, bs, self.N * self.K)
        )

        if mode == ModeKeys.PREDICT:
            return torch.reshape(z_NK, ret_shape), num_samples, num_components
        else:
            return torch.reshape(z_NK, ret_shape)

    def kl_q_p(self, log_writer=None, prefix=None, curr_iter=None):
        kl_separated = td.kl_divergence(self.q_dist, self.p_dist)
        if len(kl_separated.size()) < 2:
            kl_separated = torch.unsqueeze(kl_separated, dim=0)

        kl_minibatch = torch.mean(kl_separated[:, -1], dim=0, keepdim=True)

        if log_writer is not None:
            log_writer.log(
                {prefix + "/true_kl": torch.sum(kl_minibatch).item()},
                step=curr_iter,
                commit=False,
            )

        if self.kl_min > 0:
            kl_lower_bounded = torch.clamp(kl_minibatch, min=self.kl_min)
            kl = torch.sum(kl_lower_bounded)
        else:
            kl = torch.sum(kl_minibatch)

        return kl

    def q_log_prob(self, z):
        k = z.size()[0]
        z_NK = torch.reshape(z, [k, -1, self.N, self.K])
        return torch.sum(self.q_dist.log_prob(z_NK), dim=2)

    def p_log_prob(self, z):
        k = z.size()[0]
        z_NK = torch.reshape(z, [k, -1, self.N, self.K])
        return torch.sum(self.p_dist.log_prob(z_NK), dim=2)

    def get_p_dist_probs(self):
        return self.p_dist.probs

    @staticmethod
    def all_one_hot_combinations(N, K):
        return (
            np.eye(K)
            .take(np.reshape(np.indices([K] * N), [N, -1]).T, axis=0)
            .reshape(-1, N * K)
        )  # [K**N, N*K]

    def summarize_for_tensorboard(self, log_writer, prefix, curr_iter):
        log_writer.log(
            {
                prefix
                + "/latent/p_z_x": wandb.Histogram(
                    self.p_dist.probs.detach().cpu().numpy()
                ),
                prefix
                + "/latent/q_z_xy": wandb.Histogram(
                    self.q_dist.probs.detach().cpu().numpy()
                ),
                prefix
                + "/latent/p_z_x_logits": wandb.Histogram(
                    self.p_dist.logits.detach().cpu().numpy()
                ),
                prefix
                + "/latent/q_z_xy_logits": wandb.Histogram(
                    self.q_dist.logits.detach().cpu().numpy()
                ),
            },
            step=curr_iter,
            commit=False,
        )
        if self.z_dim <= 9:
            for i in range(self.N):
                for j in range(self.K):
                    log_writer.log(
                        {
                            prefix
                            + "/latent/q_z_xy_logit{0}{1}".format(
                                i, j
                            ): wandb.Histogram(
                                self.q_dist.logits[:, i, j].detach().cpu().numpy()
                            )
                        },
                        step=curr_iter,
                        commit=False,
                    )
