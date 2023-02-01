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

from math import sqrt
from typing import Optional, Tuple

import torch
import torch.distributions as td
import torch.nn as nn

from trajectron.model.model_utils import UpdateMode


class BayesianLastLayer(nn.Module):
    def __init__(
        self,
        pred_state_length: int,
        phi_dim: int,
        num_GMM_components: int,
        sigma_eps_init: float,
        alpha_init: float,
        fixed_sigma: bool,
        fixed_alpha: bool,
        S_init_diag_add: float,
        device,
    ):
        super(BayesianLastLayer, self).__init__()

        self.phi_dim = phi_dim
        self._S_init_diag_add = S_init_diag_add
        self.device = device

        # Sigma_eps here is (N, ) with N learnable parameters
        # (will later make it diagonal of matrices).
        Sigma_eps_data = torch.full(
            (pred_state_length,), fill_value=sigma_eps_init, device=self.device
        )
        if fixed_sigma:
            self.Sigma_eps = Sigma_eps_data
        else:
            # This is parametrized in mgcvae.py to be positive by way of exponentiation.
            self.Sigma_eps = nn.Parameter(data=Sigma_eps_data)

        alpha_data = torch.tensor(data=alpha_init, device=self.device)
        if fixed_alpha:
            self.alpha = alpha_data
        else:
            # This is parametrized in mgcvae.py to be positive by way of exponentiation.
            self.alpha = nn.Parameter(data=alpha_data)

        # Sigma_nu is (K, N, phi_dim, phi_dim) with only 1 learnable parameter (alpha).
        self.Sigma_nu = torch.diagflat(self.alpha.expand(self.phi_dim)).expand(
            num_GMM_components, pred_state_length, -1, -1
        )

        # w_0|0 is (K, N, D, 1) with K*N*D learnable parameters.
        glorot = sqrt(6 / (phi_dim + 1))
        w_0_0 = glorot * (
            torch.rand(
                (num_GMM_components, pred_state_length, phi_dim, 1), device=self.device
            )
            * 2
            - 1
        )
        self.w_0_0 = nn.Parameter(data=w_0_0)

        # S_0|0 is (K, N, D, D) with K*N*(# in lower triangular) parameters
        glorot = sqrt(3 / phi_dim)
        S_chol_0_0 = glorot * (
            torch.rand(
                (num_GMM_components, pred_state_length, phi_dim, phi_dim),
                device=self.device,
            )
            * 2
            - 1
        )
        self.S_chol_0_0 = nn.Parameter(data=S_chol_0_0)

        self.reset_to_prior()

    def _get_S_0_0(self):
        # Ensuring the diagonal is positive with a small shift which attempts to avoid
        # numerical issues from zeros on the diagonal (from tfp.FillScaleTriL).
        # (num_components, num_outputs, phi_dim, phi_dim)
        return self.S_chol_0_0 @ self.S_chol_0_0.mT + self._S_init_diag_add * torch.eye(
            self.phi_dim, device=self.device
        )

    def reset_to_prior(self):
        self.S_t_t: torch.Tensor = self._get_S_0_0()[None, None, :, :, :]
        self.w_t_t: torch.Tensor = self.w_0_0[None, None, :, :, :, :]

        self.w_tp1_t: torch.Tensor = self.w_t_t  # A = I, b = 0, [S, B, K, N, D, 1]
        self.S_tp1_t: torch.Tensor = (
            self.S_t_t + self.Sigma_nu
        )  # A = I, [S, B, K, N, D, D]

        self.num_updates: int = 0

    def _expanded_Sigma_eps(self, num_samples: int, batch_size: int):
        return self.Sigma_eps[None, None, None, :, None, None].expand(
            (num_samples, batch_size, self.w_0_0.shape[0], -1, -1, -1)
        )  # [S, B, K, N, 1, 1]

    def get_prior(
        self, Phi_t: torch.Tensor  # [S, B, K, N, D]
    ) -> Tuple[td.MultivariateNormal, torch.Tensor]:
        S, B = Phi_t.shape[:2]
        Phi_t = Phi_t.unsqueeze(-2)  # [S, B, K, N, 1, D]

        S_0_0 = self._get_S_0_0().expand((S, B, -1, -1, -1, -1))

        # A = I
        S_1_0 = S_0_0 + self.Sigma_nu

        expanded_Sigma_eps = self._expanded_Sigma_eps(S, B)
        Sigma_pred = Phi_t @ S_1_0 @ Phi_t.mT + expanded_Sigma_eps

        w_1_0 = self.w_0_0.expand((S, B, -1, -1, -1, -1))  # [S, B,  K,  N,  D,  1]

        return td.MultivariateNormal(
            loc=w_1_0.squeeze(-1), covariance_matrix=S_1_0
        ), torch.diag_embed(Sigma_pred[..., 0, 0])

    def incorporate_batch(
        self,
        Phi_0_t: torch.Tensor,
        pis_0_t: torch.Tensor,
        Y_data: torch.Tensor,
        hist_lens: torch.Tensor,
        update_mode: UpdateMode,
        Sigma_t: Optional[torch.Tensor] = None,
    ) -> None:
        if update_mode == UpdateMode.BATCH_FROM_PRIOR:
            self.reset_to_prior()
        elif update_mode != UpdateMode.ONLINE_BATCH:
            raise ValueError(
                "update_mode must be one of {OFFLINE_BATCH, ONLINE_BATCH}."
            )

        # Looping from 0 to current timestep - 1
        history_size = Phi_0_t.shape[-3]
        for t in range(history_size):
            self.incorporate_transition(
                Phi_0_t[..., t, :, :],
                pis_0_t[..., t, :],
                Y_data[..., t, :],
                has_data_idxs=(hist_lens >= history_size - t),
                Sigma_t=Sigma_t[..., t, :] if Sigma_t is not None else None,
            )

    def incorporate_transition(
        self,
        Phi_tp1: torch.Tensor,  # [S, B, K, N, D]
        pis_tp1: torch.Tensor,  # [S, B, K]
        y_tp1: torch.Tensor,  # [S, B, K, N]
        has_data_idxs: Optional[torch.Tensor] = None,  # [B]
        Sigma_t: Optional[torch.Tensor] = None,
    ) -> None:
        S, B = Phi_tp1.shape[:2]
        Phi_tp1 = Phi_tp1.unsqueeze(-2)  # [S, B, K, N, 1, D]

        if has_data_idxs is None:
            has_data_idxs = torch.ones(
                Phi_tp1.shape[1], device=Phi_tp1.device, dtype=bool
            )

        # Adding dimensions to make it compatible with the other 6D tensors in this function.
        has_data_idxs = has_data_idxs[None, :, None, None, None, None]
        pis_tp1 = pis_tp1[:, :, :, None, None, None]  # [S, B, K, 1, 1, 1]

        # Prediction Step
        self.w_tp1_t = self.w_t_t  # A = I, b = 0, [S, B, K, N, D, 1]
        self.S_tp1_t = self.S_t_t + self.Sigma_nu  # A = I, [S, B, K, N, D, D]

        # Correction Step
        expanded_Sigma_eps: torch.Tensor = (
            self._expanded_Sigma_eps(S, B)
            if Sigma_t is None
            else Sigma_t[:, :, :, :, None, None]
        )
        P_tp1: torch.Tensor = Phi_tp1 @ self.S_tp1_t @ Phi_tp1.mT + expanded_Sigma_eps
        K_tp1: torch.Tensor = self.S_tp1_t @ torch.linalg.solve(P_tp1.mT, Phi_tp1).mT
        e_tp1: torch.Tensor = y_tp1.unsqueeze(-1).unsqueeze(-1) - (
            Phi_tp1 @ self.w_tp1_t
        )

        # This helps handle datapoints for which there
        # is no history (e.g., we should not be making an update for them).
        self.w_t_t = self.w_tp1_t + has_data_idxs * pis_tp1 * K_tp1 @ e_tp1
        self.S_t_t = self.S_tp1_t - has_data_idxs * pis_tp1 * K_tp1 @ (
            Phi_tp1 @ self.S_tp1_t
        )
        self.num_updates += 1

    def get_posterior(
        self,
        Phi_tp1: torch.Tensor,  # [S, B, K, N, D]
        single_mode_multi_sample: bool = False,
    ) -> Tuple[td.MultivariateNormal, Optional[torch.Tensor]]:
        S, B = Phi_tp1.shape[:2]

        ret_Sigma_pred: Optional[torch.Tensor] = None
        if not single_mode_multi_sample:
            Phi_tp1 = Phi_tp1.unsqueeze(-2)  # [S, B, K, N, 1, D]

            expanded_Sigma_eps: torch.Tensor = self._expanded_Sigma_eps(S, B)

            Sigma_pred: torch.Tensor = (
                Phi_tp1 @ self.S_tp1_t @ Phi_tp1.mT + expanded_Sigma_eps
            )
            ret_Sigma_pred = torch.diag_embed(Sigma_pred[..., 0, 0])

        return (
            td.MultivariateNormal(
                loc=self.w_tp1_t.squeeze(-1).expand(S, B, -1, -1, -1),
                covariance_matrix=self.S_tp1_t.expand(S, B, -1, -1, -1, -1),
            ),
            ret_Sigma_pred,
        )

    def forward(self, Phi_data, Y_data, z_logits, ph):
        raise NotImplementedError(
            "BayesianLastLayer's forward is not meant to be directly used, please use its other methods."
        )
