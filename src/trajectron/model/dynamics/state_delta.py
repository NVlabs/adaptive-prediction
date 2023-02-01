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

from typing import Optional

import torch
import torch.distributions as td

from trajectron.model.components import GMM2D
from trajectron.model.dynamics import Dynamic


class StateDelta(Dynamic):
    def integrate_samples(self, delta: torch.Tensor, x: torch.Tensor, dt: torch.Tensor):
        """
        Integrates deterministic samples of delta position.

        :param v: Velocity samples
        :param x: Not used for SI.
        :return: Position samples
        """
        p_0 = self.initial_conditions["pos"].unsqueeze(1).unsqueeze(0)
        return torch.cumsum(delta, dim=2) + p_0

    def integrate_distribution(
        self, delta_dist: GMM2D, x: torch.Tensor, dt: torch.Tensor
    ):
        r"""
        Integrates the GMM delta position distribution to a distribution over position.
        The Kalman Equations are used.

        .. math:: \mu_{t+1} =\textbf{F} \mu_{t}

        .. math:: \mathbf{\Sigma}_{t+1}={\textbf {F}} \mathbf{\Sigma}_{t} {\textbf {F}}^{T}

        .. math::
            \textbf{F} = \left[
                            \begin{array}{cccc}
                                \sigma_x^2 & \rho_p \sigma_x \sigma_y & 0 & 0 \\
                                \rho_p \sigma_x \sigma_y & \sigma_y^2 & 0 & 0 \\
                                0 & 0 & \sigma_{v_x}^2 & \rho_v \sigma_{v_x} \sigma_{v_y} \\
                                0 & 0 & \rho_v \sigma_{v_x} \sigma_{v_y} & \sigma_{v_y}^2 \\
                            \end{array}
                        \right]_{t}

        :param v_dist: Joint GMM Distribution over delta positions in x and y direction.
        :param x: Not used for StateDelta.
        :return: Joint GMM Distribution over position in x and y direction.
        """
        p_0 = self.initial_conditions["pos"][None, :, None, None, :]
        pos_mus = torch.cumsum(delta_dist.mus, dim=2) + p_0

        delta_dist_sigma_matrix = delta_dist.get_covariance_matrix()
        pos_dist_sigma_matrix = torch.cumsum(delta_dist_sigma_matrix, dim=2)

        return GMM2D.from_log_pis_mus_cov_mats(
            delta_dist.pis_cat_dist.logits, pos_mus, pos_dist_sigma_matrix
        )

    def iterative_dist_integration(
        self, delta_dist: GMM2D, prev_dist: Optional[GMM2D] = None
    ):
        if prev_dist is None:
            return delta_dist

        return GMM2D.from_log_pis_mus_cov_mats(
            log_pis=delta_dist.pis_cat_dist.logits,
            mus=prev_dist.mus + delta_dist.mus,
            cov_mats=prev_dist.get_covariance_matrix()
            + delta_dist.get_covariance_matrix(),
        )
