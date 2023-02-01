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

import torch

from trajectron.model.components import GMM2D
from trajectron.model.dynamics import Dynamic
from trajectron.utils import block_diag


class SingleIntegrator(Dynamic):
    def integrate_samples(self, v, x, dt):
        """
        Integrates deterministic samples of velocity.

        :param v: Velocity samples
        :param x: Not used for SI.
        :return: Position samples
        """
        p_0 = self.initial_conditions["pos"].unsqueeze(1).unsqueeze(0)
        return torch.cumsum(v * dt[None, :, None, None], dim=2) + p_0

    def integrate_distribution(self, v_dist, x, dt):
        r"""
        Integrates the GMM velocity distribution to a distribution over position.
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

        :param v_dist: Joint GMM Distribution over velocity in x and y direction.
        :param x: Not used for SI.
        :return: Joint GMM Distribution over position in x and y direction.
        """
        # Adding all these dimensions to dt to match the shape of v_dist.mus
        dt = dt[:, None, None, None]
        p_0 = self.initial_conditions["pos"].unsqueeze(1)
        ph = v_dist.mus.shape[-3]
        sample_batch_dim = list(v_dist.mus.shape[0:2])
        pos_dist_sigma_matrix_list = []

        pos_mus = p_0[:, None] + torch.cumsum(v_dist.mus, dim=2) * dt

        vel_dist_sigma_matrix = v_dist.get_covariance_matrix()
        pos_dist_sigma_matrix_t = torch.zeros(
            sample_batch_dim + [v_dist.components, 2, 2], device=self.device
        )

        F = torch.eye(4, device=self.device, dtype=torch.float32).repeat(
            dt.shape[0], 1, 1, 1
        )
        F[..., 0:2, 2:] = torch.eye(2, device=self.device, dtype=torch.float32) * dt
        F_t = F.mT

        for t in range(ph):
            vel_sigma_matrix_t = vel_dist_sigma_matrix[:, :, t]
            full_sigma_matrix_t = block_diag(
                [pos_dist_sigma_matrix_t, vel_sigma_matrix_t]
            )
            pos_dist_sigma_matrix_t = F[..., :2, :].matmul(
                full_sigma_matrix_t.matmul(F_t)[..., :2]
            )
            pos_dist_sigma_matrix_list.append(pos_dist_sigma_matrix_t)

        pos_dist_sigma_matrix = torch.stack(pos_dist_sigma_matrix_list, dim=2)
        return GMM2D.from_log_pis_mus_cov_mats(
            v_dist.log_pis, pos_mus, pos_dist_sigma_matrix
        )
