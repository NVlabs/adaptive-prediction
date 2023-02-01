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
import torch.nn as nn

from trajectron.model.components import GMM2D
from trajectron.model.dynamics import Dynamic
from trajectron.utils import block_diag


class Unicycle(Dynamic):
    def create_graph(self, xz_size):
        model_if_absent = nn.Linear(xz_size + 1, 1)
        self.p0_model = self.model_registrar.get_model(
            f"{self.node_type}/unicycle_initializer", model_if_absent
        )

    def dynamic(self, x_0, us, dt_orig):
        r"""
        TODO: Boris: Add docstring
        :param x_0: [4, B] which is (x, y, phi, v)
        :param u: [2, B, T, K (= components)] which is (dphi, longitudinal acc)
        :return:
        """
        # Making room for time and components dimensions.
        x_init = x_0.unsqueeze(-1).unsqueeze(-1)
        dphi = us[0]
        a = us[1]
        dt = dt_orig.unsqueeze(1)

        phi = x_init[2] + torch.cumsum(dphi * dt, dim=1)
        v = x_init[3] + torch.cumsum(a * dt, dim=1)

        mask = torch.abs(dphi) <= 1e-2
        dphi = ~mask * dphi + (mask) * 1

        dt_sq = torch.square(dt)
        phi_p_omega_dt = phi + dphi * dt
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        sin_phi_p_omega_dt = torch.sin(phi_p_omega_dt)
        cos_phi_p_omega_dt = torch.cos(phi_p_omega_dt)

        one_on_dphi = torch.reciprocal(dphi)
        dsin_domega = (sin_phi_p_omega_dt - sin_phi) * one_on_dphi
        dcos_domega = (cos_phi_p_omega_dt - cos_phi) * one_on_dphi
        a_on_dphi = a * one_on_dphi

        x_regular = x_init[0] + torch.cumsum(
            a_on_dphi * dcos_domega
            + v * dsin_domega
            + a_on_dphi * sin_phi_p_omega_dt * dt,
            dim=1,
        )
        y_regular = x_init[1] + torch.cumsum(
            -v * dcos_domega
            + a_on_dphi * dsin_domega
            - a_on_dphi * cos_phi_p_omega_dt * dt,
            dim=1,
        )

        x_small_dphi = x_init[0] + torch.cumsum(
            v * cos_phi * dt + (a / 2) * cos_phi * dt_sq, dim=1
        )
        y_small_dphi = x_init[1] + torch.cumsum(
            v * sin_phi * dt + (a / 2) * sin_phi * dt_sq, dim=1
        )

        states_regular = torch.stack((x_regular, y_regular, phi, v), dim=-1)
        states_small_dphi = torch.stack((x_small_dphi, y_small_dphi, phi, v), dim=-1)

        return torch.where(~mask.unsqueeze(-1), states_regular, states_small_dphi)

    def integrate_samples(self, control_samples, x, dt):
        r"""
        TODO: Boris: Add docstring
        :param x: (x, y, phi, v)
        :param u: (dphi, longitudinal acc)
        :return:
        """
        num_samples, batch_num, timesteps, control_dim = control_samples.shape
        p_0 = self.initial_conditions["pos"].unsqueeze(1)
        v_0 = self.initial_conditions["vel"].unsqueeze(1)

        # In case the input is batched because of the robot in online use we repeat this to match the batch size of x.
        if p_0.size()[0] != x.size()[0]:
            p_0 = p_0.repeat(x.size()[0], 1, 1)
            v_0 = v_0.repeat(x.size()[0], 1, 1)

        phi_0 = torch.atan2(v_0[..., 1], v_0[..., 0])

        phi_0 = phi_0 + torch.tanh(self.p0_model(torch.cat((x, phi_0), dim=-1)))

        u = control_samples.permute((3, 0, 1, 2)).unsqueeze(-1)
        x = torch.stack(
            [p_0[..., 0], p_0[..., 1], phi_0, torch.norm(v_0, dim=-1)], dim=0
        ).squeeze(dim=-1)

        integrated = self.dynamic(
            x, u.reshape(u.shape[0], -1, *u.shape[3:]), dt.unsqueeze(-1)
        )[..., 0, :2]
        return integrated.reshape(num_samples, batch_num, timesteps, control_dim)

    def compute_control_jacobians(
        self, xs: torch.Tensor, us: torch.Tensor, dt_orig: torch.Tensor
    ) -> torch.Tensor:
        r"""
        TODO: Boris: Add docstring
        :param x: (x, y, phi, v)
        :param u: (dphi, longitudinal acc)
        :return:
        """
        num_samples, batch_size, ph, num_components, state_dim = xs.shape
        control_dim = us.shape[-1]
        Gs = torch.zeros(
            (num_samples, batch_size, ph, num_components, state_dim, control_dim),
            device=self.device,
            dtype=torch.float32,
        )

        phi = xs[..., 2]
        v = xs[..., 3]
        dphi = us[..., 0]
        a = us[..., 1]
        dt = dt_orig.unsqueeze(0).unsqueeze(-1)
        dt_sq = torch.square(dt)

        mask = torch.abs(dphi) <= 1e-2
        dphi = ~mask * dphi + (mask) * 1

        phi_p_omega_dt = phi + dphi * dt
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        sin_phi_p_omega_dt = torch.sin(phi_p_omega_dt)
        cos_phi_p_omega_dt = torch.cos(phi_p_omega_dt)

        one_on_dphi = torch.reciprocal(dphi)
        dsin_domega = (sin_phi_p_omega_dt - sin_phi) * one_on_dphi
        dcos_domega = (cos_phi_p_omega_dt - cos_phi) * one_on_dphi

        v_on_dphi = v * one_on_dphi
        a_on_dphi = a * one_on_dphi
        two_a_on_dphi_sq = 2 * torch.square(a_on_dphi)

        Gs[..., 0, 0] = (
            v_on_dphi * cos_phi_p_omega_dt * dt
            - v_on_dphi * dsin_domega
            - two_a_on_dphi_sq * sin_phi_p_omega_dt * dt
            - two_a_on_dphi_sq * dcos_domega
            + a_on_dphi * cos_phi_p_omega_dt * dt_sq
        )
        Gs[..., 0, 1] = (
            one_on_dphi * dcos_domega + one_on_dphi * sin_phi_p_omega_dt * dt
        )

        Gs[..., 1, 0] = (
            v_on_dphi * dcos_domega
            - two_a_on_dphi_sq * dsin_domega
            + two_a_on_dphi_sq * cos_phi_p_omega_dt * dt
            + v_on_dphi * sin_phi_p_omega_dt * dt
            + a_on_dphi * sin_phi_p_omega_dt * dt_sq
        )
        Gs[..., 1, 1] = (
            one_on_dphi * dsin_domega - one_on_dphi * cos_phi_p_omega_dt * dt
        )

        Gs[..., 2, 0] = dt

        Gs[..., 3, 1] = dt

        Gs_sm = torch.zeros(
            (num_samples, batch_size, ph, num_components, state_dim, control_dim),
            device=self.device,
            dtype=torch.float32,
        )

        Gs_sm[..., 0, 1] = cos_phi * dt_sq / 2

        Gs_sm[..., 1, 1] = sin_phi * dt_sq / 2

        Gs_sm[..., 3, 1] = dt

        return torch.where(~mask.unsqueeze(-1).unsqueeze(-1), Gs, Gs_sm)

    def compute_jacobians(
        self, xs: torch.Tensor, us: torch.Tensor, dt_orig: torch.Tensor
    ) -> torch.Tensor:
        r"""
        TODO: Boris: Add docstring
        :param x: (x, y, phi, v)
        :param u: (dphi, longitudinal acc)
        :return:
        """
        num_samples, batch_size, ph, num_components, state_dim = xs.shape
        Fs = torch.zeros(
            (num_samples, batch_size, ph, num_components, state_dim, state_dim),
            device=xs.device,
            dtype=xs.dtype,
        )

        phi = xs[..., 2]
        v = xs[..., 3]
        dphi = us[..., 0]
        a = us[..., 1]
        dt = dt_orig.unsqueeze(0).unsqueeze(-1)
        dt_sq = torch.square(dt)

        mask = torch.abs(dphi) <= 1e-2
        dphi = ~mask * dphi + (mask) * 1

        phi_p_omega_dt = phi + dphi * dt
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        sin_phi_p_omega_dt = torch.sin(phi_p_omega_dt)
        cos_phi_p_omega_dt = torch.cos(phi_p_omega_dt)

        one_on_dphi = torch.reciprocal(dphi)
        dsin_domega = (sin_phi_p_omega_dt - sin_phi) * one_on_dphi
        dcos_domega = (cos_phi_p_omega_dt - cos_phi) * one_on_dphi

        a_on_dphi = a * one_on_dphi

        Fs[..., 0, 0] = 1
        Fs[..., 1, 1] = 1
        Fs[..., 2, 2] = 1
        Fs[..., 3, 3] = 1

        Fs[..., 0, 2] = (
            v * dcos_domega
            - a_on_dphi * dsin_domega
            + a_on_dphi * cos_phi_p_omega_dt * dt
        )
        Fs[..., 0, 3] = dsin_domega

        Fs[..., 1, 2] = (
            v * dsin_domega
            + a_on_dphi * dcos_domega
            + a_on_dphi * sin_phi_p_omega_dt * dt
        )
        Fs[..., 1, 3] = -dcos_domega

        Fs_sm = torch.zeros(
            (num_samples, batch_size, ph, num_components, state_dim, state_dim),
            device=self.device,
            dtype=torch.float32,
        )

        Fs_sm[..., 0, 0] = 1
        Fs_sm[..., 1, 1] = 1
        Fs_sm[..., 2, 2] = 1
        Fs_sm[..., 3, 3] = 1

        Fs_sm[..., 0, 2] = -v * sin_phi * dt - a * sin_phi * dt_sq / 2
        Fs_sm[..., 0, 3] = cos_phi * dt

        Fs_sm[..., 1, 2] = v * cos_phi * dt + a * cos_phi * dt_sq / 2
        Fs_sm[..., 1, 3] = sin_phi * dt

        return torch.where(~mask.unsqueeze(-1).unsqueeze(-1), Fs, Fs_sm)

    def integrate_distribution(
        self,
        control_dist_dphi_a: GMM2D,
        encoded_context: torch.Tensor,
        dt: torch.Tensor,
    ):
        """_summary_

        Args:
            control_dist_dphi_a (GMM2D): _description_
            encoded_context (torch.Tensor): _description_
            dt (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        (
            num_samples,
            batch_dim,
            ph,
            num_components,
            control_dim,
        ) = control_dist_dphi_a.mus.shape
        p_0 = self.initial_conditions["pos"].unsqueeze(1)
        v_0 = self.initial_conditions["vel"].unsqueeze(1)

        # In case the input is batched because of the robot in online use we repeat this to match the batch size of x.
        if p_0.shape[0] != encoded_context.shape[0]:
            p_0 = p_0.repeat(encoded_context.shape[0], 1, 1)
            v_0 = v_0.repeat(encoded_context.shape[0], 1, 1)

        phi_0 = torch.atan2(v_0[..., 1], v_0[..., 0])

        phi_0 = phi_0 + torch.tanh(
            self.p0_model(torch.cat((encoded_context, phi_0), dim=-1))
        )

        # Adding a new dimension so dt's resulting
        # (batch_shape, 1) shape broadcasts easily.
        dt = dt[:, None]
        us = control_dist_dphi_a.mus.permute((4, 0, 1, 2, 3))
        x_0 = torch.stack(
            [p_0[..., 0], p_0[..., 1], phi_0, torch.norm(v_0, dim=-1)], dim=0
        )

        # Combining the sample and batch dimensions here,
        # since they're essentially just a batch here.
        xs: torch.Tensor = self.dynamic(
            x_0.reshape(x_0.shape[0], -1),
            us.reshape(us.shape[0], -1, *us.shape[3:]),
            dt,
        )  # [S*B, T, K, 4]

        # [S, B, T, K, 4]
        xs = xs.reshape(num_samples, batch_dim, *xs.shape[1:])

        # Precomputing all the Jacobians we'll need.
        us = us.permute((1, 2, 3, 4, 0))
        Fs = self.compute_jacobians(xs, us, dt)
        Gs = self.compute_control_jacobians(xs, us, dt)

        # Getting Sigma_u
        dist_sigma_matrix: torch.Tensor = control_dist_dphi_a.get_covariance_matrix()

        # Precomputing all G @ Sigma_u @ G^T
        quadform_G_Sigmau: torch.Tensor = Gs.matmul(dist_sigma_matrix.matmul(Gs.mT))

        # Precomputing the first Sigma_x, which is easy as
        # Sigma_x_0 = F_0 @ ZERO @ F_0^T + G_0 @ Sigma_u_0 @ G_0^T
        #           = G_0 @ Sigma_u_0 @ G_0^T
        state_dist_sigma_matrix_t: torch.Tensor = quadform_G_Sigmau[:, :, 0]

        pos_dist_sigma_matrix: torch.Tensor = torch.empty(
            (num_samples, batch_dim, ph, num_components, 2, 2),
            dtype=xs.dtype,
            device=xs.device,
        )
        pos_dist_sigma_matrix[:, :, 0] = state_dist_sigma_matrix_t[..., :2, :2]
        for t in range(1, ph):
            F_t = Fs[:, :, t]
            state_dist_sigma_matrix_t = (
                F_t.matmul(state_dist_sigma_matrix_t.matmul(F_t.mT))
                + quadform_G_Sigmau[:, :, t]
            )
            pos_dist_sigma_matrix[:, :, t] = state_dist_sigma_matrix_t[..., :2, :2]

        return GMM2D.from_log_pis_mus_cov_mats(
            log_pis=control_dist_dphi_a.log_pis,
            mus=xs[..., :2],
            cov_mats=pos_dist_sigma_matrix,
        )
