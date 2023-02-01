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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trajectron.utils.matrix_utils import sym_mat_inv

np.random.seed(0)
torch.manual_seed(0)


def test_cholesky():
    phi_dim = 64
    num_GMM_components = 25

    glorot = sqrt(3 / phi_dim)
    L_chol_0_vec = glorot * (
        torch.rand((num_GMM_components, phi_dim * (phi_dim + 1) // 2)) * 2 - 1
    )
    L_chol_0_vec_param = torch.nn.Parameter(data=L_chol_0_vec)

    tril_indices = torch.tril_indices(phi_dim, phi_dim)
    L_chol_0 = torch.zeros((num_GMM_components, phi_dim, phi_dim))
    L_chol_0[:, tril_indices[0], tril_indices[1]] = L_chol_0_vec_param

    # Ensuring the diagonal is positive by using a Softplus transformation followed by a small shift which
    # attempts to avoid numerical issues from zeros on the diagonal (from tfp.FillScaleTriL).
    diag_indices = range(phi_dim)
    L_chol_0[:, diag_indices, diag_indices] = (
        F.softplus(torch.diagonal(L_chol_0, dim1=-2, dim2=-1)) + 1e-5
    )

    L = L_chol_0 @ torch.transpose(
        L_chol_0, -1, -2
    )  # (num_components, phi_dim, phi_dim)

    eigvals = torch.linalg.eigvalsh(L)

    L_inv = torch.linalg.inv(L)

    L_inv_chol = sym_mat_inv(L)

    assert torch.allclose(
        L_inv @ L, torch.eye(phi_dim).unsqueeze(0), rtol=1e-3, atol=1e-5
    )
    assert torch.allclose(
        L_inv_chol @ L, torch.eye(phi_dim).unsqueeze(0), rtol=1e-3, atol=1e-5
    )
