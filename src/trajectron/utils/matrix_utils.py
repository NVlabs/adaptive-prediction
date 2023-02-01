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
import torch.nn as nn


class Symmetric(nn.Module):
    def forward(self, X: torch.Tensor):
        return X.triu() + X.triu(1).mT  # Return a symmetric matrix

    def right_inverse(self, A: torch.Tensor):
        return A.triu()


class Positive(nn.Module):
    def forward(self, X: torch.Tensor):
        return torch.exp(X)

    def right_inverse(self, A: torch.Tensor):
        return torch.log(A)


def sym_mat_inv(M: torch.Tensor):
    return torch.cholesky_inverse(torch.linalg.cholesky(M))


def vec(M: torch.Tensor):
    return torch.flatten(M.mT, start_dim=-2)


def kron(A, B):
    """
    Kronecker product of matrices A and B with leading batch dimensions.
    Batch dimensions are broadcast. From: https://gist.github.com/yulkang/4a597bcc5e9ccf8c7291f8ecb776382d
    :type A: torch.Tensor
    :type B: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(A.shape[-2:]) * torch.tensor(B.shape[-2:]))
    res = A.unsqueeze(-1).unsqueeze(-3) * B.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)


def mat(x: torch.Tensor, rows=None):
    if rows is None:
        rows = x.shape[-1]
    elif rows > x.shape[-1]:
        raise ValueError(
            f"The number of output rows ({rows}) cannot be greater than x.shape[-1] ({x.shape[-1]})"
        )
    return kron(x[..., None, :], torch.eye(rows, device=x.device))


def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend) + v.shape + torch.Size([1] * n_dim_to_append)
    )


def block_diag(m):
    """
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n, device=m.device).unsqueeze(-2), d - 3, 1)
    return (m2 * eye).reshape(siz0 + torch.Size(torch.tensor(siz1) * n))


def tile(a, dim, n_tile, device="cpu"):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
    ).to(device)
    return torch.index_select(a, dim, order_index)
