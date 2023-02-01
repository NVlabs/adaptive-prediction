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

from trajectron.model.components.gmm2d import GMM2D


def test_gmm2d():
    # Tensors are of shape (S, B, T, K[, D])
    dist = GMM2D(
        log_pis=torch.full((1, 1, 1, 2), fill_value=0.5),
        mus=torch.concat(
            [torch.zeros((1, 1, 1, 1, 2)), torch.ones((1, 1, 1, 1, 2))], dim=-2
        ),
        log_sigmas=-torch.ones((1, 1, 1, 2, 2)) * 2,
        corrs=torch.zeros((1, 1, 1, 2)),
    )

    print()
