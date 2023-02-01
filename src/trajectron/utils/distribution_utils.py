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


def dist_mode(dist: GMM2D) -> torch.Tensor:
    # Probs are the same across timesteps.
    probs = dist.pis_cat_dist.probs[..., 0, :]
    argmax_probs = probs.reshape(-1, probs.shape[-1]).argmax(dim=-1)
    ml_means = dist.mus[0, torch.arange(argmax_probs.shape[0]), :, argmax_probs]
    return ml_means
