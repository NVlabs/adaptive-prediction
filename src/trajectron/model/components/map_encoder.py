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
import torch.nn.functional as F


class CNNMapEncoder(nn.Module):
    def __init__(
        self, map_channels, hidden_channels, output_size, masks, strides, patch_size
    ):
        super(CNNMapEncoder, self).__init__()
        self.convs = nn.ModuleList()
        input_size = (map_channels, patch_size, patch_size)
        x_dummy = torch.ones(input_size).unsqueeze(0) * torch.tensor(float("nan"))

        for i, hidden_size in enumerate(hidden_channels):
            self.convs.append(
                nn.Conv2d(
                    map_channels if i == 0 else hidden_channels[i - 1],
                    hidden_channels[i],
                    masks[i],
                    stride=strides[i],
                )
            )
            x_dummy = self.convs[i](x_dummy)

        self.fc = nn.Linear(x_dummy.numel(), output_size)

    def forward(self, x, training):
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
