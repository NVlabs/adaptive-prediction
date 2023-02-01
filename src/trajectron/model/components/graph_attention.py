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

import math
import numbers
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init


class GraphMultiTypeAttention(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, bias=True, types=1):
        super(GraphMultiTypeAttention, self).__init__()
        self.types = types
        self.in_features = in_features
        self.out_features = out_features
        self.node_self_loop_weight = Parameter(
            torch.Tensor(hidden_features, in_features[0])
        )

        self.weight_per_type = nn.ParameterList()
        for i in range(types):
            self.weight_per_type.append(
                Parameter(torch.Tensor(hidden_features, in_features[i]))
            )
        if bias:
            self.bias = Parameter(torch.Tensor(hidden_features))
        else:
            self.register_parameter("bias", None)

        self.linear_to_out = nn.Linear(hidden_features, out_features, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weight_per_type:
            bound = 1 / math.sqrt(weight.size(1))
            init.uniform_(weight, -bound, bound)
        bound = 1 / math.sqrt(self.node_self_loop_weight.size(1))
        init.uniform_(self.node_self_loop_weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs, types, edge_weights):
        weight_list = list()
        for i, type in enumerate(types):
            weight_list.append(
                (edge_weights[i] / len(edge_weights)) * self.weight_per_type[type].T
            )
        weight_list.append(self.node_self_loop_weight.T)
        weight = torch.cat(weight_list, dim=0)
        stacked_input = torch.cat(inputs, dim=-1)
        output = stacked_input.matmul(weight)

        output = output

        if self.bias is not None:
            output += self.bias

        return torch.relu(self.linear_to_out(torch.relu(output)))

    def extra_repr(self):
        return "in_features={}, hidden_features={},, out_features={}, types={}, bias={}".format(
            self.in_features,
            self.hidden_features,
            self.out_features,
            self.types,
            self.bias is not None,
        )
