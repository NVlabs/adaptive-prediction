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

from itertools import product

import numpy as np
import orjson

from .node_type import NodeTypeEnum


class Environment(object):
    def __init__(
        self,
        node_type_list,
        standardization,
        scenes=None,
        attention_radius=None,
        robot_type=None,
        dt=None,
    ):
        self.scenes = scenes
        self.node_type_list = node_type_list
        self.attention_radius = attention_radius
        self.NodeType = NodeTypeEnum(node_type_list)
        self.robot_type = robot_type
        self.dt = dt

        self.standardization = standardization
        self.standardize_param_memo = dict()

        self._scenes_resample_prop = None

    def get_edge_types(self):
        return list(product(self.NodeType, repeat=2))

    def get_standardize_params(self, state, node_type):
        memo_key = (orjson.dumps(state), node_type)
        if memo_key in self.standardize_param_memo:
            return self.standardize_param_memo[memo_key]

        standardize_mean_list = list()
        standardize_std_list = list()
        for entity, dims in state.items():
            for dim in dims:
                standardize_mean_list.append(
                    self.standardization[node_type][entity][dim]["mean"]
                )
                standardize_std_list.append(
                    self.standardization[node_type][entity][dim]["std"]
                )
        standardize_mean = np.stack(standardize_mean_list)
        standardize_std = np.stack(standardize_std_list)

        self.standardize_param_memo[memo_key] = (standardize_mean, standardize_std)
        return standardize_mean, standardize_std

    def standardize(self, array, state, node_type, mean=None, std=None):
        if mean is None and std is None:
            mean, std = self.get_standardize_params(state, node_type)
        elif mean is None and std is not None:
            mean, _ = self.get_standardize_params(state, node_type)
        elif mean is not None and std is None:
            _, std = self.get_standardize_params(state, node_type)
        return np.where(np.isnan(array), np.array(np.nan), (array - mean) / std)

    def unstandardize(self, array, state, node_type, mean=None, std=None):
        if mean is None and std is None:
            mean, std = self.get_standardize_params(state, node_type)
        elif mean is None and std is not None:
            mean, _ = self.get_standardize_params(state, node_type)
        elif mean is not None and std is None:
            _, std = self.get_standardize_params(state, node_type)
        return array * std + mean

    @property
    def scenes_resample_prop(self):
        if self._scenes_resample_prop is None:
            self._scenes_resample_prop = np.array(
                [scene.resample_prob for scene in self.scenes]
            )
            self._scenes_resample_prop = self._scenes_resample_prop / np.sum(
                self._scenes_resample_prop
            )
        return self._scenes_resample_prop


class EnvironmentMetadata(Environment):
    """The purpose of this class is to provide the exact same data that an Environment object does, but without the
    huge scenes list (which makes this easy to serialize for pickling, e.g., for multiprocessing).
    """

    def __init__(
        self,
        node_type_list,
        standardization,
        attention_radius,
        robot_type,
        dt,
        standardize_param_memo,
        scenes_resample_prop,
    ):
        super(EnvironmentMetadata, self).__init__(
            node_type_list=node_type_list,
            standardization=standardization,
            scenes=None,
            attention_radius=attention_radius,
            robot_type=robot_type,
            dt=dt,
        )
        self.standardize_param_memo = standardize_param_memo
        self._scenes_resample_prop = scenes_resample_prop

    @classmethod
    def from_env(cls, env: Environment):
        return cls(
            node_type_list=env.node_type_list,
            standardization=env.standardization,
            attention_radius=env.attention_radius,
            robot_type=env.robot_type,
            dt=env.dt,
            standardize_param_memo=env.standardize_param_memo,
            scenes_resample_prop=env._scenes_resample_prop,
        )
