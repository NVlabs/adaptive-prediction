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

class Dynamic(object):
    def __init__(
        self, dyn_limits, device, model_registrar, xz_size, node_type, hyperparams
    ):
        self.device = device
        self.dyn_limits = dyn_limits
        self.initial_conditions = None
        self.model_registrar = model_registrar
        self.node_type = node_type
        self.hyperparams = hyperparams
        self.init_constants()
        self.create_graph(xz_size)

    def set_initial_condition(self, init_con):
        self.initial_conditions = init_con

    def init_constants(self):
        pass

    def create_graph(self, xz_size):
        pass

    def integrate_samples(self, s, x, dt):
        raise NotImplementedError

    def integrate_distribution(self, dist, x, dt):
        raise NotImplementedError

    def create_graph(self, xz_size):
        pass
