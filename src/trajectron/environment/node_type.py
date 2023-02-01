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

class NodeType(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if type(other) == str and self.name == other:
            return True
        else:
            return isinstance(other, self.__class__) and self.name == other.name

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name)

    def __add__(self, other):
        return self.name + other


class NodeTypeEnum(list):
    def __init__(self, node_type_list):
        self.node_type_list = node_type_list
        node_types = [
            NodeType(name, node_type_list.index(name) + 1) for name in node_type_list
        ]
        super().__init__(node_types)

    def __getattr__(self, name):
        if not name.startswith("_") and name in object.__getattribute__(
            self, "node_type_list"
        ):
            return self[object.__getattribute__(self, "node_type_list").index(name)]
        else:
            return object.__getattribute__(self, name)
