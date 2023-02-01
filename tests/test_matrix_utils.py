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

from trajectron.utils.matrix_utils import mat


def test_mat():
    x = torch.tensor([1, 2, 3, 4])

    assert torch.allclose(
        mat(x, rows=2),
        torch.tensor(
            [[1, 0, 2, 0, 3, 0, 4, 0], [0, 1, 0, 2, 0, 3, 0, 4]], dtype=torch.float
        ),
    )

    x = torch.tensor([1, 2])

    assert torch.allclose(
        mat(x), torch.tensor([[1, 0, 2, 0], [0, 1, 0, 2]], dtype=torch.float)
    )
