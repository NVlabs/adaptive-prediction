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

import functools
import math
from enum import Enum

import numpy as np
import torch
import torch.nn.utils.rnn as rnn


class ModeKeys(Enum):
    TRAIN = 1
    EVAL = 2
    PREDICT = 3


class UpdateMode(Enum):
    BATCH_FROM_PRIOR = 1
    ITERATIVE = 2
    ONLINE_BATCH = 3
    NO_UPDATE = 4


def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3, decay=1.0):
    # Lambda function to calculate the LR
    lr_lambda = (
        lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize) * decay**it
    )

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x))

    return lr_lambda


def to_one_hot(labels, n_labels):
    return torch.eye(n_labels, device=labels.device)[labels]


def exp_anneal(anneal_kws):
    device = anneal_kws["device"]
    start = torch.tensor(anneal_kws["start"], device=device)
    finish = torch.tensor(anneal_kws["finish"], device=device)
    rate = torch.tensor(anneal_kws["rate"], device=device)
    return lambda step: finish - (finish - start) * torch.pow(
        rate, torch.tensor(step, dtype=torch.float, device=device)
    )


def sigmoid_anneal(anneal_kws):
    device = anneal_kws["device"]
    start = torch.tensor(anneal_kws["start"], device=device)
    finish = torch.tensor(anneal_kws["finish"], device=device)
    center_step = torch.tensor(
        anneal_kws["center_step"], device=device, dtype=torch.float
    )
    steps_lo_to_hi = torch.tensor(
        anneal_kws["steps_lo_to_hi"], device=device, dtype=torch.float
    )
    return lambda step: start + (finish - start) * torch.sigmoid(
        (torch.tensor(float(step), device=device) - center_step)
        * (1.0 / steps_lo_to_hi)
    )


class CustomLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        super(CustomLR, self).__init__(optimizer, lr_lambda, last_epoch)

    def get_lr(self):
        return [
            lmbda(self.last_epoch)
            for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)
        ]


def mutual_inf_mc(x_dist):
    dist = x_dist.__class__
    H_y = dist(probs=x_dist.probs.mean(dim=0)).entropy()
    return (H_y - x_dist.entropy().mean(dim=0)).sum()


def roll_by_gather(mat: torch.Tensor, dim: int, shifts: torch.LongTensor):
    # assumes 3D array
    batch, ts, dim = mat.shape

    arange1 = (
        torch.arange(ts, device=shifts.device)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(batch, -1, dim)
    )
    # print(arange1)
    arange2 = (arange1 - shifts[:, None, None]) % ts
    # print(arange2)
    return torch.gather(mat, 1, arange2)


def run_lstm_on_variable_length_seqs(lstm_module, seqs, seq_lens):
    packed_seqs = rnn.pack_padded_sequence(
        seqs, seq_lens, batch_first=True, enforce_sorted=False
    )
    packed_output, (h_n, c_n) = lstm_module(packed_seqs)
    output, _ = rnn.pad_packed_sequence(
        packed_output, batch_first=True, total_length=seqs.shape[1]
    )

    return output, (h_n, c_n)


def extract_subtensor_per_batch_element(tensor, indices):
    batch_idxs = torch.arange(start=0, end=len(indices))

    batch_idxs = batch_idxs[~torch.isnan(indices)]
    indices = indices[~torch.isnan(indices)]
    if indices.size == 0:
        return None
    else:
        indices = indices.long()
    if tensor.is_cuda:
        batch_idxs = batch_idxs.to(tensor.get_device())
        indices = indices.to(tensor.get_device())
    return tensor[batch_idxs, indices]


def unpack_RNN_state(state_tuple):
    # PyTorch returned LSTM states have 3 dims:
    # (num_layers * num_directions, batch, hidden_size)

    state = torch.cat(state_tuple, dim=0).permute(1, 0, 2)
    # Now state is (batch, 2 * num_layers * num_directions, hidden_size)

    state_size = state.size()
    return torch.reshape(state, (-1, state_size[1] * state_size[2]))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# using wonder's beautiful simplification:
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))
