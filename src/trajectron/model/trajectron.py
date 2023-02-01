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
from typing import Any, Dict, List, Union

import numpy as np
import torch
from torch import nn
from trajdata import AgentBatch, AgentType

import trajectron.evaluation as evaluation
from trajectron.model.components import BayesianLastLayer
from trajectron.model.mgcvae import MultimodalGenerativeCVAE
from trajectron.model.model_utils import UpdateMode


class Trajectron(nn.Module):
    def __init__(self, model_registrar, hyperparams, log_writer, device):
        super(Trajectron, self).__init__()
        self.hyperparams = hyperparams
        self.log_writer = log_writer
        self.device = device
        self.curr_iter = 0

        self.model_registrar = model_registrar
        self.node_models_dict = nn.ModuleDict()
        self.nodes = set()

        self.state = self.hyperparams["state"]
        self.state_length = dict()
        for state_type in self.state.keys():
            self.state_length[state_type] = int(
                np.sum(
                    [
                        len(entity_dims)
                        for entity_dims in self.state[state_type].values()
                    ]
                )
            )
        self.pred_state = self.hyperparams["pred_state"]

    def set_environment(self):
        self.node_models_dict.clear()
        edge_types = list(product(AgentType, repeat=2))

        for node_type in AgentType:
            # Only add a Model for NodeTypes we want to predict
            if node_type.name in self.pred_state.keys():
                self.node_models_dict[node_type.name] = MultimodalGenerativeCVAE(
                    node_type,
                    self.model_registrar,
                    self.hyperparams,
                    self.device,
                    edge_types,
                    log_writer=self.log_writer,
                )

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter
        for node_str, model in self.node_models_dict.items():
            model.set_curr_iter(curr_iter)

    def set_annealing_params(self):
        for node_str, model in self.node_models_dict.items():
            model.set_annealing_params()

    def step_annealers(self):
        for node_type in self.node_models_dict:
            self.node_models_dict[node_type].step_annealers()

    def forward(self, batch, update_mode: UpdateMode = UpdateMode.BATCH_FROM_PRIOR):
        return self.train_loss(batch, update_mode=update_mode)

    def train_loss(
        self, batch: AgentBatch, update_mode: UpdateMode = UpdateMode.BATCH_FROM_PRIOR
    ):
        batch.to(self.device)

        # Run forward pass
        losses: List[torch.Tensor] = list()
        node_type: AgentType
        for node_type in batch.agent_types():
            model: MultimodalGenerativeCVAE = self.node_models_dict[node_type.name]

            agent_type_batch = batch.for_agent_type(node_type)
            losses.append(model(agent_type_batch, update_mode))

        return sum(losses)

    def predict_and_evaluate_batch(
        self,
        batch: AgentBatch,
        update_mode: UpdateMode = UpdateMode.NO_UPDATE,
        output_for_pd: bool = False,
    ) -> Union[List[Dict[str, Any]], Dict[AgentType, Dict[str, torch.Tensor]]]:
        """Predicts from a batch and then evaluates the output, returning the batched errors."""
        batch.to(self.device)

        # Run forward pass
        if output_for_pd:
            results: List[Dict[str, Any]] = list()
        else:
            results: Dict[AgentType, Dict[str, torch.Tensor]] = dict()

        node_type: AgentType
        for node_type in batch.agent_types():
            model: MultimodalGenerativeCVAE = self.node_models_dict[node_type.name]

            agent_type_batch = batch.for_agent_type(node_type)

            ph = agent_type_batch.agent_fut.shape[1]

            predictions = model.predict(
                agent_type_batch,
                prediction_horizon=ph,
                num_samples=1,
                z_mode=True,
                gmm_mode=True,
                full_dist=False,
                output_dists=False,
                update_mode=update_mode,
            )

            # Run forward pass
            y_dists, _ = model.predict(
                agent_type_batch,
                prediction_horizon=ph,
                num_samples=1,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                output_dists=True,
                update_mode=update_mode,
            )

            batch_eval: Dict[
                str, torch.Tensor
            ] = evaluation.compute_batch_statistics_pt(
                agent_type_batch.agent_fut[..., :2],
                prediction_output_dict=predictions,
                y_dists=y_dists,
            )

            if output_for_pd:
                batch_eval["data_idx"] = agent_type_batch.data_idx
                results.append(batch_eval)
            else:
                results[node_type] = batch_eval

        return results

    def predict(
        self,
        batch: AgentBatch,
        update_mode: UpdateMode = UpdateMode.NO_UPDATE,
        num_samples=1,
        prediction_horizon=None,
        z_mode=False,
        gmm_mode=False,
        full_dist=True,
        all_z_sep=False,
        output_dists=True,
    ):
        """Obtains model predictions for a batch of data.

        Args:
            batch (AgentBatch): _description_
            num_samples (int, optional): _description_. Defaults to 1.
            prediction_horizon (_type_, optional): _description_. Defaults to None.
            z_mode (bool, optional): _description_. Defaults to False.
            gmm_mode (bool, optional): _description_. Defaults to False.
            full_dist (bool, optional): _description_. Defaults to True.
            all_z_sep (bool, optional): _description_. Defaults to False.
            output_dists (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        batch.to(self.device)

        predictions_dict = {}
        dists_dict = {}

        node_type: AgentType
        for node_type in batch.agent_types():
            model: MultimodalGenerativeCVAE = self.node_models_dict[node_type.name]

            agent_type_batch = batch.for_agent_type(node_type)

            if prediction_horizon is None:
                ph = agent_type_batch.agent_fut.shape[1]
            else:
                ph = prediction_horizon

            # Run forward pass
            pred_object = model.predict(
                agent_type_batch,
                prediction_horizon=ph,
                num_samples=num_samples,
                z_mode=z_mode,
                gmm_mode=gmm_mode,
                full_dist=full_dist,
                all_z_sep=all_z_sep,
                output_dists=output_dists,
                update_mode=update_mode,
            )

            if output_dists:
                y_dists, predictions = pred_object
            else:
                predictions = pred_object

            predictions_np = predictions.cpu().detach().numpy()
            if output_dists:
                y_dists.set_device(torch.device("cpu"))

            # Assign predictions to node
            for i, agent_name in enumerate(agent_type_batch.agent_name):
                predictions_dict[f"{str(node_type)}/{agent_name}"] = predictions_np[
                    :, i
                ]
                if output_dists:
                    dists_dict[f"{str(node_type)}/{agent_name}"] = y_dists.get_for_node(
                        i
                    )

        if output_dists:
            return dists_dict, predictions_dict
        else:
            return predictions_dict

    def reset_adaptive_info(self):
        model: MultimodalGenerativeCVAE
        for model in self.node_models_dict.values():
            blr_layer: BayesianLastLayer = model.node_modules[
                model.node_type + "/decoder/last_layer"
            ]
            blr_layer.reset_to_prior()

    def adaptive_predict(
        self,
        batch: AgentBatch,
        update_mode: UpdateMode,
        num_samples=1,
        prediction_horizon=None,
        z_mode=False,
        gmm_mode=False,
        full_dist=True,
        all_z_sep=False,
        output_dists=True,
    ):
        """Obtains model predictions for a batch of data, but also keeps track of
        the Bayesian last-layer's posterior information (Kn and Ln).

        Args:
            batch (AgentBatch): _description_
            num_samples (int, optional): _description_. Defaults to 1.
            prediction_horizon (_type_, optional): _description_. Defaults to None.
            z_mode (bool, optional): _description_. Defaults to False.
            gmm_mode (bool, optional): _description_. Defaults to False.
            full_dist (bool, optional): _description_. Defaults to True.
            all_z_sep (bool, optional): _description_. Defaults to False.
            output_dists (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        batch.to(self.device)

        node_type: AgentType = batch.agent_types()[0]
        model: MultimodalGenerativeCVAE = self.node_models_dict[node_type.name]

        if prediction_horizon is None:
            ph = batch.agent_fut.shape[1]
        else:
            ph = prediction_horizon

        # Run forward pass
        return model.adaptive_predict(
            batch,
            prediction_horizon=ph,
            num_samples=num_samples,
            update_mode=update_mode,
            z_mode=z_mode,
            gmm_mode=gmm_mode,
            full_dist=full_dist,
            all_z_sep=all_z_sep,
            output_dists=output_dists,
        )
