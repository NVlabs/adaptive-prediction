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

import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
import torch.optim as optim
import trajdata.utils.arr_utils as arr_utils
import wandb
from torch.nn.utils.rnn import pack_padded_sequence
from trajdata import AgentBatch, AgentType

import trajectron.model.dynamics as dynamic_module
import trajectron.utils.matrix_utils as matrix_utils
from trajectron.environment.scene_graph import DirectedEdge
from trajectron.model.components import *
from trajectron.model.model_utils import *


class MultimodalGenerativeCVAE(nn.Module):
    def __init__(
        self,
        node_type_obj: AgentType,
        model_registrar,
        hyperparams: Dict,
        device,
        edge_types,
        log_writer=None,
    ):
        super(MultimodalGenerativeCVAE, self).__init__()

        self.hyperparams = hyperparams
        self.node_type_obj: AgentType = node_type_obj
        self.node_type: str = node_type_obj.name
        self.model_registrar = model_registrar
        self.log_writer = log_writer
        self.device = device
        self.edge_types: List[Tuple[AgentType, AgentType]] = [
            edge_type for edge_type in edge_types if edge_type[1] is node_type_obj
        ]
        self.curr_iter = 0

        self.node_modules = nn.ModuleDict()

        self.state = self.hyperparams["state"]
        self.pred_state = self.hyperparams["pred_state"][self.node_type]
        self.state_length = int(
            np.sum(
                [
                    len(entity_dims)
                    for entity_dims in self.state[self.node_type].values()
                ]
            )
        )
        if self.hyperparams["incl_robot_node"]:
            self.robot_state_length = int(
                np.sum(
                    [
                        len(entity_dims)
                        for entity_dims in self.state[AgentType.VEHICLE.name].values()
                    ]
                )
            )
        self.pred_state_length = int(
            np.sum([len(entity_dims) for entity_dims in self.pred_state.values()])
        )

        edge_types_str = [
            DirectedEdge.get_str_from_types(*edge_type) for edge_type in self.edge_types
        ]
        self.create_graphical_model(edge_types_str)

        dynamic_class = getattr(
            dynamic_module, hyperparams["dynamic"][self.node_type]["name"]
        )
        dyn_limits = hyperparams["dynamic"][self.node_type]["limits"]
        self.dynamic = dynamic_class(
            dyn_limits,
            device,
            self.model_registrar,
            self.x_size,
            self.node_type,
            self.hyperparams,
        )

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter

    def add_submodule(self, name, model_if_absent):
        self.node_modules[name] = self.model_registrar.get_model(name, model_if_absent)

    def clear_submodules(self):
        self.node_modules.clear()

    def create_node_models(self):
        ############################
        #   Node History Encoder   #
        ############################
        self.add_submodule(
            self.node_type + "/node_history_encoder",
            model_if_absent=nn.LSTM(
                input_size=self.state_length,
                hidden_size=self.hyperparams["enc_rnn_dim_history"],
                batch_first=True,
            ),
        )

        ###########################
        #   Node Future Encoder   #
        ###########################
        # We'll create this here, but then later check if in training mode.
        # Based on that, we'll factor this into the computation graph (or not).
        self.add_submodule(
            self.node_type + "/node_future_encoder",
            model_if_absent=nn.LSTM(
                input_size=self.pred_state_length,
                hidden_size=self.hyperparams["enc_rnn_dim_future"],
                bidirectional=True,
                batch_first=True,
            ),
        )
        # These are related to how you initialize states for the node future encoder.
        self.add_submodule(
            self.node_type + "/node_future_encoder/initial_h",
            model_if_absent=nn.Linear(
                self.state_length, self.hyperparams["enc_rnn_dim_future"]
            ),
        )
        self.add_submodule(
            self.node_type + "/node_future_encoder/initial_c",
            model_if_absent=nn.Linear(
                self.state_length, self.hyperparams["enc_rnn_dim_future"]
            ),
        )

        ############################
        #   Robot Future Encoder   #
        ############################
        # We'll create this here, but then later check if we're next to the robot.
        # Based on that, we'll factor this into the computation graph (or not).
        if self.hyperparams["incl_robot_node"]:
            self.add_submodule(
                "robot_future_encoder",
                model_if_absent=nn.LSTM(
                    input_size=self.robot_state_length,
                    hidden_size=self.hyperparams["enc_rnn_dim_future"],
                    bidirectional=True,
                    batch_first=True,
                ),
            )
            # These are related to how you initialize states for the robot future encoder.
            self.add_submodule(
                "robot_future_encoder/initial_h",
                model_if_absent=nn.Linear(
                    self.robot_state_length, self.hyperparams["enc_rnn_dim_future"]
                ),
            )
            self.add_submodule(
                "robot_future_encoder/initial_c",
                model_if_absent=nn.Linear(
                    self.robot_state_length, self.hyperparams["enc_rnn_dim_future"]
                ),
            )

        if self.hyperparams["edge_encoding"]:
            ##############################
            #   Edge Influence Encoder   #
            ##############################
            # NOTE: The edge influence encoding happens during calls
            # to forward or incremental_forward, so we don't create
            # a model for it here for the max and sum variants.
            if self.hyperparams["edge_influence_combine_method"] == "bi-rnn":
                self.add_submodule(
                    self.node_type + "/edge_influence_encoder",
                    model_if_absent=nn.LSTM(
                        input_size=self.hyperparams["enc_rnn_dim_edge"],
                        hidden_size=self.hyperparams["enc_rnn_dim_edge_influence"],
                        bidirectional=True,
                        batch_first=True,
                    ),
                )

                # Four times because we're trying to mimic a bi-directional
                # LSTM's output (which, here, is c and h from both ends).
                self.eie_output_dims = (
                    4 * self.hyperparams["enc_rnn_dim_edge_influence"]
                )

            elif self.hyperparams["edge_influence_combine_method"] == "attention":
                # Chose additive attention because of https://arxiv.org/pdf/1703.03906.pdf
                # We calculate an attention context vector using the encoded edges as the "encoder"
                # (that we attend _over_) and the node history encoder representation
                # as the "decoder state" (that we attend _on_).
                self.add_submodule(
                    self.node_type + "/edge_influence_encoder",
                    model_if_absent=nn.MultiheadAttention(
                        embed_dim=self.hyperparams["enc_rnn_dim_history"],
                        num_heads=1,
                        kdim=self.hyperparams["enc_rnn_dim_edge"],
                        vdim=self.hyperparams["enc_rnn_dim_edge"],
                        batch_first=self.hyperparams["adaptive"],
                    ),
                )

                self.eie_output_dims = self.hyperparams["enc_rnn_dim_edge_influence"]

        ###################
        #   Map Encoder   #
        ###################
        if self.hyperparams["map_encoding"]:
            if self.node_type in self.hyperparams["map_encoder"]:
                me_params = self.hyperparams["map_encoder"][self.node_type]
                self.add_submodule(
                    self.node_type + "/map_encoder",
                    model_if_absent=CNNMapEncoder(
                        me_params["map_channels"],
                        me_params["hidden_channels"],
                        me_params["output_size"],
                        me_params["masks"],
                        me_params["strides"],
                        me_params["patch_size"],
                    ),
                )

        ################################
        #   Discrete Latent Variable   #
        ################################
        self.latent = DiscreteLatent(self.hyperparams, self.device)

        ######################################################################
        #   Various Fully-Connected Layers from Encoder to Latent Variable   #
        ######################################################################
        # Node History Encoder
        x_size = self.hyperparams["enc_rnn_dim_history"]
        if self.hyperparams["edge_encoding"]:
            #              Edge Encoder
            x_size += self.eie_output_dims
        if self.hyperparams["incl_robot_node"]:
            #              Future Conditional Encoder
            x_size += 4 * self.hyperparams["enc_rnn_dim_future"]
        if (
            self.hyperparams["map_encoding"]
            and self.node_type in self.hyperparams["map_encoder"]
        ):
            #              Map Encoder
            x_size += self.hyperparams["map_encoder"][self.node_type]["output_size"]

        z_size = self.hyperparams["N"] * self.hyperparams["K"]

        if self.hyperparams["p_z_x_MLP_dims"] != 0:
            self.add_submodule(
                self.node_type + "/p_z_x",
                model_if_absent=nn.Linear(x_size, self.hyperparams["p_z_x_MLP_dims"]),
            )
            hx_size = self.hyperparams["p_z_x_MLP_dims"]
        else:
            hx_size = x_size

        self.add_submodule(
            self.node_type + "/hx_to_z",
            model_if_absent=nn.Linear(hx_size, self.latent.z_dim),
        )

        if self.hyperparams["q_z_xy_MLP_dims"] != 0:
            self.add_submodule(
                self.node_type + "/q_z_xy",
                #                                           Node Future Encoder
                model_if_absent=nn.Linear(
                    x_size + 4 * self.hyperparams["enc_rnn_dim_future"],
                    self.hyperparams["q_z_xy_MLP_dims"],
                ),
            )
            hxy_size = self.hyperparams["q_z_xy_MLP_dims"]
        else:
            #                           Node Future Encoder
            hxy_size = x_size + 4 * self.hyperparams["enc_rnn_dim_future"]

        self.add_submodule(
            self.node_type + "/hxy_to_z",
            model_if_absent=nn.Linear(hxy_size, self.latent.z_dim),
        )

        ###################
        #   Decoder GRU   #
        ###################
        if self.hyperparams["incl_robot_node"]:
            decoder_input_dims = (
                self.pred_state_length + self.robot_state_length + z_size + x_size
            )
        else:
            decoder_input_dims = self.pred_state_length + z_size + x_size

        self.add_submodule(
            self.node_type + "/decoder/state_action",
            model_if_absent=nn.Linear(
                z_size + x_size if self.hyperparams["adaptive"] else self.state_length,
                self.pred_state_length,
            ),
        )

        self.add_submodule(
            self.node_type + "/decoder/rnn",
            model_if_absent=nn.GRU(
                z_size + x_size, self.hyperparams["dec_rnn_dim"], batch_first=True
            ),
        )
        self.add_submodule(
            self.node_type + "/decoder/rnn_cell",
            model_if_absent=nn.GRUCell(
                decoder_input_dims, self.hyperparams["dec_rnn_dim"]
            ),
        )
        self.add_submodule(
            self.node_type + "/decoder/initial_h",
            model_if_absent=nn.Linear(z_size + x_size, self.hyperparams["dec_rnn_dim"]),
        )

        if self.hyperparams["adaptive"]:
            # We're treating the output dims separately as in Section 3.B
            self.add_submodule(
                self.node_type + "/decoder/post_rnn_x",
                model_if_absent=nn.Linear(
                    self.hyperparams["dec_rnn_dim"], self.hyperparams["dec_final_dim"]
                ),
            )
            self.add_submodule(
                self.node_type + "/decoder/post_rnn_y",
                model_if_absent=nn.Linear(
                    self.hyperparams["dec_rnn_dim"], self.hyperparams["dec_final_dim"]
                ),
            )

            if self.hyperparams["single_mode_multi_sample"]:
                self.add_submodule(
                    self.node_type + "/decoder/post_rnn_Sigma_t",
                    model_if_absent=nn.Linear(
                        self.hyperparams["dec_rnn_dim"], self.pred_state_length
                    ),
                )

        # This is here to keep the number of parameters the same between the adaptive version and
        # base model.
        one_layer_equivalent = 2 * self.hyperparams["dec_final_dim"]
        self.add_submodule(
            self.node_type + "/decoder/post_rnn",
            model_if_absent=nn.Linear(
                self.hyperparams["dec_rnn_dim"], one_layer_equivalent
            ),
        )

        if not self.hyperparams["adaptive"]:
            ###################
            #   Decoder GMM   #
            ###################
            self.GMM_mus_dim = self.pred_state_length
            self.GMM_log_sigmas_dim = self.pred_state_length
            self.GMM_corrs_dim = 1
            GMM_dims = self.GMM_mus_dim + self.GMM_log_sigmas_dim + self.GMM_corrs_dim

            self.add_submodule(
                self.node_type + "/decoder/proj_to_GMM_log_pis",
                model_if_absent=nn.Linear(
                    one_layer_equivalent, self.hyperparams["GMM_components"]
                ),
            )
            self.add_submodule(
                self.node_type + "/decoder/proj_to_GMM_mus",
                model_if_absent=nn.Linear(
                    one_layer_equivalent,
                    self.hyperparams["GMM_components"] * self.pred_state_length,
                ),
            )
            self.add_submodule(
                self.node_type + "/decoder/proj_to_GMM_log_sigmas",
                model_if_absent=nn.Linear(
                    one_layer_equivalent,
                    self.hyperparams["GMM_components"] * self.pred_state_length,
                ),
            )
            self.add_submodule(
                self.node_type + "/decoder/proj_to_GMM_corrs",
                model_if_absent=nn.Linear(
                    one_layer_equivalent, self.hyperparams["GMM_components"]
                ),
            )

        ###################
        #   Decoder BLR   #
        ###################
        if self.hyperparams["adaptive"]:
            blr_layer = BayesianLastLayer(
                self.pred_state_length,
                self.hyperparams["dec_final_dim"],
                self.hyperparams["N"] * self.hyperparams["K"],
                self.hyperparams["sigma_eps_init"],
                self.hyperparams["alpha_init"],
                self.hyperparams["fixed_sigma"],
                self.hyperparams["fixed_alpha"],
                self.hyperparams.get("S0_diag_add", 1e-5),
                self.device,
            )

            if not self.hyperparams["fixed_sigma"]:
                P.register_parametrization(
                    blr_layer, "Sigma_eps", matrix_utils.Positive()
                )

            if not self.hyperparams["fixed_alpha"]:
                P.register_parametrization(blr_layer, "alpha", matrix_utils.Positive())

            self.add_submodule(
                self.node_type + "/decoder/last_layer", model_if_absent=blr_layer
            )

        self.x_size = x_size
        self.z_size = z_size

    def create_edge_models(self, edge_types):
        for edge_type in edge_types:
            neighbor_state_length = int(
                np.sum(
                    [
                        len(entity_dims)
                        for entity_dims in self.state[edge_type.split("->")[0]].values()
                    ]
                )
            )
            if self.hyperparams["edge_state_combine_method"] == "pointnet":
                self.add_submodule(
                    edge_type + "/pointnet_encoder",
                    model_if_absent=nn.Sequential(
                        nn.Linear(self.state_length, 2 * self.state_length),
                        nn.ReLU(),
                        nn.Linear(2 * self.state_length, 2 * self.state_length),
                        nn.ReLU(),
                    ),
                )

                edge_encoder_input_size = 2 * self.state_length + self.state_length

            elif self.hyperparams["edge_state_combine_method"] == "attention":
                self.add_submodule(
                    self.node_type + "/edge_attention_combine",
                    model_if_absent=TemporallyBatchedAdditiveAttention(
                        encoder_hidden_state_dim=self.state_length,
                        decoder_hidden_state_dim=self.state_length,
                    ),
                )
                edge_encoder_input_size = self.state_length + neighbor_state_length

            else:
                edge_encoder_input_size = self.state_length + neighbor_state_length

            self.add_submodule(
                edge_type + "/edge_encoder",
                model_if_absent=nn.LSTM(
                    input_size=edge_encoder_input_size,
                    hidden_size=self.hyperparams["enc_rnn_dim_edge"],
                    batch_first=True,
                ),
            )

    def create_graphical_model(self, edge_types):
        """
        Creates or queries all trainable components.

        :param edge_types: List containing strings for all possible edge types for the node type.
        :return: None
        """
        self.clear_submodules()

        ############################
        #   Everything but Edges   #
        ############################
        self.create_node_models()

        #####################
        #   Edge Encoders   #
        #####################
        if self.hyperparams["edge_encoding"]:
            self.create_edge_models(edge_types)

        for name, module in self.node_modules.items():
            module.to(self.device)

    def create_new_scheduler(
        self, name, annealer, annealer_kws, creation_condition=True
    ):
        value_scheduler = None
        rsetattr(self, name + "_scheduler", value_scheduler)
        if creation_condition:
            annealer_kws["device"] = self.device
            value_annealer = annealer(annealer_kws)
            rsetattr(self, name + "_annealer", value_annealer)

            # This is the value that we'll update on each call of
            # step_annealers().
            rsetattr(self, name, value_annealer(0).clone().detach())
            dummy_optimizer = optim.Optimizer(
                [rgetattr(self, name)], {"lr": value_annealer(0).clone().detach()}
            )
            rsetattr(self, name + "_optimizer", dummy_optimizer)

            value_scheduler = CustomLR(dummy_optimizer, value_annealer)
            rsetattr(self, name + "_scheduler", value_scheduler)

        self.schedulers.append(value_scheduler)
        self.annealed_vars.append(name)

    def set_annealing_params(self):
        self.schedulers = list()
        self.annealed_vars = list()

        self.create_new_scheduler(
            name="kl_weight",
            annealer=sigmoid_anneal,
            annealer_kws={
                "start": self.hyperparams["kl_weight_start"],
                "finish": self.hyperparams["kl_weight"],
                "center_step": self.hyperparams["kl_crossover"],
                "steps_lo_to_hi": self.hyperparams["kl_crossover"]
                / self.hyperparams["kl_sigmoid_divisor"],
            },
        )

        self.create_new_scheduler(
            name="latent.temp",
            annealer=exp_anneal,
            annealer_kws={
                "start": self.hyperparams["tau_init"],
                "finish": self.hyperparams["tau_final"],
                "rate": self.hyperparams["tau_decay_rate"],
            },
        )

        self.create_new_scheduler(
            name="latent.z_logit_clip",
            annealer=sigmoid_anneal,
            annealer_kws={
                "start": self.hyperparams["z_logit_clip_start"],
                "finish": self.hyperparams["z_logit_clip_final"],
                "center_step": self.hyperparams["z_logit_clip_crossover"],
                "steps_lo_to_hi": self.hyperparams["z_logit_clip_crossover"]
                / self.hyperparams["z_logit_clip_divisor"],
            },
            creation_condition=self.hyperparams["use_z_logit_clipping"],
        )

    def step_annealers(self):
        # This should manage all of the step-wise changed
        # parameters automatically.
        for idx, annealed_var in enumerate(self.annealed_vars):
            if rgetattr(self, annealed_var + "_scheduler") is not None:
                # First we step the scheduler.
                with warnings.catch_warnings():  # We use a dummy optimizer: Warning because no .step() was called on it
                    warnings.simplefilter("ignore")
                    rgetattr(self, annealed_var + "_scheduler").step()

                # Then we set the annealed vars' value.
                rsetattr(
                    self,
                    annealed_var,
                    rgetattr(self, annealed_var + "_optimizer").param_groups[0]["lr"],
                )

        if self.hyperparams.get("log_annealers", False):
            self.summarize_annealers()

    def summarize_annealers(self):
        if self.log_writer is not None:
            for annealed_var in self.annealed_vars:
                if rgetattr(self, annealed_var) is not None:
                    self.log_writer.log(
                        {
                            f"{str(self.node_type)}/{annealed_var.replace('.', '/')}": rgetattr(
                                self, annealed_var
                            )
                        },
                        step=self.curr_iter,
                        commit=False,
                    )

    def obtain_encoded_tensors(
        self, mode: ModeKeys, batch: AgentBatch
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Encodes input and output tensors for node and robot.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :return: tuple(x, x_nr_t, y_e, y_r, y, n_s_t0)
            WHERE
            - x: Encoded input / condition tensor to the CVAE x_e.
            - x_r_t: Robot state (if robot is in scene).
            - y_e: Encoded label / future of the node.
            - y_r: Encoded future of the robot.
            - y: Label / future of the node.
            - n_s_t0: Standardized current state of the node.
        """

        enc, x_r_t, y_e, y_r, y = None, None, None, None, None
        initial_dynamics = dict()

        batch_size = batch.agent_hist.shape[0]

        #########################################
        # Provide basic information to encoders #
        #########################################
        node_history_st = batch.agent_hist
        node_history_st_len = batch.agent_hist_len
        node_present_state_st = node_history_st[
            torch.arange(node_history_st.shape[0]), node_history_st_len - 1
        ]

        initial_dynamics["pos"] = node_present_state_st[:, 0:2]
        initial_dynamics["vel"] = node_present_state_st[:, 2:4]

        self.dynamic.set_initial_condition(initial_dynamics)

        if self.hyperparams["incl_robot_node"]:
            robot = batch.robot_fut
            robot_lens = batch.robot_fut_len
            x_r_t, y_r = robot[:, 0], robot[:, 1:]

        ##################
        # Encode History #
        ##################
        node_history_encoded = self.encode_node_history(
            mode, node_history_st, node_history_st_len
        )

        ##################
        # Encode Present #
        ##################
        node_present = node_present_state_st  # [bs, state_dim]

        ##################
        # Encode Future #
        ##################
        if mode != ModeKeys.PREDICT:
            y = batch.agent_fut[..., :2]
            y_lens = batch.agent_fut_len

        ##############################
        # Encode Node Edges per Type #
        ##############################
        if self.hyperparams["edge_encoding"]:
            if batch.num_neigh.max() == 0:
                total_edge_influence = torch.zeros_like(node_history_encoded)
            else:
                # Encode edges
                encoded_edges = self.encode_edge(
                    mode,
                    node_history_st,
                    node_history_st_len,
                    batch.neigh_hist,
                    batch.neigh_hist_len,
                    batch.neigh_types,
                    batch.num_neigh,
                )
                #####################
                # Encode Node Edges #
                #####################
                total_edge_influence, attn_weights = self.encode_total_edge_influence(
                    mode,
                    encoded_edges,
                    batch.num_neigh,
                    node_history_encoded,
                    node_history_st_len,
                    batch_size,
                )

        ################
        # Map Encoding #
        ################
        if (
            self.hyperparams["map_encoding"]
            and self.node_type in self.hyperparams["map_encoder"]
        ):
            if (
                self.hyperparams["log_maps"]
                and self.log_writer
                and (self.curr_iter + 1) % 500 == 0
            ):
                image = wandb.Image(batch.maps[0], caption=f"Batch Map 0")
                self.log_writer.log(
                    {f"{self.node_type}/maps": image}, step=self.curr_iter, commit=False
                )

            encoded_map = self.node_modules[self.node_type + "/map_encoder"](
                batch.maps * 2.0 - 1.0, (mode == ModeKeys.TRAIN)
            )
            do = self.hyperparams["map_encoder"][self.node_type]["dropout"]
            encoded_map = F.dropout(encoded_map, do, training=(mode == ModeKeys.TRAIN))

        ######################################
        # Concatenate Encoder Outputs into x #
        ######################################
        enc_concat_list = list()

        # Every node has an edge-influence encoder (which could just be zero).
        if self.hyperparams["edge_encoding"]:
            enc_concat_list.append(total_edge_influence)  # [bs/nbs, enc_rnn_dim]

        # Every node has a history encoder.
        enc_concat_list.append(node_history_encoded)  # [bs/nbs, enc_rnn_dim_history]

        if self.hyperparams["incl_robot_node"]:
            robot_future_encoder = self.encode_robot_future(
                mode, x_r_t, y_r, robot_lens
            )
            enc_concat_list.append(robot_future_encoder)

        if (
            self.hyperparams["map_encoding"]
            and self.node_type in self.hyperparams["map_encoder"]
        ):
            if self.log_writer:
                self.log_writer.log(
                    {
                        f"{self.node_type}/encoded_map_max": torch.max(
                            torch.abs(encoded_map)
                        ).item()
                    },
                    step=self.curr_iter,
                    commit=False,
                )
            enc_concat_list.append(
                encoded_map.unsqueeze(1).expand((-1, node_history_encoded.shape[1], -1))
                if self.hyperparams["adaptive"]
                else encoded_map
            )

        enc = torch.cat(enc_concat_list, dim=-1)

        if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
            y_e = self.encode_node_future(mode, node_present, y, y_lens)
            if self.hyperparams["adaptive"]:
                y_e = y_e.expand((-1, enc.shape[1], -1))

        return enc, x_r_t, y_e, y_r, y

    def encode_node_history(self, mode, node_hist, node_hist_len):
        """
        Encodes the nodes history.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param node_hist: Historic and current state of the node. [bs, mhl, state]
        :param node_hist_len: Number of timesteps for which data is available [bs]
        :return: Encoded node history tensor. [bs, enc_rnn_dim]
        """
        outputs, _ = run_lstm_on_variable_length_seqs(
            self.node_modules[self.node_type + "/node_history_encoder"],
            seqs=node_hist,
            seq_lens=node_hist_len,
        )

        outputs = F.dropout(
            outputs,
            p=1.0 - self.hyperparams["rnn_kwargs"]["dropout_keep_prob"],
            training=(mode == ModeKeys.TRAIN),
        )  # [bs, max_time, enc_rnn_dim]

        if self.hyperparams["adaptive"]:
            # NaN is the default padding value in trajdata's AgentBatch
            # (which is good for catching bugs, but bad for ML training).
            return torch.nan_to_num(outputs)
        else:
            return outputs[torch.arange(outputs.shape[0]), node_hist_len - 1]

    def encode_edge(
        self,
        mode: ModeKeys,
        node_history_st: torch.Tensor,
        node_history_st_len: torch.Tensor,
        neigh_hist: torch.Tensor,
        neigh_hist_len: torch.Tensor,
        neigh_types: torch.Tensor,
        num_neigh: torch.Tensor,
    ) -> torch.Tensor:

        if neigh_hist.shape[2] < node_history_st.shape[1]:
            neigh_hist = F.pad(
                neigh_hist,
                pad=(0, 0, 0, node_history_st.shape[1] - neigh_hist.shape[2]),
                value=np.nan,
            )
        elif neigh_hist.shape[2] > node_history_st.shape[1]:
            node_history_st = F.pad(
                node_history_st,
                pad=(0, 0, 0, neigh_hist.shape[2] - node_history_st.shape[1]),
                value=np.nan,
            )

        node_hist_lens_for_cat = node_history_st_len.unsqueeze(1).expand(
            (-1, neigh_hist.shape[1])
        )
        joint_history_len = torch.minimum(
            neigh_hist_len, node_hist_lens_for_cat
        ).flatten()
        has_data: torch.Tensor = joint_history_len > 0

        node_hist_for_cat = node_history_st.repeat_interleave(
            neigh_hist.shape[1], dim=0, output_size=has_data.shape[0]
        )[has_data]
        neigh_hist_for_cat = neigh_hist.reshape(-1, *neigh_hist.shape[2:])[has_data]

        joint_history_len = joint_history_len[has_data]
        joint_neigh_types = neigh_types.flatten()[has_data]

        node_shifts = joint_history_len - node_hist_lens_for_cat.flatten()[has_data]
        neigh_shifts = joint_history_len - neigh_hist_len.flatten()[has_data]

        node_hist_for_cat = roll_by_gather(
            node_hist_for_cat, dim=1, shifts=node_shifts.to(node_hist_for_cat.device)
        )
        neigh_hist_for_cat = roll_by_gather(
            neigh_hist_for_cat, dim=1, shifts=neigh_shifts.to(neigh_hist_for_cat.device)
        )

        joint_history = torch.cat([neigh_hist_for_cat, node_hist_for_cat], dim=-1)

        total_neighbors: int = num_neigh.sum().item()

        sorting_indices = torch.empty(
            total_neighbors, dtype=torch.long, device=self.device
        )

        returns: List[torch.Tensor] = list()
        num_already_done = 0
        for neigh_type in joint_neigh_types.unique(sorted=False):
            if neigh_type < 0:
                # AgentType is non-negative (by design, negative values are padding values).
                continue

            matches_type = joint_neigh_types == neigh_type

            num_matching_type = matches_type.sum()
            sorting_indices[matches_type] = torch.arange(
                start=num_already_done,
                end=num_already_done + num_matching_type,
                dtype=sorting_indices.dtype,
                device=self.device,
            )
            joint_history_type = joint_history[matches_type]
            joint_history_type_len = joint_history_len[matches_type]

            outputs, _ = run_lstm_on_variable_length_seqs(
                self.node_modules[
                    DirectedEdge.get_str_from_types(
                        AgentType(neigh_type.item()), self.node_type_obj
                    )
                    + "/edge_encoder"
                ],
                seqs=joint_history_type,
                seq_lens=joint_history_type_len,
            )

            outputs = F.dropout(
                outputs,
                p=1.0 - self.hyperparams["rnn_kwargs"]["dropout_keep_prob"],
                training=(mode == ModeKeys.TRAIN),
            )  # [bs, max_time, enc_rnn_dim]

            ret = outputs[torch.arange(outputs.shape[0]), joint_history_type_len - 1]
            returns.append(ret)
            num_already_done += num_matching_type

        batch_indexed_outputs = torch.concat(returns, dim=0)[sorting_indices]

        if self.hyperparams["dynamic_edges"] == "yes":
            max_hist_len: int = joint_history_len.max().item()
            extended_addition_filter: torch.Tensor = torch.tensor(
                self.hyperparams["edge_addition_filter"],
                dtype=torch.float,
                device=self.device,
            )
            if extended_addition_filter.shape[0] < max_hist_len:
                extended_addition_filter = F.pad(
                    extended_addition_filter,
                    (0, max_hist_len - extended_addition_filter.shape[0]),
                    mode="constant",
                    value=1.0,
                )

            edge_weights: torch.Tensor = extended_addition_filter[joint_history_len - 1]
            batch_indexed_outputs *= edge_weights.unsqueeze(1)

        encoded_edges: Tuple[torch.Tensor, ...] = torch.split(
            batch_indexed_outputs, num_neigh.tolist()
        )
        return nn.utils.rnn.pad_sequence(encoded_edges)

    def encode_total_edge_influence(
        self,
        mode: ModeKeys,
        encoded_edges: torch.Tensor,
        num_neighbors: torch.Tensor,
        node_history_encoder: torch.Tensor,
        node_history_len: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_weights = None

        if self.hyperparams["edge_influence_combine_method"] == "sum":
            stacked_encoded_edges = torch.stack(encoded_edges, dim=0)
            combined_edges = torch.sum(stacked_encoded_edges, dim=0)

        elif self.hyperparams["edge_influence_combine_method"] == "mean":
            stacked_encoded_edges = torch.stack(encoded_edges, dim=0)
            combined_edges = torch.mean(stacked_encoded_edges, dim=0)

        elif self.hyperparams["edge_influence_combine_method"] == "max":
            stacked_encoded_edges = torch.stack(encoded_edges, dim=0)
            combined_edges = torch.max(stacked_encoded_edges, dim=0)

        elif self.hyperparams["edge_influence_combine_method"] == "bi-rnn":
            if len(encoded_edges) == 0:
                combined_edges = torch.zeros(
                    (batch_size, self.eie_output_dims), device=self.device
                )

            else:
                # axis=1 because then we get size [batch_size, max_time, depth]
                encoded_edges = torch.stack(encoded_edges, dim=1)

                _, state = self.node_modules[
                    self.node_type + "/edge_influence_encoder"
                ](encoded_edges)
                combined_edges = unpack_RNN_state(state)
                combined_edges = F.dropout(
                    combined_edges,
                    p=1.0 - self.hyperparams["rnn_kwargs"]["dropout_keep_prob"],
                    training=(mode == ModeKeys.TRAIN),
                )

        elif self.hyperparams["edge_influence_combine_method"] == "attention":
            # Used in Social Attention (https://arxiv.org/abs/1710.04689)
            max_neighbors = encoded_edges.shape[0]
            if self.hyperparams["adaptive"]:
                if len(encoded_edges) == 0:
                    max_hl = node_history_encoder.shape[1]
                    combined_edges = torch.zeros(
                        (batch_size, max_hl, self.eie_output_dims), device=self.device
                    )

                else:
                    with_neighbors = num_neighbors > 0
                    combined_edges = torch.zeros_like(node_history_encoder)

                    # Not attending over parts of the agent's history with no data.
                    len_mask = ~arr_utils.mask_up_to(
                        node_history_len.to(self.device),
                        max_len=node_history_encoder.shape[1],
                    )
                    attn_mask = len_mask.unsqueeze(-1).expand((-1, -1, max_neighbors))

                    # Not attending over padded neighbor elements.
                    key_padding_mask = torch.triu(
                        torch.ones(
                            (max_neighbors + 1, max_neighbors),
                            dtype=torch.bool,
                            device=self.device,
                        ),
                        diagonal=0,
                    )[num_neighbors]

                    # Combining the masks into one, and addressing
                    # https://github.com/pytorch/pytorch/issues/41508
                    # by filling the mask with a big negative number, as in
                    # https://github.com/pytorch/text/blob/master/torchtext/nn/modules/multiheadattention.py#L208
                    attn_mask = attn_mask.logical_or(key_padding_mask.unsqueeze(1))
                    float_attn_mask = torch.zeros_like(
                        attn_mask, dtype=node_history_encoder.dtype
                    )
                    float_attn_mask.masked_fill_(attn_mask, -1e8)

                    query = node_history_encoder[with_neighbors]
                    kv = encoded_edges.transpose(1, 0)[with_neighbors]

                    attn_weights = torch.zeros(
                        (num_neighbors.shape[0], query.shape[1], kv.shape[1]),
                        device=query.device,
                    )
                    (
                        combined_edges[with_neighbors],
                        attn_weights[with_neighbors],
                    ) = self.node_modules[self.node_type + "/edge_influence_encoder"](
                        query=query,
                        key=kv,
                        value=kv,
                        attn_mask=float_attn_mask[with_neighbors],
                    )
                    combined_edges = F.dropout(
                        combined_edges,
                        p=1.0 - self.hyperparams["rnn_kwargs"]["dropout_keep_prob"],
                        training=(mode == ModeKeys.TRAIN),
                    )

                    # Ensuring that the padded parts of the outputs are zero'd out.
                    combined_edges.masked_fill_(len_mask.unsqueeze(-1), 0.0)
                    attn_weights.masked_fill_(attn_mask, 0.0)

            else:
                if len(encoded_edges) == 0:
                    combined_edges = torch.zeros(
                        (batch_size, self.eie_output_dims), device=self.device
                    )

                else:
                    with_neighbors = num_neighbors > 0
                    combined_edges = torch.zeros_like(node_history_encoder).unsqueeze(0)

                    key_padding_mask = torch.triu(
                        torch.ones(
                            (max_neighbors + 1, max_neighbors),
                            dtype=torch.bool,
                            device=self.device,
                        ),
                        diagonal=0,
                    )[num_neighbors]
                    combined_edges[:, with_neighbors], attn_weights = self.node_modules[
                        self.node_type + "/edge_influence_encoder"
                    ](
                        query=node_history_encoder[with_neighbors].unsqueeze(0),
                        key=encoded_edges[:, with_neighbors],
                        value=encoded_edges[:, with_neighbors],
                        key_padding_mask=key_padding_mask[with_neighbors],
                        attn_mask=None,
                    )
                    combined_edges = F.dropout(
                        combined_edges.squeeze(0),
                        p=1.0 - self.hyperparams["rnn_kwargs"]["dropout_keep_prob"],
                        training=(mode == ModeKeys.TRAIN),
                    )

        return combined_edges, attn_weights

    def encode_node_future(
        self, mode, node_present, node_future, future_lens
    ) -> torch.Tensor:
        """
        Encodes the node future (during training) using a bi-directional LSTM

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param node_present: Current state of the node. [bs, state]
        :param node_future: Future states of the node. [bs, ph, state]
        :return: Encoded future.
        """
        initial_h_model = self.node_modules[
            self.node_type + "/node_future_encoder/initial_h"
        ]
        initial_c_model = self.node_modules[
            self.node_type + "/node_future_encoder/initial_c"
        ]

        # Here we're initializing the forward hidden states,
        # but zeroing the backward ones.
        initial_h = initial_h_model(node_present)
        initial_h = torch.stack(
            [initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0
        )

        initial_c = initial_c_model(node_present)
        initial_c = torch.stack(
            [initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0
        )

        initial_state = (initial_h, initial_c)

        node_future_packed = pack_padded_sequence(
            node_future, future_lens, batch_first=True, enforce_sorted=False
        )

        _, state = self.node_modules[self.node_type + "/node_future_encoder"](
            node_future_packed, initial_state
        )
        state = unpack_RNN_state(state)
        state = F.dropout(
            state,
            p=1.0 - self.hyperparams["rnn_kwargs"]["dropout_keep_prob"],
            training=(mode == ModeKeys.TRAIN),
        )

        return state.unsqueeze(1) if self.hyperparams["adaptive"] else state

    def encode_robot_future(
        self, mode, robot_present, robot_future, future_lens
    ) -> torch.Tensor:
        """
        Encodes the robot future (during training) using a bi-directional LSTM

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param robot_present: Current state of the robot. [bs, state]
        :param robot_future: Future states of the robot. [bs, ph, state]
        :return: Encoded future.
        """
        initial_h_model = self.node_modules["robot_future_encoder/initial_h"]
        initial_c_model = self.node_modules["robot_future_encoder/initial_c"]

        # Here we're initializing the forward hidden states,
        # but zeroing the backward ones.
        initial_h = initial_h_model(robot_present)
        initial_h = torch.stack(
            [initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0
        )

        initial_c = initial_c_model(robot_present)
        initial_c = torch.stack(
            [initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0
        )

        initial_state = (initial_h, initial_c)

        robot_future_packed = pack_padded_sequence(
            robot_future, future_lens, batch_first=True, enforce_sorted=False
        )

        _, state = self.node_modules["robot_future_encoder"](
            robot_future_packed, initial_state
        )
        state = unpack_RNN_state(state)
        state = F.dropout(
            state,
            p=1.0 - self.hyperparams["rnn_kwargs"]["dropout_keep_prob"],
            training=(mode == ModeKeys.TRAIN),
        )

        return state

    def q_z_xy(self, mode, enc, y_e) -> torch.Tensor:
        r"""
        .. math:: q_\phi(z \mid \mathbf{x}_i, \mathbf{y}_i)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y_e: Encoded future tensor.
        :return: Latent distribution of the CVAE.
        """
        xy = torch.cat([enc, y_e], dim=-1)

        if self.hyperparams["q_z_xy_MLP_dims"] != 0:
            dense = self.node_modules[self.node_type + "/q_z_xy"]
            h = F.dropout(
                F.relu(dense(xy)),
                p=1.0 - self.hyperparams["MLP_dropout_keep_prob"],
                training=(mode == ModeKeys.TRAIN),
            )

        else:
            h = xy

        to_latent = self.node_modules[self.node_type + "/hxy_to_z"]
        return self.latent.dist_from_h(to_latent(h), mode)

    def p_z_x(self, mode, enc):
        r"""
        .. math:: p_\theta(z \mid \mathbf{x}_i)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :return: Latent distribution of the CVAE.
        """
        if self.hyperparams["p_z_x_MLP_dims"] != 0:
            dense = self.node_modules[self.node_type + "/p_z_x"]
            h = F.dropout(
                F.relu(dense(enc)),
                p=1.0 - self.hyperparams["MLP_dropout_keep_prob"],
                training=(mode == ModeKeys.TRAIN),
            )

        else:
            h = enc

        to_latent = self.node_modules[self.node_type + "/hx_to_z"]
        return self.latent.dist_from_h(to_latent(h), mode)

    def project_to_GMM_params(
        self, tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Projects tensor to parameters of a GMM with N components and D dimensions.

        :param tensor: Input tensor.
        :return: tuple(log_pis, mus, log_sigmas, corrs)
            WHERE
            - log_pis: Weight (logarithm) of each GMM component. [N]
            - mus: Mean of each GMM component. [N, D]
            - log_sigmas: Standard Deviation (logarithm) of each GMM component. [N, D]
            - corrs: Correlation between the GMM components. [N]
        """
        log_pis = self.node_modules[self.node_type + "/decoder/proj_to_GMM_log_pis"](
            tensor
        )
        mus = self.node_modules[self.node_type + "/decoder/proj_to_GMM_mus"](tensor)
        log_sigmas = self.node_modules[
            self.node_type + "/decoder/proj_to_GMM_log_sigmas"
        ](tensor)
        corrs = torch.tanh(
            self.node_modules[self.node_type + "/decoder/proj_to_GMM_corrs"](tensor)
        )
        return log_pis, mus, log_sigmas, corrs

    def p_y_xz_old(
        self,
        mode,
        x,
        x_nr_t,
        y_r,
        n_s_t0,
        pos_hist_len,
        z_stacked,
        dt,
        prediction_horizon,
        num_samples,
        num_components=1,
        z_mode=False,
        gmm_mode=False,
        update_mode: UpdateMode = UpdateMode.NO_UPDATE,
    ):
        r"""
        .. math:: p_\psi(\mathbf{y}_i \mid \mathbf{x}_i, z)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param x_nr_t: Joint state of node and robot (if robot is in scene).
        :param y_r: Encoded future robot tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z_stacked: Stacked latent state. [num_samples_z * num_samples_gmm, bs, latent_state]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param num_components: Number of GMM components.
        :param z_mode: If True: The most likely z latent value is being used.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :return: GMM2D. If mode is Predict, also samples from the GMM.
        """
        ph = prediction_horizon
        pred_dim = self.pred_state_length

        z = torch.reshape(z_stacked, (-1, self.latent.z_dim))
        zx = torch.cat([z, x.repeat(num_samples * num_components, 1)], dim=1)

        cell = self.node_modules[self.node_type + "/decoder/rnn_cell"]
        initial_h_model = self.node_modules[self.node_type + "/decoder/initial_h"]
        post_cell: nn.Module = self.node_modules[self.node_type + "/decoder/post_rnn"]

        initial_state = initial_h_model(zx)

        log_pis, mus, log_sigmas, corrs, a_sample = [], [], [], [], []

        # Infer initial action state for node from current state
        a_0 = self.node_modules[self.node_type + "/decoder/state_action"](n_s_t0)

        state = initial_state
        if self.hyperparams["incl_robot_node"]:
            input_ = torch.cat(
                [
                    zx,
                    a_0.repeat(num_samples * num_components, 1),
                    x_nr_t.repeat(num_samples * num_components, 1),
                ],
                dim=1,
            )
        else:
            input_ = torch.cat([zx, a_0.repeat(num_samples * num_components, 1)], dim=1)

        for j in range(ph):
            h_state = cell(input_, state)
            decoder_out = F.relu(post_cell(h_state))
            log_pi_t, mu_t, log_sigma_t, corr_t = self.project_to_GMM_params(
                decoder_out
            )

            gmm = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t)  # [k;bs, pred_dim]

            if mode == ModeKeys.PREDICT and gmm_mode:
                a_t = gmm.mode()
            else:
                a_t = gmm.rsample()

            if num_components > 1:
                if mode == ModeKeys.PREDICT:
                    log_pis.append(self.latent.p_dist.logits.repeat(num_samples, 1, 1))
                else:
                    log_pis.append(self.latent.q_dist.logits.repeat(num_samples, 1, 1))
            else:
                log_pis.append(
                    torch.ones_like(
                        corr_t.reshape(num_samples, num_components, -1)
                        .permute(0, 2, 1)
                        .reshape(-1, 1)
                    )
                )

            mus.append(
                mu_t.reshape(num_samples, num_components, -1, 2)
                .permute(0, 2, 1, 3)
                .reshape(-1, 2 * num_components)
            )
            log_sigmas.append(
                log_sigma_t.reshape(num_samples, num_components, -1, 2)
                .permute(0, 2, 1, 3)
                .reshape(-1, 2 * num_components)
            )
            corrs.append(
                corr_t.reshape(num_samples, num_components, -1)
                .permute(0, 2, 1)
                .reshape(-1, num_components)
            )

            if self.hyperparams["incl_robot_node"]:
                dec_inputs = [
                    zx,
                    a_t,
                    y_r[:, j].repeat(num_samples * num_components, 1),
                ]
            else:
                dec_inputs = [zx, a_t]
            input_ = torch.cat(dec_inputs, dim=1)
            state = h_state

        log_pis = torch.stack(log_pis, dim=1)
        mus = torch.stack(mus, dim=1)
        log_sigmas = torch.stack(log_sigmas, dim=1)
        corrs = torch.stack(corrs, dim=1)

        a_dist = GMM2D(
            torch.reshape(log_pis, [num_samples, -1, ph, num_components]),
            torch.reshape(mus, [num_samples, -1, ph, num_components * pred_dim]),
            torch.reshape(log_sigmas, [num_samples, -1, ph, num_components * pred_dim]),
            torch.reshape(corrs, [num_samples, -1, ph, num_components]),
        )

        if self.hyperparams["dynamic"][self.node_type]["distribution"]:
            y_dist = self.dynamic.integrate_distribution(a_dist, x, dt)
        else:
            y_dist = a_dist

        if mode == ModeKeys.PREDICT:
            if gmm_mode:
                a_sample = a_dist.mode()
            else:
                a_sample = a_dist.rsample()
            sampled_future = self.dynamic.integrate_samples(a_sample, x, dt)
            return y_dist, sampled_future
        else:
            return y_dist

    def p_y_xz_adaptive(
        self,
        mode: ModeKeys,
        x_enc: torch.Tensor,
        x_nr_t: torch.Tensor,
        y_r: torch.Tensor,
        pos_hist: torch.Tensor,
        pos_hist_len: torch.Tensor,
        z_stacked: torch.Tensor,
        dt: torch.Tensor,
        prediction_horizon: int,
        num_samples: int,
        num_components: int = 1,
        z_mode: bool = False,
        gmm_mode: bool = False,
        update_mode: UpdateMode = UpdateMode.BATCH_FROM_PRIOR,
    ):
        r"""
        .. math:: p_\psi(\mathbf{y}_i \mid \mathbf{x}_i, z)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param x_nr_t: Joint state of node and robot (if robot is in scene).
        :param y_r: Encoded future robot tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z_stacked: Stacked latent state. [num_samples_z * num_samples_gmm, bs, latent_state]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param num_components: Number of GMM components.
        :param z_mode: If True: The most likely z latent value is being used.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :return: GMM2D. If mode is Predict, also samples from the GMM.
        """
        batch_size, hist_size, _ = x_enc.shape

        if self.hyperparams["K"] > 1 and self.hyperparams["single_mode_multi_sample"]:
            raise ValueError("--single_mode_multi_sample can only be used with --K=1")

        decoder_cell: nn.Module = self.node_modules[
            self.node_type + "/decoder/rnn_cell"
        ]
        decoder_h_init: nn.Module = self.node_modules[
            self.node_type + "/decoder/initial_h"
        ]
        post_cell_x: nn.Module = self.node_modules[
            self.node_type + "/decoder/post_rnn_x"
        ]
        post_cell_y: nn.Module = self.node_modules[
            self.node_type + "/decoder/post_rnn_y"
        ]
        if self.hyperparams["single_mode_multi_sample"]:
            post_cell_Sigma_t: nn.Module = self.node_modules[
                self.node_type + "/decoder/post_rnn_Sigma_t"
            ]
        blr_layer: BayesianLastLayer = self.node_modules[
            self.node_type + "/decoder/last_layer"
        ]

        log_pis: torch.Tensor
        if num_components > 1:
            if mode == ModeKeys.PREDICT:
                log_pis = self.latent.p_dist.logits.repeat(num_samples, 1, 1)
            else:
                log_pis = self.latent.q_dist.logits.repeat(num_samples, 1, 1)
        else:
            log_pis = torch.zeros(
                (num_samples * batch_size, hist_size, num_components),
                device=self.device,
            )

        # Rolling the time dimension so that t_0 is at index -1
        # and prior timesteps are at -2, 3, and so on.
        on_device_pos_hist_len = pos_hist_len.to(x_enc.device)
        x_enc = torch.nan_to_num(
            roll_by_gather(x_enc, 1, hist_size - on_device_pos_hist_len)
        )
        pos_hist = torch.nan_to_num(
            roll_by_gather(pos_hist, 1, hist_size - on_device_pos_hist_len)
        )
        log_pis = roll_by_gather(log_pis, 1, hist_size - on_device_pos_hist_len)

        # 1. Condition our Bayesian last layer using decoded one-step predictions of historical data.
        if z_mode:
            # Going from [S*1, B, H, K (one-hot)] to [S*1*B, H, K (one-hot)]
            z = torch.reshape(z_stacked, (-1, hist_size, self.latent.z_dim))
        else:
            # Going from [S*K, B, K (one-hot)] to [S*K*B, H, K (one-hot)]
            z = torch.reshape(z_stacked, (-1, 1, self.latent.z_dim)).expand(
                (-1, hist_size, -1)
            )

        zx = torch.cat([z, x_enc.repeat(num_samples * num_components, 1, 1)], dim=-1)

        # Going from (S*K*B, H, D) -> (S*K*B*H, D)
        zx = zx.reshape(-1, zx.shape[-1])

        p_0: torch.Tensor = pos_hist.repeat(num_samples * num_components, 1, 1).reshape(
            (-1, 2)
        )

        if self.hyperparams["incl_robot_node"]:
            cell_input = torch.cat(
                [zx, p_0, x_nr_t.repeat(num_samples * num_components, 1)], dim=-1
            )
        else:
            cell_input = torch.cat(
                [
                    zx,
                    p_0,
                ],
                dim=-1,
            )

        init_state = decoder_h_init(zx)

        # Get one-step predictions from the decoder.
        hidden_state = decoder_cell(cell_input, init_state)
        decoder_out_x = F.relu(post_cell_x(hidden_state))
        decoder_out_y = F.relu(post_cell_y(hidden_state))
        decoder_out = torch.stack(
            (decoder_out_x, decoder_out_y), dim=-2
        )  # [S*K*B*H, N, D]

        Sigma_t: Optional[torch.Tensor] = None
        if self.hyperparams["single_mode_multi_sample"]:
            Sigma_t = torch.exp(post_cell_Sigma_t(hidden_state))  # [S*K*B*H, N]
            Sigma_t = Sigma_t.reshape(
                num_samples,
                num_components,
                batch_size,
                hist_size,
                self.pred_state_length,
            ).transpose(
                1, 2
            )  # [S, B, K, H, N (pred_state_dim=2)]

        # Going back to [S, K, B, H, N (pred_state_dim=2), D],
        # and masking out timesteps without data.
        Phi_0_t = decoder_out.reshape(
            num_samples, num_components, batch_size, hist_size, 2, -1
        ).transpose(
            1, 2
        )  # [S, B, K, H, N (pred_state_dim=2), D]

        log_pis = log_pis.reshape(num_samples, batch_size, hist_size, num_components)

        len_mask = arr_utils.mask_up_to(
            hist_size - on_device_pos_hist_len, max_len=pos_hist.shape[1]
        )

        Phi_0_t = torch.masked_fill(Phi_0_t, len_mask[None, :, None, :, None, None], 0)

        Y_data = torch.diff(pos_hist, dim=-2)
        Y_data[len_mask[..., :-1]] = 0

        if update_mode == UpdateMode.ITERATIVE and blr_layer.num_updates > 0:
            # The idea here is to do the full iteration below for data up to
            # the current timestep, but then only the newly seen data after that
            # (hence the blr_layer.num_updates check).
            Phi_0_t = Phi_0_t[..., -2:, :, :]
            log_pis = log_pis[..., -2:, :]
            Y_data = Y_data[..., [-1], :]

            if self.hyperparams["single_mode_multi_sample"]:
                Sigma_t = Sigma_t[..., -2:, :]

        if self.hyperparams["only_k0"]:
            posterior_K_dist, Sigma_pred = blr_layer.get_prior(Phi_0_t[..., -1, :, :])
        else:
            # Update last layer K.
            if update_mode == UpdateMode.ITERATIVE:
                # Looping from 0 to current timestep - 1
                for t in range(Phi_0_t.shape[-3] - 1):
                    blr_layer.incorporate_transition(
                        Phi_0_t[..., t, :, :],
                        torch.exp(log_pis[..., t, :]),
                        Y_data[..., t, :],
                        (on_device_pos_hist_len >= 2 - t),
                        Sigma_t[..., t, :] if Sigma_t is not None else None,
                    )
            elif update_mode in {UpdateMode.BATCH_FROM_PRIOR, UpdateMode.ONLINE_BATCH}:
                blr_layer.incorporate_batch(
                    Phi_0_t[..., :-1, :, :],
                    torch.exp(log_pis[..., :-1, :]),
                    Y_data.unsqueeze(0).unsqueeze(2),
                    on_device_pos_hist_len
                    - 1,  # -1 because excluding the current timestep
                    update_mode,
                    Sigma_t[..., :-1, :] if Sigma_t is not None else None,
                )

            posterior_K_dist, Sigma_pred = blr_layer.get_posterior(
                Phi_0_t[..., -1, :, :], self.hyperparams["single_mode_multi_sample"]
            )

        # 2. Obtain multistep predictions with posterior K.
        z_mode_idxs: Optional[torch.Tensor] = None
        if z_mode:
            # Going from [S*1, B, H, K (one-hot)] to [S*1*B, K (one-hot)]
            z = torch.reshape(z_stacked[..., -1, :], (-1, self.latent.z_dim))
            z_mode_idxs = torch.argmax(self.latent.p_dist.probs[..., -1, :], dim=-1)
        else:
            # Going from [S*K, B, K (one-hot)] to [S*K*B, K (one-hot)]
            z = torch.reshape(z_stacked, (-1, self.latent.z_dim))

        zx = torch.cat(
            [z, x_enc[..., -1, :].repeat(num_samples * num_components, 1)], dim=-1
        )

        p_0: torch.Tensor = (
            pos_hist[..., -1, :]
            .repeat(num_samples * num_components, 1)
            .reshape((-1, 2))
        )

        if self.hyperparams["incl_robot_node"]:
            cell_input = torch.cat(
                [zx, p_0, x_nr_t.repeat(num_samples * num_components, 1)], dim=-1
            )
        else:
            cell_input = torch.cat(
                [
                    zx,
                    p_0,
                ],
                dim=-1,
            )

        num_K_samples: int = 1
        if self.hyperparams["single_mode_multi_sample"]:
            if mode == ModeKeys.PREDICT:
                if gmm_mode:
                    num_K_samples: int = 1
                else:
                    num_K_samples: int = 25  # Matching the base T++.
            else:
                num_K_samples: int = self.hyperparams["single_mode_multi_sample_num"]

            if gmm_mode:
                sampled_K: torch.Tensor = posterior_K_dist.mean.unsqueeze(0)
            else:
                sampled_K: torch.Tensor = posterior_K_dist.rsample(
                    (num_K_samples,)
                )  # [X, S, B, K, N, D]

            sampled_K = sampled_K.reshape(
                (-1, *sampled_K.shape[4:], 1)
            )  # [XSBK, N, D, 1]

            zx = zx.expand(num_K_samples, *zx.shape).reshape(-1, *zx.shape[1:])
            cell_input = cell_input.expand(num_K_samples, *cell_input.shape).reshape(
                -1, *cell_input.shape[1:]
            )

        else:
            sampled_K: torch.Tensor = posterior_K_dist.mean  # [S, B, K, N, D]
            if z_mode_idxs is not None:
                # If we are sampling specific z_modes per batch,
                # then select them here.
                sampled_K = sampled_K[:, torch.arange(batch_size), z_mode_idxs]
                Sigma_pred = Sigma_pred[
                    :, torch.arange(batch_size), z_mode_idxs
                ].unsqueeze(2)
            sampled_K = sampled_K.reshape(
                (-1, *sampled_K.shape[-2:], 1)
            )  # [SBK, N, D, 1]

            # Making Sigma_pred go from [S, B, K, 2, 2] to [S*K*B, 1, 2, 2]
            Sigma_pred = Sigma_pred.transpose(1, 2).reshape(
                (-1, 1, *Sigma_pred.shape[-2:])
            )
            Sigma_eps = (
                torch.diag_embed(blr_layer.Sigma_eps)  # [N] -> [N, N]
                .expand(
                    # [N, N] -> [S, B, K, N, N]
                    num_samples,
                    batch_size,
                    num_components,
                    -1,
                    -1,
                )
                .transpose(1, 2)
                .reshape(
                    # [S, K, B, N, N] ->  [S*K*B, 1, N, N]
                    (-1, 1, Y_data.shape[-1], Y_data.shape[-1])
                )
            )  # [S*K*B, 1, N, N]

        hidden_state: torch.Tensor = decoder_h_init(zx)

        log_pis = log_pis[..., -1, :].reshape((-1, 1))
        if self.hyperparams["single_mode_multi_sample"]:
            log_pis = log_pis.expand(num_K_samples, -1, -1).reshape(-1, 1)

        mus_list, log_sigmas_list, corrs_list = [], [], []
        prev_y_dist: Optional[GMM2D] = None
        for j in range(prediction_horizon):
            hidden_state: torch.Tensor = decoder_cell(cell_input, hidden_state)
            decoder_out_x = F.relu(post_cell_x(hidden_state))
            decoder_out_y = F.relu(post_cell_y(hidden_state))
            decoder_out = torch.stack(
                (decoder_out_x, decoder_out_y), dim=-2
            )  # [X*S*K*B, N, D]

            if self.hyperparams["single_mode_multi_sample"]:
                Sigma_t_diag = torch.exp(post_cell_Sigma_t(hidden_state))
                Sigma_t = torch.diag_embed(Sigma_t_diag).unsqueeze(
                    -3
                )  # [X*S*K*B, 1, N, N]

                mus = (
                    (decoder_out.unsqueeze(-2) @ sampled_K)
                    .squeeze(-1)
                    .squeeze(-1)
                    .reshape(-1, self.pred_state_length)
                )  # [X*S*K*B, N]
            else:
                mus = (
                    (decoder_out.unsqueeze(-2) @ sampled_K).squeeze(-1).squeeze(-1)
                )  # [S*K*B, N]

            # Reshaping below to make room for sample dimension as well as prediction horizon dimension.
            # Since we're unrolling the decoder RNN step-by-step, ph will always be 1.
            if self.hyperparams["single_mode_multi_sample"]:
                u_dist = GMM2D.from_log_pis_mus_cov_mats(
                    log_pis=log_pis,
                    mus=mus.unsqueeze(1),
                    cov_mats=Sigma_t,
                )
            else:
                u_dist = GMM2D.from_log_pis_mus_cov_mats(
                    log_pis=log_pis,
                    mus=mus.unsqueeze(1),
                    cov_mats=Sigma_pred if j == 0 else Sigma_eps,
                )

            y_dist: GMM2D = self.dynamic.iterative_dist_integration(u_dist, prev_y_dist)

            # Keeping track of the x_dist parameters, which will later be merged
            # into a set of GMMs over time.
            mus_list.append(
                y_dist.mus.reshape(
                    num_K_samples * num_samples, num_components, 1, batch_size, -1
                ).transpose(1, 3)
            )
            log_sigmas_list.append(
                y_dist.log_sigmas.reshape(
                    num_K_samples * num_samples, num_components, 1, batch_size, -1
                ).transpose(1, 3)
            )
            corrs_list.append(
                y_dist.corrs.reshape(
                    num_K_samples * num_samples, num_components, 1, batch_size
                ).transpose(1, 3)
            )

            if mode == ModeKeys.PREDICT and gmm_mode:
                p_t = y_dist.mode()
            else:
                p_t = y_dist.rsample()

            if self.hyperparams["incl_robot_node"]:
                dec_inputs = [
                    zx,
                    p_t,
                    y_r[:, j].repeat(num_samples * num_components, 1),
                ]
            else:
                dec_inputs = [zx, p_t]

            cell_input = torch.cat(dec_inputs, dim=-1)
            prev_y_dist = y_dist

            # Sampling for the next sampled_K
            if self.hyperparams["single_mode_multi_sample"]:
                next_dist = td.MultivariateNormal(
                    loc=sampled_K.squeeze(-1),
                    scale_tril=torch.sqrt(blr_layer.Sigma_nu).expand(
                        sampled_K.shape[0], -1, -1, -1
                    ),
                )
                sampled_K: torch.Tensor = next_dist.rsample().unsqueeze(
                    -1
                )  # [X*S*K*B, N, D, 1]

        if self.hyperparams["single_mode_multi_sample"] and mode == ModeKeys.PREDICT:
            y_dist = GMM2D(
                log_pis=log_pis.reshape(
                    num_K_samples * num_samples, num_components, 1, batch_size
                )
                .transpose(0, 1)
                .transpose(1, 3)
                .expand((-1, -1, prediction_horizon, -1)),
                mus=torch.concat(mus_list, dim=2).transpose(0, 3),
                log_sigmas=torch.concat(log_sigmas_list, dim=2).transpose(0, 3),
                corrs=torch.concat(corrs_list, dim=2).transpose(0, 3),
            )
        else:
            y_dist = GMM2D(
                log_pis=log_pis.reshape(
                    num_K_samples * num_samples, num_components, 1, batch_size
                )
                .transpose(1, 3)
                .expand((-1, -1, prediction_horizon, -1)),
                mus=torch.concat(mus_list, dim=2),
                log_sigmas=torch.concat(log_sigmas_list, dim=2),
                corrs=torch.concat(corrs_list, dim=2),
            )

        if mode == ModeKeys.PREDICT:
            if gmm_mode:
                y_sample = y_dist.mode()
            else:
                y_sample = y_dist.rsample()
            return y_dist, y_sample
        else:
            return y_dist

    def p_y_xz(self, *args, **kwargs):
        if self.hyperparams["adaptive"]:
            return self.p_y_xz_adaptive(*args, **kwargs)
        else:
            return self.p_y_xz_old(*args, **kwargs)

    def encoder(self, mode, enc, y_e, num_samples=None):
        """
        Encoder of the CVAE.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y_e: Encoded future tensor.
        :param num_samples: Number of samples from the latent space during Prediction.
        :return: tuple(z, kl_obj)
            WHERE
            - z: Samples from the latent space.
            - kl_obj: KL Divergenze between q and p
        """
        if mode == ModeKeys.TRAIN:
            sample_ct = self.hyperparams["k"]
        elif mode == ModeKeys.EVAL:
            sample_ct = self.hyperparams["k_eval"]
        elif mode == ModeKeys.PREDICT:
            sample_ct = num_samples
            if num_samples is None:
                raise ValueError("num_samples cannot be None with mode == PREDICT.")

        self.latent.q_dist = self.q_z_xy(mode, enc, y_e)
        self.latent.p_dist = self.p_z_x(mode, enc)

        z = self.latent.sample_q(sample_ct, mode)

        if mode == ModeKeys.TRAIN:
            kl_obj = self.latent.kl_q_p(
                self.log_writer, "%s" % str(self.node_type), self.curr_iter
            )
            if self.log_writer is not None:
                self.log_writer.log(
                    {f"{str(self.node_type)}/kl": kl_obj.item()},
                    step=self.curr_iter,
                    commit=False,
                )
        else:
            kl_obj = None

        return z, kl_obj

    def decoder(
        self,
        mode: ModeKeys,
        enc: torch.Tensor,
        x_nr_t: torch.Tensor,
        y: torch.Tensor,
        y_r: torch.Tensor,
        pos_hist: torch.Tensor,
        pos_hist_len: torch.Tensor,
        z: torch.Tensor,
        dt: torch.Tensor,
        num_samples: int,
        update_mode: UpdateMode,
    ):
        """
        Decoder of the CVAE.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param x: Input / Condition tensor.
        :param x_nr_t: Joint state of node and robot (if robot is in scene).
        :param y: Future tensor.
        :param y_r: Encoded future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z: Stacked latent state.
        :param num_samples: Number of samples from the latent space.
        :return: Log probability of y over p.
        """

        num_components = self.hyperparams["N"] * self.hyperparams["K"]
        y_dist = self.p_y_xz(
            mode,
            enc,
            x_nr_t,
            y_r,
            pos_hist,
            pos_hist_len,
            z,
            dt,
            y.shape[1],
            num_samples,
            num_components=num_components,
            update_mode=update_mode,
        )

        if self.hyperparams["single_mode_multi_sample"]:
            log_p_ynt_xz = y_dist.log_prob(torch.nan_to_num(y))
            log_p_yt_xz = torch.logsumexp(log_p_ynt_xz, dim=0, keepdim=True) - np.log(
                log_p_ynt_xz.shape[0]
            )
        else:
            log_p_yt_xz = torch.clamp(
                y_dist.log_prob(torch.nan_to_num(y)),
                max=self.hyperparams["log_p_yt_xz_max"],
            )

        if (
            self.hyperparams["log_histograms"]
            and self.log_writer
            and (self.curr_iter + 1) % 500 == 0
        ):
            self.log_writer.log(
                {
                    f"{str(self.node_type)}/log_p_yt_xz": wandb.Histogram(
                        log_p_yt_xz.detach().cpu().numpy()
                    )
                },
                step=self.curr_iter,
                commit=False,
            )
        if self.log_writer and self.hyperparams["adaptive"]:
            blr_layer: BayesianLastLayer = self.node_modules[
                self.node_type + "/decoder/last_layer"
            ]
            if num_components > 1:
                pass
            else:
                self.log_writer.log(
                    {
                        f"{str(self.node_type)}/last_layer/Sigma_eps_00": blr_layer.Sigma_eps[
                            0
                        ].item(),
                        # f'{str(self.node_type)}/last_layer/Sigma_eps_01': blr_layer.Sigma_eps[0, 1].item(),
                        # f'{str(self.node_type)}/last_layer/Sigma_eps_10': blr_layer.Sigma_eps[1, 0].item(),
                        f"{str(self.node_type)}/last_layer/Sigma_eps_11": blr_layer.Sigma_eps[
                            1
                        ].item(),
                        f"{str(self.node_type)}/last_layer/alpha": blr_layer.alpha.item()
                        if not self.hyperparams["fixed_alpha"]
                        else blr_layer.alpha,
                    },
                    step=self.curr_iter,
                    commit=False,
                )

        nan_mask = (~y.isfinite()).any(dim=-1)
        log_p_y_xz = torch.sum(log_p_yt_xz.masked_fill_(nan_mask, 0.0), dim=-1)

        # if self.hyperparams["single_mode_multi_sample"]:
        #     log_p_y_xz /= log_p_yt_xz.shape[-1]

        return log_p_y_xz

    def forward(
        self, batch: AgentBatch, update_mode: UpdateMode = UpdateMode.BATCH_FROM_PRIOR
    ) -> torch.Tensor:
        return self.train_loss(batch, update_mode=update_mode)

    def train_loss(
        self, batch: AgentBatch, update_mode: UpdateMode = UpdateMode.BATCH_FROM_PRIOR
    ) -> torch.Tensor:
        """
        Calculates the training loss for a batch.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :return: Scalar tensor -> nll loss
        """
        mode = ModeKeys.TRAIN

        enc, x_nr_t, y_e, y_r, y = self.obtain_encoded_tensors(mode, batch)

        z, kl = self.encoder(mode, enc, y_e)

        if self.hyperparams["adaptive"]:
            pos_hist: torch.Tensor = batch.agent_hist[..., :2]
        else:
            pos_hist: torch.Tensor = batch.agent_hist[
                torch.arange(batch.agent_hist.shape[0]), batch.agent_hist_len - 1
            ]

        log_p_y_xz = self.decoder(
            mode,
            enc,
            x_nr_t,
            y,
            y_r,
            pos_hist,
            batch.agent_hist_len,
            z,
            batch.dt,
            self.hyperparams["k"],
            update_mode,
        )

        log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)  # [nbs]
        log_likelihood = torch.mean(log_p_y_xz_mean)

        mutual_inf_q = mutual_inf_mc(self.latent.q_dist)
        mutual_inf_p = mutual_inf_mc(self.latent.p_dist)

        ELBO = log_likelihood - self.kl_weight * kl + 1.0 * mutual_inf_p
        loss = -ELBO

        if (
            self.hyperparams["log_histograms"]
            and self.log_writer
            and (self.curr_iter + 1) % 500 == 0
        ):
            self.log_writer.log(
                {
                    f"{str(self.node_type)}/log_p_y_xz": wandb.Histogram(
                        log_p_y_xz_mean.detach().cpu().numpy()
                    )
                },
                step=self.curr_iter,
                commit=False,
            )

        if self.log_writer:
            self.log_writer.log(
                {
                    f"{str(self.node_type)}/mutual_information_q": mutual_inf_q.item(),
                    f"{str(self.node_type)}/mutual_information_p": mutual_inf_p.item(),
                    f"{str(self.node_type)}/log_likelihood": log_likelihood.item(),
                    f"{str(self.node_type)}/loss": loss.item(),
                },
                step=self.curr_iter,
                commit=False,
            )
            if self.hyperparams["log_histograms"] and (self.curr_iter + 1) % 500 == 0:
                self.latent.summarize_for_tensorboard(
                    self.log_writer, str(self.node_type), self.curr_iter
                )
        return loss

    def predict(
        self,
        batch: AgentBatch,
        prediction_horizon,
        num_samples,
        z_mode=False,
        gmm_mode=False,
        full_dist=True,
        all_z_sep=False,
        output_dists=False,
        update_mode=UpdateMode.NO_UPDATE,
    ):
        """
        Predicts the future of a batch of nodes.

        :param batch: Input batch of data.
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param z_mode: If True: Select the most likely latent state.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :param all_z_sep: Samples each latent mode individually without merging them into a GMM.
        :param full_dist: Samples all latent states and merges them into a GMM as output.
        :return:
        """
        mode = ModeKeys.PREDICT

        enc, x_nr_t, _, y_r, _ = self.obtain_encoded_tensors(mode, batch)

        self.latent.p_dist = self.p_z_x(mode, enc)

        z, num_samples, num_components = self.latent.sample_p(
            num_samples,
            mode,
            most_likely_z=z_mode,
            full_dist=full_dist,
            all_z_sep=all_z_sep,
        )

        if self.hyperparams["adaptive"]:
            pos_hist: torch.Tensor = batch.agent_hist[..., :2]
        else:
            # This is the old n_s_t0 (just the state at the current timestep, t=0).
            pos_hist: torch.Tensor = batch.agent_hist[
                torch.arange(batch.agent_hist.shape[0]), batch.agent_hist_len - 1
            ]

        y_dist, our_sampled_future = self.p_y_xz(
            mode,
            enc,
            x_nr_t,
            y_r,
            pos_hist,
            batch.agent_hist_len,
            z,
            batch.dt,
            prediction_horizon,
            num_samples,
            num_components,
            z_mode,
            gmm_mode,
            update_mode,
        )

        if output_dists:
            return y_dist, our_sampled_future
        else:
            return our_sampled_future

    def adaptive_predict(
        self,
        batch: AgentBatch,
        prediction_horizon: int,
        num_samples: int,
        update_mode: UpdateMode,
        z_mode: bool = False,
        gmm_mode: bool = False,
        full_dist: bool = True,
        all_z_sep: bool = False,
        output_dists: bool = False,
    ):
        """
        Predicts the future of a batch of nodes.

        :param batch: Input batch of data.
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param z_mode: If True: Select the most likely latent state.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :param all_z_sep: Samples each latent mode individually without merging them into a GMM.
        :param full_dist: Samples all latent states and merges them into a GMM as output.
        :return:
        """
        if not self.hyperparams["adaptive"]:
            raise ValueError(
                "hyperparams['adaptive'] must be True to call adaptive_predict."
            )

        mode = ModeKeys.PREDICT

        enc, x_nr_t, _, y_r, _ = self.obtain_encoded_tensors(mode, batch)

        self.latent.p_dist = self.p_z_x(mode, enc)

        z, num_samples, num_components = self.latent.sample_p(
            num_samples,
            mode,
            most_likely_z=z_mode,
            full_dist=full_dist,
            all_z_sep=all_z_sep,
        )

        pos_hist: torch.Tensor = batch.agent_hist[..., :2]

        y_dist, our_sampled_future = self.p_y_xz_adaptive(
            mode,
            enc,
            x_nr_t,
            y_r,
            pos_hist,
            batch.agent_hist_len,
            z,
            batch.dt,
            prediction_horizon,
            num_samples,
            num_components,
            z_mode,
            gmm_mode,
            update_mode=update_mode,
        )

        if output_dists:
            return y_dist, our_sampled_future
        else:
            return our_sampled_future
