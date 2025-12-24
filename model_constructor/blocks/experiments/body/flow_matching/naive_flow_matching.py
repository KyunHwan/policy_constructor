from abc import ABC
import torch
import torch.nn as nn
import einops

from ..flow_matching_body_template import FlowMatchingBodyTemplate
from ....basic_blocks.transformer_decoder import TransformerDecoder


class NaiveFlowMatching(FlowMatchingBodyTemplate):
    def __init__(self, 
                 action_dim: int,
                 cond_proprio_dim: int,
                 cond_visual_dim: int,

                 transformer_d_model: int,
                 transformer_nhead: int,
                 transformer_dim_feedforward: int,
                 transformer_dropout: float,
                 transformer_activation: str,
                 transformer_batch_first: bool,
                 transformer_tgt_is_causal: bool,
                 transformer_num_layers: int,
                 transformer_action_chunk_size: int,
                 **kwargs):
        super().__init__(**kwargs)

        self.noise_projection = torch.nn.Linear(action_dim, transformer_d_model)
        self.proprio_projection = torch.nn.Sequential(
            *[torch.nn.Linear(cond_proprio_dim, transformer_d_model)]
        )
        self.visual_projection = torch.nn.Linear(cond_visual_dim, transformer_d_model)

        self.decoder = TransformerDecoder(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            activation=transformer_activation,
            batch_first=transformer_batch_first,
            tgt_is_causal=transformer_tgt_is_causal,
            num_layers=transformer_num_layers,
            action_chunk_size=transformer_action_chunk_size,
        )

        self.mlp = torch.nn.Sequential(
            *[torch.nn.Sequential(
                torch.nn.Linear(transformer_d_model, transformer_d_model),
                torch.nn.ELU()
            ) for _ in range(num_layers)]
        )

        self.output_layer = torch.nn.Linear(transformer_d_model, action_dim)
        # nn.Softsign
        
    def forward(self, 
                time: float,
                noise: torch.Tensor, 
                cond_proprio: torch.Tensor, 
                cond_visual: torch.Tensor, 
                cond_semantic: torch.Tensor | None=None,
                **kwargs) -> torch.Tensor:
        """
            Parmaeters:
                time: time step in the ODE integration process
                noise: (batch, sequence, feature) shape
                cond_proprio: (batch, sequence, channel) shape
                cond_visual: (batch, num_frames, channel, H, W) shape
                cond_semantic: (batch, features) or (batch, 1, features) shape
                
            Output:
                action: (batch, sequence, feature) shape
        """
        
        noise_input = self.noise_projection(noise)
        proprio_input = self.proprio_projection(cond_proprio)
        visual_input = self.visual_projection(cond_visual)

        x = noise_encoded + proprio_encoded + visual_encoded

        x = self.mlp(x)

        return self.output_layer(x)