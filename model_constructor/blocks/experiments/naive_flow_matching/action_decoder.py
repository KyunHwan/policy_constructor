import torch
import torch.nn as nn
import einops

from ..templates.flow_matching import FlowMatchingBodyTemplate
from ...basic_blocks.transformer_decoder import TransformerDecoder
from ..utils.pos_encoding import get_sinusoidal_pos_encoding
from ..utils.time_embedding import get_time_embedding



class ActionDecoder(FlowMatchingBodyTemplate):
    def __init__(self, 
                 action_dim: int,

                 transformer_d_model: int,
                 transformer_nhead: int,
                 transformer_dim_feedforward: int,
                 transformer_dropout: float,
                 transformer_activation: str,
                 transformer_batch_first: bool,
                 transformer_tgt_is_causal: bool,
                 transformer_num_layers: int,
                 transformer_action_chunk_size: int,

                 use_cond_semantic_projection: bool,
                 use_cond_semantic: bool = False,
                 cond_semantic_dim: int | None=None,
                 **kwargs):
        super().__init__(**kwargs)

        assert ((use_cond_semantic == True and cond_semantic_dim is not None) or (use_cond_semantic == False and cond_semantic_dim is None))
        
        self.transformer_hidden_dim = transformer_d_model
        self.use_cond_semantic = use_cond_semantic
        self.use_cond_semantic_projection = use_cond_semantic_projection
        self.cond_semantic_dim = cond_semantic_dim

        self.noise_projection = nn.Sequential(
            *[
                nn.Linear(action_dim, transformer_d_model),
                nn.ELU(),
            ]
        )

        self.semantic_projection = None
        if self.use_cond_semantic and self.use_cond_semantic_projection:
            self.semantic_projection = nn.Sequential(
                *[
                    nn.Linear(self.cond_semantic_dim, self.transformer_hidden_dim),
                    nn.ELU(),
                ]
            )

        self.time_mlp = nn.Sequential(
            nn.Linear(transformer_d_model, transformer_d_model * 2),
            nn.SiLU(), # SiLU is standard for diffusion/flow MLPs
            nn.Linear(transformer_d_model * 2, transformer_d_model),
        )

        self.decoder = TransformerDecoder(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            activation=transformer_activation,
            batch_first=transformer_batch_first,
            tgt_is_causal=transformer_tgt_is_causal,
            num_layers=transformer_num_layers,
            num_tokens=transformer_action_chunk_size,
        )

        self.output_layer = nn.Sequential(
            *[
                nn.ELU(),
                nn.Linear(transformer_d_model, action_dim),
            ]
        )

    def forward(self, 
                time: torch.Tensor,
                noise: torch.Tensor, 
                memory_input: torch.Tensor, # memory input as cross-attention
                discrete_semantic_input: torch.Tensor | None=None,
                **kwargs) -> torch.Tensor:
        """
            For the below tensor shape and ordering to work, transformer_batch_first should be set to True

            Parmaeters:
                time: (batch, time) shape; time step in the ODE integration process
                noise: (batch, sequence, features) shape
                memory_input: (batch, sequence, features) shape
                discrete_semantic_input: (batch, features) or (batch, num_semantic, features) shape
                
            Output:
                action: (batch, sequence, feature) shape
        """
        assert noise.ndim == 3 and memory_input.ndim == 3 and memory_input.shape[2] == self.transformer_hidden_dim
        
        if discrete_semantic_input is not None: 
           assert (discrete_semantic_input.ndim == 2 and discrete_semantic_input.shape[1] == self.transformer_hidden_dim) \
               or (discrete_semantic_input.ndim == 3 and discrete_semantic_input.shape[2] == self.transformer_hidden_dim)
        
        noise_input = self.noise_projection(noise)
        
        
        # conditioning
        memory_input = memory_input + \
                       get_sinusoidal_pos_encoding(memory_input.shape[1], self.transformer_hidden_dim, memory_input.device)

        # time embedding for flow matching
        if time.ndim > 1: time = time.squeeze(-1)
        time_emb = get_time_embedding(time, self.transformer_hidden_dim) # (batch, d_model)
        time_emb = einops.rearrange(self.time_mlp(time_emb), 'b d -> b 1 d') # (batch, 1, d_model)
        memory_input = torch.cat([time_emb, memory_input], dim=1)

        # semantic input
        if self.use_cond_semantic:
            if not self.use_cond_semantic_projection and discrete_semantic_input.shape[-1] != self.transformer_hidden_dim:
                raise ValueError(f"cond_semantic must have dimension {self.transformer_hidden_dim}, got {discrete_semantic_input.shape[-1]}!")
            semantic_input = self.semantic_projection(discrete_semantic_input) if self.use_cond_semantic_projection else discrete_semantic_input
            if semantic_input.ndim == 2: 
                semantic_input = einops.rearrange(semantic_input, 'b d -> b 1 d')
            memory_input = torch.cat([semantic_input, memory_input], dim=1)

        decoded_output = self.decoder(noise_input, memory_input)

        output = self.output_layer(decoded_output)

        return output
    