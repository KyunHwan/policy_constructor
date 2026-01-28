import torch
import torch.nn as nn
import einops

from ..templates.multimodal_encoder import MultiModalEncoderTemplate
from ...basic_blocks.transformer_encoder import NonCausalTransformerEncoder
from ..utils.pos_encoding import get_sinusoidal_pos_encoding



class InfoEmbedder(MultiModalEncoderTemplate):
    def __init__(self, 
                 cond_proprio_dim: int,
                 cond_visual_dim: int,

                 transformer_d_model: int,
                 transformer_nhead: int,
                 transformer_dim_feedforward: int,
                 transformer_dropout: float,
                 transformer_activation: str,
                 transformer_batch_first: bool,
                 transformer_num_layers: int,
                 
                 use_cls_token: bool,
                 num_cls_token: int,
                 use_action: bool,
                 action_dim: int | None,
                 use_cond_semantic: bool,
                 use_cond_semantic_projection: bool,
                 cond_semantic_dim: int | None,
                 **kwargs):
        super().__init__(**kwargs)

        # inputs should be consistent
        assert ((use_action == True and action_dim is not None) or (use_action == False and action_dim is None)) \
           and ((use_cond_semantic == True and cond_semantic_dim is not None) or (use_cond_semantic == False and cond_semantic_dim is None))
        
        self.transformer_hidden_dim = transformer_d_model
        
        self.use_action = use_action
        self.action_dim = action_dim

        self.use_cond_semantic = use_cond_semantic
        self.use_cond_semantic_projection = use_cond_semantic_projection
        self.cond_semantic_dim = cond_semantic_dim

        self.use_cls_token = use_cls_token
        
        self.cls_token = None
        if self.use_cls_token:
            self.num_cls_token = num_cls_token
            self.cls_token = nn.Parameter(torch.zeros(self.num_cls_token, self.transformer_hidden_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.action_projection = None
        if self.use_action:
            if action_dim != self.transformer_hidden_dim:
                self.action_projection = nn.Sequential(
                    *[
                        nn.LayerNorm(action_dim),
                        nn.Linear(action_dim, 2 * transformer_d_model),
                        nn.SiLU(),
                        nn.Dropout(p=0.05),
                        nn.Linear(2 * transformer_d_model, transformer_d_model),
                    ]
                )
        
        self.semantic_projection = None
        if self.use_cond_semantic and self.use_cond_semantic_projection:
            if cond_semantic_dim != self.transformer_hidden_dim:
                self.semantic_projection = torch.nn.Sequential(
                    *[
                        nn.LayerNorm(cond_semantic_dim),
                        nn.Linear(cond_semantic_dim, 2 * transformer_d_model),
                        nn.SiLU(),
                        nn.Dropout(p=0.05),
                        nn.Linear(2 * transformer_d_model, transformer_d_model),
                    ]
                )
        self.proprio_projection = None
        if cond_proprio_dim != self.transformer_hidden_dim:
            self.proprio_projection = torch.nn.Sequential(
                *[
                    nn.LayerNorm(cond_proprio_dim),
                    nn.Linear(cond_proprio_dim, 2 * transformer_d_model),
                    nn.SiLU(),
                    nn.Dropout(p=0.05),
                    nn.Linear(2 * transformer_d_model, transformer_d_model),
                ]
            )

        self.visual_projection = None
        if cond_visual_dim != self.transformer_hidden_dim:
            self.visual_projection = torch.nn.Sequential(
                    *[
                        nn.LayerNorm(cond_visual_dim),
                        nn.Linear(cond_visual_dim, 2 * transformer_d_model),
                        nn.SiLU(),
                        nn.Dropout(p=0.05),
                        nn.Linear(2 * transformer_d_model, transformer_d_model),
                    ]
                )

        self.encoder = NonCausalTransformerEncoder(
            d_model=self.transformer_hidden_dim,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            activation=transformer_activation,
            batch_first=transformer_batch_first,
            num_layers=transformer_num_layers,
        )

    def forward(self,
                cond_proprio: torch.Tensor, # latent proprio features
                cond_visual: torch.Tensor, # latent visual features
                cond_semantic: torch.Tensor | None=None, # latent semantic features
                action: torch.Tensor | None=None, # latent action features
                **kwargs) -> dict[str, torch.Tensor]:
        """
            For the below tensor shape and ordering to work, transformer_batch_first should be set to True

            Parameters:
                
                cond_proprio: (batch, sequence, features) shape
                cond_visual: (batch, sequence, features) shape
                cond_semantic: (batch, features) 
                            or (batch, num_semantic, features) shape
                action: (batch, sequence, features) shape
                
            Output:
                {
                    'cls_token': (batch, 1, feature) shape (this is the first token in the sequence of encoder_output)
                    'encoder_output': (batch, sequence, feature) shape
                }
        """
        assert cond_proprio.ndim == 3 \
           and cond_visual.ndim == 3 \
           and (action is None or action.ndim == 3) \
           and (cond_semantic is None or cond_semantic.ndim == 2 or cond_semantic.ndim == 3)
        
        batch_size = cond_proprio.shape[0]

        # proprio data
        if self.proprio_projection is None:
            proprio_input = cond_proprio
        else:
            proprio_input = self.proprio_projection(cond_proprio)
        
        if self.visual_projection is not None:
            cond_visual = self.visual_projection(cond_visual)
        encoder_input = torch.cat([cond_visual, proprio_input], dim=1) 

        # semantic data
        if self.use_cond_semantic:
            if not self.use_cond_semantic_projection and cond_semantic.shape[-1] != self.transformer_hidden_dim:
                raise ValueError(f"cond_semantic must have dimension {self.transformer_hidden_dim}, got {cond_semantic.shape[-1]}!")
            if self.semantic_projection is None:
                semantic_input = cond_semantic
            else:
                semantic_input = self.semantic_projection(cond_semantic) if self.use_cond_semantic_projection else cond_semantic
            if semantic_input.ndim == 2: 
                semantic_input = einops.rearrange(semantic_input, 'b d -> b 1 d')
            encoder_input = torch.cat([semantic_input, encoder_input], dim=1)
        
        # action data
        if self.use_action: 
            if self.action_projection is None:
                action_input = action
            else:
                action_input = self.action_projection(action)
            encoder_input = torch.cat([action_input, encoder_input], dim=1)

        # position embedding
        encoder_input = encoder_input + get_sinusoidal_pos_encoding(encoder_input.shape[1], self.transformer_hidden_dim, encoder_input.device)

        # cls token
        if self.use_cls_token:
            encoder_input = torch.cat([self.cls_token.expand(batch_size, self.num_cls_token, self.transformer_hidden_dim), encoder_input], dim=1)

        encoder_output = self.encoder(encoder_input)

        return {
            'cls_token' : encoder_output[:, :self.num_cls_token, :] if self.use_cls_token else None,
            'encoder_output' : encoder_output
        }
    