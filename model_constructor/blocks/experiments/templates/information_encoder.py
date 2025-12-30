import torch
import torch.nn as nn
import einops

from .multimodal_encoder import MultiModalEncoderTemplate
from ...basic_blocks.transformer_encoder import NonCausalTransformerEncoder
from ..utils.pos_encoding import get_sinusoidal_pos_encoding



class InformationEncoder(MultiModalEncoderTemplate):
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

                 num_cameras: int,
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
            self.action_projection = nn.Sequential(
            *[
                nn.Linear(self.action_dim, transformer_d_model),
                nn.ELU(),
            ]
        )
        
        self.semantic_projection = None    
        if self.use_cond_semantic and self.use_cond_semantic_projection:
            self.semantic_projection = torch.nn.Sequential(
                *[
                    torch.nn.Linear(cond_semantic_dim, self.transformer_hidden_dim),
                    torch.nn.ELU(),
                ]
            )

        self.proprio_projection = torch.nn.Sequential(
            *[
                torch.nn.Linear(cond_proprio_dim, self.transformer_hidden_dim),
                torch.nn.ELU(),
            ]
        )

        self.num_cameras = num_cameras
        self.visual_projection = nn.ModuleList(
            [torch.nn.Sequential(
                *[
                    torch.nn.Linear(cond_visual_dim, self.transformer_hidden_dim),
                    torch.nn.ELU(),
                ]
            ) for _ in range(num_cameras)]
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

            Parmaeters:
                
                cond_proprio: (batch, sequence, features) shape
                cond_visual: (batch, num_frames, sequence, channel, height, width) 
                             (batch, num_frames, channel, height, width) 
                          or (batch, num_frames, channel, sequence) shape
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
           and (cond_visual.ndim == 6 or cond_visual.ndim == 5 or cond_visual.ndim == 4) \
           and (action is None or action.ndim == 3) \
           and (cond_semantic is None or cond_semantic.ndim == 2 or cond_semantic.ndim == 3)
        
        batch_size = cond_proprio.shape[0]

        # proprio data
        proprio_input = self.proprio_projection(cond_proprio)

        # visual data
        if cond_visual.ndim == 4: 
            cond_visual = einops.rearrange(cond_visual, 'b n c s -> b n s c')
        if cond_visual.ndim == 5: 
            cond_visual = einops.rearrange(cond_visual, 'b n c h w -> b n (h w) c')
        if cond_visual.ndim == 6: 
            cond_visual = einops.rearrange(cond_visual, 'b n t c h w -> b n (t h w) c')
        
        if cond_visual.shape[1] != self.num_cameras:
            raise ValueError(f"Number of cameras {self.num_cameras} != Number of visual frames {cond_visual.shape[1]}")
        
        projected_visuals = []
        for i in range(self.num_cameras):
            projected_visuals.append(self.visual_projection[i](einops.rearrange(cond_visual[:, i, :, :], 'b s c -> b 1 s c')))
        projected_visual_input = einops.rearrange(torch.cat(projected_visuals, dim=1), 'b n s c -> b (n s) c')

        encoder_input = torch.cat([projected_visual_input, proprio_input], dim=1) 

        # semantic data
        if self.use_cond_semantic:
            if not self.use_cond_semantic_projection and cond_semantic.shape[-1] != self.transformer_hidden_dim:
                raise ValueError(f"cond_semantic must have dimension {self.transformer_hidden_dim}, got {cond_semantic.shape[-1]}!")
            semantic_input = self.semantic_projection(cond_semantic) if self.use_cond_semantic_projection else cond_semantic
            if semantic_input.ndim == 2: 
                semantic_input = einops.rearrange(semantic_input, 'b d -> b 1 d')
            encoder_input = torch.cat([semantic_input, encoder_input], dim=1)
        
        # action data
        if self.use_action: 
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
    