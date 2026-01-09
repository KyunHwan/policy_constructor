import torch
import torch.nn as nn
import einops

from ..templates.multimodal_encoder import MultiModalEncoderTemplate
from ..templates.information_encoder import InformationEncoder


class VQVAE_Prior(MultiModalEncoderTemplate):
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
                 
                 use_cond_semantic: bool,
                 use_cond_semantic_projection: bool,
                 cond_semantic_dim: int | None,
                 num_cameras: int,
                 use_cls_token: bool,
                 num_cls_token: int,
                 
                 **kwargs):
        super().__init__(**kwargs)


        self.encoder = InformationEncoder(
            cond_proprio_dim=cond_proprio_dim,
            cond_visual_dim=cond_visual_dim,
            transformer_d_model=transformer_d_model,
            transformer_nhead=transformer_nhead,
            transformer_dim_feedforward=transformer_dim_feedforward,
            transformer_dropout=transformer_dropout,
            transformer_activation=transformer_activation,
            transformer_batch_first=transformer_batch_first,
            transformer_num_layers=transformer_num_layers,
            use_cls_token=use_cls_token,
            num_cls_token=num_cls_token,
            use_action=False,
            action_dim=None,
            use_cond_semantic=use_cond_semantic,
            use_cond_semantic_projection=use_cond_semantic_projection,
            cond_semantic_dim=cond_semantic_dim,
            num_cameras=num_cameras,
            **kwargs
        )

    def forward(self,
                cond_proprio: torch.Tensor, # latent proprio features
                cond_visual: torch.Tensor, # latent visual features
                cond_semantic: torch.Tensor | None=None, # latent semantic features
                action: torch.Tensor | None=None, # latent action features
                **kwargs) -> torch.Tensor:
        """
            For the below tensor shape and ordering to work, transformer_batch_first should be set to True

            Parmaeters:
                
                cond_proprio: (batch, sequence, features) shape
                cond_visual: (batch, num_frames, channel, height, width) or (batch, num_frames, channel, sequence) shape   
                             The visual information should only be that of the head camera                           
                action: (batch, sequence, features) shape
                cond_semantic: (batch, features) or (batch, 1, features) shape
                
            Output:
                action: (batch, sequence, feature) shape
        """
        return self.encoder(cond_proprio=cond_proprio, 
                            cond_visual=cond_visual,
                            cond_semantic=cond_semantic,
                            action=action,)['cls_token']
                            