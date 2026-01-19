import torch
import torch.nn as nn
import einops

class ProprioProjector(nn.Module):
    def __init__(self, 
                 cond_proprio_dim: int,
                 transformer_d_model: int,
                 **kwargs):
        super().__init__(**kwargs)

        # inputs should be consistent
        
        self.transformer_hidden_dim = transformer_d_model

        self.proprio_projection = None
        if cond_proprio_dim != self.transformer_hidden_dim:
            self.proprio_projection = torch.nn.Sequential(
                *[
                    torch.nn.Linear(cond_proprio_dim, self.transformer_hidden_dim * 2),
                    torch.nn.ELU(),
                    torch.nn.Linear(self.transformer_hidden_dim * 2, self.transformer_hidden_dim * 2),
                    torch.nn.ELU(),
                    torch.nn.Linear(self.transformer_hidden_dim * 2, self.transformer_hidden_dim),
                    nn.LayerNorm(self.transformer_hidden_dim)
                ]
            )

    def forward(self,
                cond_proprio: torch.Tensor, # latent proprio features,
                cond_visual: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """
            For the below tensor shape and ordering to work, transformer_batch_first should be set to True

            Parmaeters:
                
                cond_proprio: (batch, sequence, features) shape
                cond_visual: (batch, num_frames, sequence, channel, height, width) 
                             (batch, num_frames, channel, height, width) 
                          or (batch, num_frames, channel, sequence) shape
                
            Output:
                
        """
        assert cond_proprio.ndim == 3 \
           and (cond_visual.ndim == 6 or cond_visual.ndim == 5 or cond_visual.ndim == 4) \
           and self.transformer_hidden_dim

        # visual data
        if cond_visual.ndim == 4: 
            cond_visual = einops.rearrange(cond_visual, 'b n c s -> b n s c')
        if cond_visual.ndim == 5: 
            cond_visual = einops.rearrange(cond_visual, 'b n c h w -> b n (h w) c')
        if cond_visual.ndim == 6: 
            cond_visual = einops.rearrange(cond_visual, 'b n t c h w -> b n (t h w) c')
        
        projected_visual_input = einops.rearrange(cond_visual, 'b n s c -> b (n s) c')

        # proprio data
        if self.proprio_projection is None:
            return torch.cat([projected_visual_input, cond_proprio], dim=1) 
        else:
            return torch.cat([projected_visual_input, self.proprio_projection(cond_proprio)], dim=1) 
        

        