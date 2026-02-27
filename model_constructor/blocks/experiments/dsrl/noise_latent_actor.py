import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.ops import MLP

class Noise_Latent_Actor(nn.Module):
    def __init__(self, 
                 input_dim=64,
                 action_chunk_size: int=50,
                 action_dim: int=32, # 40 actions with 24 action dimension
                 num_layers: int=3,
                 num_hidden_dim: int=2048,
                 dropout: float=0.0,
                 init_method = 'xavier'
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.action_chunk_size = action_chunk_size
        self.action_dim = action_dim
        self.init_method = init_method

        self.model = MLP(
            in_channels = self.input_dim,
            hidden_channels = [*[num_hidden_dim for _ in range(num_layers)], num_hidden_dim],
            activation_layer = nn.ELU,
            norm_layer = nn.LayerNorm,
            dropout = dropout
        )
        self.last_layer = nn.Sequential(
                nn.Linear(
                    in_features = num_hidden_dim, 
                    out_features = self.action_chunk_size * self.action_dim
                ),
                nn.Tanh() # in order to prevent going outside the domain of the Gaussian noise
        )

        # === apply init AFTER modules are built ===
        if self.init_method == "xavier":
            self.model.apply(self._xavier_init_fn)
            self.last_layer.apply(self._xavier_init_fn)

    def forward(self, flattened_tensor):
        """
        flattened_tensor: (batch, dim)
        Output shape: (batch, action_chunk_size, action_dim)
        """
        intermediate_output = self.model(flattened_tensor)
        noisy_action = self.last_layer(intermediate_output)
        num_batch, _ = noisy_action.shape
        
        return noisy_action.reshape(num_batch, self.action_chunk_size, self.action_dim)

    def _xavier_init_fn(self, m: nn.Module):
        """
        This runs on every submodule when called via .apply(...)
        We only touch Linear and Conv weights.
        """
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict, strict=True)