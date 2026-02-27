import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.ops import MLP

class Q_Function(nn.Module):
    def __init__(self, 
                 input_dim=64,
                 num_ensemble: int=3,
                 num_layers: int=3,
                 num_hidden_dim: int=2048,
                 dropout: float=0.0,
                 init_method = 'xavier'
                 ):
        """
        Each Q function is an MLP with num_layers and num_hidden_dim with activation and normalization scheme.
        For example, activation can be tanh or GELU and normalization can be LayerNorm or BatchNorm.
        """
        super().__init__()
        self.input_dim = input_dim
        self.init_method = init_method
        self.num_ensemble = num_ensemble
        
        self.num_hidden_dim = num_hidden_dim
        self.q_ensembles_hidden_layers = nn.ModuleList(
            [
                MLP(
                    in_channels = self.input_dim,
                    hidden_channels = [*[self.num_hidden_dim for _ in range(num_layers)], self.num_hidden_dim],
                    activation_layer = nn.ELU,
                    norm_layer = nn.LayerNorm,
                    dropout = dropout
                )
                for _ in range(self.num_ensemble)
            ]
        )
        self.q_ensembles_output_layers = nn.ModuleList(
            [nn.Linear(
                    in_features = self.num_hidden_dim, 
                    out_features = 1
                )
                for _ in range(self.num_ensemble)
            ]
        )

        # === apply init AFTER modules are built ===
        if self.init_method == "xavier":
            # don't touch vision_backend (likely pretrained)
            self.q_ensembles_hidden_layers.apply(self._xavier_init_fn)
            self.q_ensembles_output_layers.apply(self._xavier_init_fn)
    
    def forward(self, flattened_tensor):
        """
        flattened_tensor: (batch, dim)

        Outputs mean value of Q function ensemble
        (batch, mean_val)
        """
        running_sum = 0

        for i in range(self.num_ensemble):
            Q_i_hidden_layer_output = self.q_ensembles_hidden_layers[i](flattened_tensor)
            Q_i_val = self.q_ensembles_output_layers[i](Q_i_hidden_layer_output)
            running_sum += Q_i_val
        
        return running_sum / self.num_ensemble
    
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
