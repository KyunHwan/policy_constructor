import torch
import torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int,
                 dropout: float,
                 activation: str,
                 batch_first: bool,
                 tgt_is_causal: bool,
                 num_layers: int,
                 num_tokens: int,
                 ):
        """
        Parameters:
            d_model (int): the number of expected features in the input.

            nhead (int): the number of heads in the multiheadattention models.

            dim_feedforward (int): the dimension of the feedforward network model.

            dropout (float): the dropout value.

            activation (Union[str, Callable[[Tensor], Tensor]]): the activation function of the intermediate layer, can be a string (“relu” or “gelu”) or a unary callable. 
                                                                 Default: relu

            num_layers (int): the number of sub-decoder-layers in the decoder.

            
            batch_first (bool): If True, then the input and output tensors are provided as (batch, seq, feature). 
                                Default: False (seq, batch, feature).
            
            tgt_is_causal (bool): If specified, applies a causal mask as tgt mask. 
                                  Default: False. 
                                  Warning: tgt_is_causal provides a hint that tgt_mask is the causal mask. 
                                  Providing incorrect hints can result in incorrect execution, including forward and backward compatibility.
            
            num_tokens (int): number of tokens given as input to the decoder
            
        """
        super().__init__()
        self.batch_first = batch_first
        self.tgt_is_causal = tgt_is_causal
        self.num_tokens = num_tokens
        self.d_model = d_model

        self.decoder = nn.TransformerDecoder(
                            decoder_layer = nn.TransformerDecoderLayer(
                                                d_model=d_model,
                                                nhead=nhead,
                                                dropout=dropout,
                                                activation=activation,
                                                dim_feedforward=dim_feedforward,
                                                batch_first=batch_first
                                            ), 
                            num_layers = num_layers, )
        
        
        # masking self-attention to make the action causal --> similar to diffusion policy
        self.register_buffer('tgt_causal_mask', torch.nn.Transformer.generate_square_subsequent_mask(self.num_tokens))

    def forward(self, tgt_input: torch.Tensor, memory_input: torch.Tensor) -> torch.Tensor:
        """
            Be careful about the tensor shape - (seq, batch, feature) if batch_first is False else (batch, seq, feature)
        """
        return self.decoder(
                    tgt = tgt_input,
                    memory = memory_input,
                    tgt_mask = self.tgt_causal_mask if self.tgt_is_causal else None,
                    tgt_is_causal=self.tgt_is_causal
                )