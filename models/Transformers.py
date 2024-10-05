import flax.linen as nn
from .layers import Tokenizer, Detokenizer, EncoderBlock, PreNormEncoderBlock



class TransformerEncoder(nn.Module):
    num_layers : int
    embed_dim : int
    num_heads : int
    dim_feedforward : int
    dropout_prob : float

    def setup(self):
        self.layers = [PreNormEncoderBlock(embed_dim=self.embed_dim, 
                                           num_heads=self.num_heads, 
                                           dim_feedforward=self.dim_feedforward, 
                                           dropout_prob=self.dropout_prob) for _ in range(self.num_layers)]

    def __call__(self, x, training):
        for l in self.layers:
            x = l(x, train=training)
        return x
    

class Transformer(nn.Module):
    num_layers: int
    embed_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: int
    variable_indices: dict

    def setup(self):
        self.tokenizer = Tokenizer(variable_indices=self.variable_indices, embed_dim=self.embed_dim)
        
        self.transformer_encoder = TransformerEncoder(num_layers=self.num_layers, 
                                                      embed_dim=self.embed_dim, 
                                                      num_heads=self.num_heads, 
                                                      dim_feedforward=self.dim_feedforward,
                                                      dropout_prob=self.dropout_prob)
        
        self.detokenizer = Detokenizer(variable_indices=self.variable_indices)
    
    def __call__(self, x, training):
        x = self.tokenizer(x)
        z = self.transformer_encoder(x, training)
        z = self.detokenizer(z)
        return z