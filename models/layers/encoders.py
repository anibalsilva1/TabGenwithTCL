from typing import Union, Callable
import flax.linen as nn

from .multiAttnHead import MultiheadAttention

class Encoder(nn.Module):
    hidden_dim: Union[int, list]
    latent_dim: int
    activation_fn: Callable = nn.silu

    @nn.compact
    def __call__(self, x):
        if isinstance(self.hidden_dim, (list, tuple)):
            for hidden_dim in self.hidden_dim:
                x = nn.Dense(hidden_dim)(x)
                x = self.activation_fn(x)        
        else:
            x = nn.Dense(self.hidden_dim)(x)
            x = self.activation_fn(x)
        
        mu = nn.Dense(self.latent_dim, name='mean')(x)
        logvar = nn.Dense(self.latent_dim, name='logvar')(x)
        
        return mu, logvar

class DGEncoder(nn.Module):
    hidden_dim: Union[int, list]
    latent_dim: int
    embed_dim: int
    activation_fn: Callable = nn.silu

    @nn.compact
    def __call__(self, x):
        if isinstance(self.hidden_dim, (list, tuple)):
            for hidden_dim in self.hidden_dim:
                x = nn.DenseGeneral(features=(hidden_dim, self.embed_dim), axis=(-2, -1))(x)
                x = self.activation_fn(x)
        
        else:
            x = nn.DenseGeneral(features=(self.hidden_dim, self.embed_dim), axis=(-2, -1))(x)
            x = self.activation_fn(x)
        
        mu = nn.DenseGeneral(features=(self.latent_dim, self.embed_dim), axis=(-2, -1), name='mu')(x)
        logvar = nn.DenseGeneral(features=(self.latent_dim, self.embed_dim), axis=(-2, -1), name='logvar')(x)

        return mu, logvar

class TensoFormerEncoder(nn.Module):
    hidden_dim: Union[int, list]
    latent_dim: int
    embed_dim: int
    activation_fn: Callable = nn.silu

    @nn.compact
    def __call__(self, x):
        if isinstance(self.hidden_dim, (list, tuple)):
            for hidden_dim in self.hidden_dim:
                x = nn.DenseGeneral(features=(hidden_dim, self.embed_dim), axis=(-2, -1))(x)
                x = self.activation_fn(x)
        else:
            x = nn.DenseGeneral(features=(self.hidden_dim, self.embed_dim), axis=(-2, -1))(x)
            x = self.activation_fn(x)
        
        x = nn.DenseGeneral(features=(self.latent_dim, self.embed_dim), axis=(-2, -1))(x)
        return x
    

class EncoderBlock(nn.Module):
    embed_dim : int
    num_heads : int
    dim_feedforward : int
    dropout_prob : float

    def setup(self):
        # Attention layer
        self.self_attn = MultiheadAttention(embed_dim=self.embed_dim,
                                            num_heads=self.num_heads)
        # Two-layer MLP
        self.linear = [
            nn.Dense(self.dim_feedforward),
            nn.Dropout(self.dropout_prob),
            nn.relu,
            nn.Dense(self.embed_dim)
        ]
        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(self, x, train):
        # Attention part
        attn_out, _ = self.self_attn(x)
        x = x + self.dropout(attn_out, deterministic=not train)
        x = self.norm1(x)

        # MLP part
        linear_out = x
        for l in self.linear:
            linear_out = l(linear_out) if not isinstance(l, nn.Dropout) else l(linear_out, deterministic=not train)
        x = x + self.dropout(linear_out, deterministic=not train)
        x = self.norm2(x)

        return x
    

class PreNormEncoderBlock(nn.Module):
    embed_dim : int
    num_heads : int
    dim_feedforward : int
    dropout_prob : float

    def setup(self):
        # Attention layer
        self.self_attn = MultiheadAttention(embed_dim=self.embed_dim,
                                            num_heads=self.num_heads)
        # Two-layer MLP
        self.linear = [
            nn.Dense(self.dim_feedforward),
            nn.Dropout(self.dropout_prob),
            nn.relu,
            nn.Dense(self.embed_dim)
        ]
        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(self, x, train):
        # Attention part
        x_residual = self.norm1(x)
        attn_out, _ = self.self_attn(x_residual)
        x = x + self.dropout(attn_out, deterministic=not train)

        # MLP part
        x_residual = self.norm2(x)
        for l in self.linear:
            x_residual = l(x_residual) if not isinstance(l, nn.Dropout) else l(x_residual, deterministic=not train)
        x = x + self.dropout(x_residual, deterministic=not train)

        return x