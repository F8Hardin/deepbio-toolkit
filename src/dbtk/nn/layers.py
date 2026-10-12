import abc
import copy
from explainable_attention.utils import flex_attention_with_scores, naive_flex_attention_with_scores
import lightning as L
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.attention.flex_attention import flex_attention, _vmap_for_bhqkv
import torch.nn.functional as F
from typing import Any, Callable, cast, List, Optional, Sequence, Tuple, TypeVar, Union
import warnings
from mamba_ssm import Mamba3

from .._utils import deprecated, export

try:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex
except ImportError:
    from torch._higher_order_ops.flex_attention import TransformGetItemToIndex

_compiled_flex_attention = torch.compile(flex_attention)

# Multi-head Attention Mechanisms ------------------------------------------------------------------

class DataContainer:
    """
    A placeholder object for storing data without notifying torch module instance of changes.
    """
    ...

@export
class MultiheadAttention(nn.Module):

    @staticmethod
    def relative_positional_encoding() -> Callable:
        def score_mod(score, b, h, q_idx, kv_idx):
            return score + (q_idx - kv_idx)
        return score_mod

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        score_mod: Optional[Callable] = None,
        block_mask = None,
        head_embed_dim: Optional[int] = None,
        device=None,
        dtype=None,
        _compile: bool = True
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.score_mod = score_mod
        self.block_mask = block_mask
        self.bias = bias

        if head_embed_dim is None:
            assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads if head_embed_dim is not provided"
            head_embed_dim = embed_dim // num_heads
        self.head_embed_dim = head_embed_dim
        assert self.dropout == 0.0, "Dropout is not currently supported."

        # Default no-op score mod
        if self.score_mod is None:
            self.score_mod = lambda score, *_: score

        # Parameters
        self.in_proj_weight = nn.Parameter(
            torch.randn((3 * embed_dim, self.head_embed_dim*num_heads), **factory_kwargs)
        )
        self.register_parameter("q_proj_weight", None)
        self.register_parameter("k_proj_weight", None)
        self.register_parameter("v_proj_weight", None)
        if bias:
            self.in_proj_bias = nn.Parameter(torch.randn(3 * self.head_embed_dim*num_heads, **factory_kwargs))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = NonDynamicallyQuantizableLinear(
            self.head_embed_dim * num_heads, embed_dim, bias=bias, **factory_kwargs
        )

        # Flex Attention properties
        self._data_container = DataContainer()
        self._data_container.mask = None

        # Score mod with masking function
        def _score_mod_with_mask(score, b, h, q_idx, kv_idx):
            score = self.score_mod(score, b, h, q_idx, kv_idx)
            score = torch.where(self._data_container.mask[b, q_idx, kv_idx], score, float("-inf"))
            return score
        self._score_mod_with_mask = _score_mod_with_mask

        if _compile:
            self.compiled_flex_attention()
        else:
            self.eager_flex_attention()

    def compiled_flex_attention(self):
        self._flex_attention = torch.compile(flex_attention_with_scores)
        return self

    def eager_flex_attention(self):
        # only use our eager implementation here. compiled version is
        # extremely slow right now...
        self._flex_attention = naive_flex_attention_with_scores
        return self

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights=True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights=True,
        is_causal=False
    ) -> torch.Tensor:
        b, n_q, _ = query.shape
        _, n_k, _ = key.shape

        # Compute linear projections
        q, k, v = F._in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)
        q = q.view((b, n_q, self.num_heads, self.head_embed_dim)).transpose(-2, -3)
        k = k.view((b, n_k, self.num_heads, self.head_embed_dim)).transpose(-2, -3)
        v = v.view((b, n_k, self.num_heads, self.head_embed_dim)).transpose(-2, -3)

        try:
            # Compute attention mask
            self._data_container.mask = self.merge_masks(attn_mask, key_padding_mask, query, _native=False)

            # Determine score mod to use
            score_mod = self.score_mod if self._data_container.mask is None else self._score_mod_with_mask

            # Compute attention
            attention = self._flex_attention(
                q, k, v,
                score_mod=score_mod,
                block_mask=self.block_mask,
                return_attention_scores=need_weights
            )

            if need_weights:
                attention, attention_weights = attention
            else:
                attention_weights = None

            # Output projection
            attention = attention.transpose(-2, -3).reshape((b, n_q, self.num_heads*self.head_embed_dim)) # [b, n_q, d_k*num_heads]
            output = self.out_proj(attention)

            if need_weights:
                if average_attn_weights:
                    attention_weights = attention_weights.mean(-3)
                return output, attention_weights
            return output
        finally:
            # Remove the mask when we're done
            self._data_container.mask = None

    def merge_masks(
        self,
        attention_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        query: torch.Tensor,
        _native: bool = True
    ) -> Union[torch.Tensor, None]:
        if _native:
            return nn.MultiheadAttention.merge_masks(self, attention_mask, key_padding_mask, query)
        if key_padding_mask is not None:
            b, n_q, _ = query.shape
            key_padding_mask = key_padding_mask.view(b, 1, -1).expand(b, n_q, -1)
        if attention_mask is None:
            return key_padding_mask
        if key_padding_mask is None:
            return attention_mask
        return attention_mask | key_padding_mask

    @property
    def batch_first(self) -> bool:
        return True

    @property
    def _qkv_same_embed_dim(self) -> bool:
        return True


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    Override the standard transformer encoder layer to use our own multihead attention
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: Optional[bool] = None,
        norm_first: bool = False,
        bias: bool = True,
        score_mod: Optional[Callable] = None,
        block_mask = None,
        dim_head: Optional[int] = None,
        device=None,
        dtype=None,
        _compile: bool = False
    ):
        if batch_first is None:
            warnings.warn("batch_first is not set, defaulting to True", UserWarning)
        assert batch_first, "batch_first must be True"
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
            **factory_kwargs,)

        self.self_attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            # dropout=dropout, not currently supported
            bias=bias,
            score_mod=score_mod,
            block_mask=block_mask,
            head_embed_dim=dim_head,
            **factory_kwargs,
            _compile=_compile
        )

# Deprecated layers after this point...

@export
@deprecated("")
class MultiHeadAttention(L.LightningModule):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        head_embed_dim: Optional[int] = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias
        if head_embed_dim is None:
            assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads if head_embed_dim is not provided"
            head_embed_dim = embed_dim // num_heads
        self.head_embed_dim = head_embed_dim

        # Create projection layers
        self.w_query = nn.Linear(embed_dim, self.head_embed_dim * num_heads, bias=bias)
        self.w_key = nn.Linear(embed_dim, self.head_embed_dim * num_heads, bias=bias)
        self.w_value = nn.Linear(embed_dim, self.head_embed_dim * num_heads, bias=bias)
        self.w_output = nn.Linear(self.head_embed_dim * num_heads, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attention_head_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        *extra_dims_q, n_q, _ = query.size()
        *extra_dims_k, n_k, _ = key.size()
        *extra_dims_v, _, _ = value.size()

        # Linear projections
        q = self.w_query(query).view((*extra_dims_q, n_q, self.num_heads, self.head_embed_dim)).transpose(-2, -3) # [..., h, n_q, d_k]
        k = self.w_key(key).view((*extra_dims_k, n_k, self.num_heads, self.head_embed_dim)).transpose(-2, -3) # [..., h, n_k, d_k]
        v = self.w_value(value).view((*extra_dims_v, n_k, self.num_heads, self.head_embed_dim)).transpose(-2, -3) # [..., h, n_k, d_k]

        # Compute attention weights
        attention_weights = self.compute_attention_weights(
            q,
            k,
            self.merge_mask(attention_mask, key_padding_mask),
            attention_head_mask)
        attention_weights = self.dropout(attention_weights)

        attention = torch.matmul(attention_weights, v) # [..., h, n_q, d_k]
        attention = attention.transpose(-2, -3).reshape((*extra_dims_v, n_q, self.num_heads*self.head_embed_dim)) # [..., n_q, d_k*num_heads]
        output = self.w_output(attention)

        if return_attention_weights:
            if average_attention_weights:
                n = attention_head_mask.sum() if attention_head_mask is not None else attention_weights.size(-1)
                attention_weights = attention_weights.sum(dim=-3) / n
            return output, attention_weights
        return output

    def merge_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor]
    ) -> Union[torch.Tensor, None]:
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(-2)
        if attention_mask is None:
            return key_padding_mask
        if key_padding_mask is None:
            return attention_mask
        return attention_mask | key_padding_mask

    def compute_attention_weights(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        attention_head_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        attention_weights = torch.matmul(query, key.transpose(-2, -1))/np.sqrt(self.head_embed_dim)
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(attention_mask.unsqueeze(-3), float("-inf"))
        attention_weights = F.softmax(attention_weights, dim=-1) # [..., h, n_q, n_k]
        if attention_head_mask is not None:
            attention_weights = attention_weights * attention_head_mask.view((-1, 1, 1))
        return attention_weights

    def __len__(self):
        return self.num_heads


@export
@deprecated("")
class RelativeMultiHeadAttention(MultiHeadAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_length: int,
        dropout: float = 0.0,
        bias: bool = True,
        head_embed_dim: Optional[int] = None
    ):
        super().__init__(embed_dim, num_heads, dropout, bias, head_embed_dim)
        self.max_length = max_length
        self.pos_embeddings = nn.Parameter(torch.randn(self.head_embed_dim, 2*max_length - 1)) # (dxn)

    def _skew(self, x: torch.Tensor):
        """
        Memory-efficient skew operation.
        """
        n = x.shape[-1] - x.shape[-1]//2
        x = F.pad(x, (0, 1))
        skewed = x.flatten(-2).narrow(-1, 0, x.shape[-2]*(x.shape[-1] - 1)).view((*x.shape[:-1], -1))
        rel = skewed.narrow(-1, n - 1, n)
        return rel

    def compute_attention_weights(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        attention_head_mask: Optional[torch.Tensor]
    ):
        # Get required position embeddings
        n = key.shape[-2]
        pos_embeddings = F.pad(self.pos_embeddings, (n - self.max_length, n - self.max_length), mode="replicate")
        att_qk = torch.matmul(query, key.transpose(-2, -1))
        att_qrel = self._skew(torch.matmul(query, pos_embeddings))
        attention_weights = (att_qk + att_qrel) / np.sqrt(self.head_embed_dim)
        # att_qrel = self._skew(torch.matmul(query, self.pos_embeddings), key.shape[-2])
        # att_krel = self._skew(torch.matmul(key, self.pos_embeddings.flip(-1)), query.shape[-2]).transpose(-1, -2)
        # attention_weights = (att_qk + att_qrel + att_krel) / np.sqrt(self.head_embed_dim)
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(attention_mask.unsqueeze(-3), float("-inf"))
        attention_weights = F.softmax(attention_weights, dim=-1) # [..., h, n_q, n_k]
        if attention_head_mask is not None:
            attention_weights = attention_weights * attention_head_mask.view((-1, 1, 1))
        return attention_weights

# Transformer Generics -----------------------------------------------------------------------------

@export
@deprecated("")
class MultiHeadAttentionBlock(L.LightningModule):
    def __init__(
        self,
        mha: MultiHeadAttention,
        feedforward_dim: int,
        feedforward_activation: Union[L.LightningModule, Callable] = F.gelu,
        norm_first: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.mha = mha
        self.feedforward_dim = feedforward_dim
        self.feedforward_activation = feedforward_activation
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(mha.embed_dim)
        self.norm2 = nn.LayerNorm(mha.embed_dim)
        self.feedforward_linear1 = nn.Linear(mha.embed_dim, feedforward_dim)
        self.feedforward_linear2 = nn.Linear(feedforward_dim, mha.embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _cross_attention_block(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        attention_head_mask: Optional[torch.Tensor],
        average_attention_weights: bool,
        return_attention_weights: bool
    ):
        attention_output = self.mha(
            x,
            y,
            y,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            attention_head_mask=attention_head_mask,
            average_attention_weights=average_attention_weights,
            return_attention_weights=return_attention_weights)
        if isinstance(attention_output, tuple):
            attention_output, *extra = attention_output
        else:
            extra = None
        return self.dropout1(attention_output), extra

    def _feedforward_block(self, x: torch.Tensor):
        return self.dropout2(
            self.feedforward_linear2(self.feedforward_activation(self.feedforward_linear1(x)))
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attention_head_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False
    ):
        if self.norm_first:
            if x is y:
                x_norm = y_norm = self.norm1(x)
            else:
                x_norm = self.norm1(x)
                y_norm = self.norm1(y)
            attention_output, extra_output = self._cross_attention_block(
                x_norm,
                y_norm,
                attention_mask,
                key_padding_mask,
                attention_head_mask,
                average_attention_weights,
                return_attention_weights)
            x = x + attention_output
            x = x + self._feedforward_block(self.norm2(x))
        else:
            attention_output, extra_output = self._cross_attention_block(
                x,
                y,
                attention_mask,
                key_padding_mask,
                attention_head_mask,
                average_attention_weights,
                return_attention_weights)
            x = self.norm1(x + attention_output)
            x = self.norm2(x + self._feedforward_block(x))
            return x, extra_output
        if extra_output is not None:
            return x, *extra_output
        return x

    @property
    def embed_dim(self):
        return self.mha.embed_dim

# Transformer Encoders -----------------------------------------------------------------------------

@deprecated("")
class ITransformerEncoder(abc.ABC, L.LightningModule):
    """
    The interface for a transformer encoder block.
    """
    @abc.abstractmethod
    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        attention_head_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False
    ) -> Any:
        return NotImplemented

    @property
    @abc.abstractmethod
    def embed_dim(self) -> int:
        return NotImplemented

@export
@deprecated("")
class TransformerEncoderBlock(ITransformerEncoder, L.LightningModule):
    def __init__(
        self,
        mha: MultiHeadAttention,
        feedforward_dim: int,
        feedforward_activation: Union[L.LightningModule, Callable] = F.gelu,
        norm_first: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.mab = MultiHeadAttentionBlock(
            mha=mha,
            feedforward_dim=feedforward_dim,
            feedforward_activation=feedforward_activation,
            norm_first=norm_first,
            dropout=dropout)
        self.attention_head_mask = nn.Parameter(torch.ones(mha.num_heads), requires_grad=False)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        attention_head_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False
    ):
        if attention_head_mask is None:
            attention_head_mask = self.attention_head_mask
        return self.mab.forward(
            x=src,
            y=src,
            attention_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            attention_head_mask=attention_head_mask,
            average_attention_weights=average_attention_weights,
            return_attention_weights=return_attention_weights)

    @property
    def embed_dim(self) -> int:
        return self.mab.embed_dim

    @property
    def num_heads(self) -> int:
        return self.mab.mha.num_heads


@export
@deprecated("")
class InducedSetAttentionBlock(ITransformerEncoder, L.LightningModule):
    def __init__(
        self,
        mha: MultiHeadAttention,
        num_inducing_points: int,
        feedforward_dim: int,
        feedforward_activation: Union[L.LightningModule, Callable] = F.gelu,
        norm_first: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_inducing_points = num_inducing_points
        self.inducing_points = nn.Parameter(torch.randn(num_inducing_points, mha.embed_dim))
        self.mab1 = MultiHeadAttentionBlock(copy.deepcopy(mha), feedforward_dim, feedforward_activation, norm_first, dropout)
        self.mab2 = MultiHeadAttentionBlock(copy.deepcopy(mha), feedforward_dim, feedforward_activation, norm_first, dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        attention_head_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False
    ):
        if src_mask is not None:
            raise Exception("Attention mask not supported for InducedSetAttentionBlock.")
        i = self.inducing_points.unsqueeze(-3).expand(*src.shape[:-2], -1, -1)
        h = self.mab1(i, src, key_padding_mask=src_key_padding_mask)
        return self.mab2(
            src,
            h,
            attention_head_mask=attention_head_mask,
            average_attention_weights=average_attention_weights,
            return_attention_weights=return_attention_weights)

    @property
    def embed_dim(self):
        return self.mab1.mha.embed_dim

    @property
    def num_heads(self):
        return self.mab1.mha.num_heads

@export
@deprecated("")
class TransformerEncoder(ITransformerEncoder, L.LightningModule):
    def __init__(self, encoder_layer: ITransformerEncoder, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        attention_head_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False
    ):
        extra_outputs = []
        output = src
        head_mask = None
        for i, layer in enumerate(cast(Sequence[ITransformerEncoder], self.layers)):
            if attention_head_mask is not None:
                head_mask = attention_head_mask[i]
            output = layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                attention_head_mask=head_mask,
                average_attention_weights=average_attention_weights,
                return_attention_weights=return_attention_weights)
            if isinstance(output, tuple):
                output, *extra_output = output
                if len(extra_output) > 0:
                    extra_outputs.append(extra_output)
        if len(extra_outputs) > 0:
            return output, *zip(*extra_outputs)
        return output

    @property
    def attention_head_mask(self):
        return torch.stack([layer.attention_head_mask for layer in self.layers])

    @attention_head_mask.setter
    def attention_head_mask(self, attention_head_mask):
        for layer, mask in zip(self.layers, attention_head_mask):
            layer.attention_head_mask[:] = mask

    @property
    def embed_dim(self):
        return self.layers[0].embed_dim

    @property
    def num_heads(self):
        return self.layers[0].num_heads

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, index):
        return self.layers[index]

# Transformer Decoders -----------------------------------------------------------------------------

@deprecated("")
class ITransformerDecoder(abc.ABC):
    """
    The interface for a transformer encoder block.
    """
    @abc.abstractmethod
    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        target_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False
    ):
        return NotImplemented


@export
@deprecated("")
class TransformerDecoderBlock(ITransformerDecoder, L.LightningModule):

    def __init__(
        self,
        mha: MultiHeadAttention,
        feedforward_dim: int,
        feedforward_activation: Union[L.LightningModule, Callable] = F.gelu,
        norm_first: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_mha = copy.deepcopy(mha)
        self.cross_mha = copy.deepcopy(mha)
        self.feedforward_dim = feedforward_dim
        self.feedforward_activation = feedforward_activation
        self.norm_first = norm_first

        self.norm1 = nn.LayerNorm(self.self_mha.embed_dim)
        self.norm2 = nn.LayerNorm(self.self_mha.embed_dim)
        self.norm3 = nn.LayerNorm(self.self_mha.embed_dim)

        self.feedforward_linear1 = nn.Linear(self.self_mha.embed_dim, feedforward_dim)
        self.feedforward_linear2 = nn.Linear(feedforward_dim, self.self_mha.embed_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def _self_attention_block(
        self,
        target: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        average_attention_weights: bool,
        return_attention_weights: bool
    ) -> Tuple[torch.Tensor, Optional[List[Any]]]:
        attention_output = self.self_mha(
            target,
            target,
            target,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            average_attention_weights=average_attention_weights,
            return_attention_weights=return_attention_weights)
        if isinstance(attention_output, tuple):
            attention_output, *extra = attention_output
        else:
            extra = None
        return self.dropout1(attention_output), extra

    def _cross_attention_block(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        average_attention_weights: bool,
        return_attention_weights: bool
    ) -> Tuple[torch.Tensor, Optional[List[Any]]]:
        attention_output = self.cross_mha(
            target,
            memory,
            memory,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            average_attention_weights=average_attention_weights,
            return_attention_weights=return_attention_weights)
        if isinstance(attention_output, tuple):
            attention_output, *extra = attention_output
        else:
            extra = None
        return self.dropout2(attention_output), extra

    def _feedforward_block(self, x: torch.Tensor):
        return self.dropout3(
            self.feedforward_linear2(self.feedforward_activation(self.feedforward_linear1(x)))
        )

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        target_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False
    ):
        x = target
        if self.norm_first:
            attention_output, extra_output_sa = self._self_attention_block(
                self.norm1(x),
                target_mask,
                target_key_padding_mask,
                average_attention_weights,
                return_attention_weights)
            x = x + attention_output
            attention_output, extra_output_ca = self._cross_attention_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                average_attention_weights,
                return_attention_weights)
            x = x + attention_output
            x = x + self._feedforward_block(self.norm3(x))
        else:
            attention_output, extra_output_sa = self._self_attention_block(
                x,
                target_mask,
                target_key_padding_mask,
                average_attention_weights,
                return_attention_weights)
            x = self.norm1(x + attention_output)
            attention_output, extra_output_ca = self._cross_attention_block(
                x,
                memory,
                memory_mask,
                memory_key_padding_mask,
                average_attention_weights,
                return_attention_weights)
            x = self.norm2(x + attention_output)
            x = self.norm3(x + self._feedforward_block(x))
        extra_output = ()
        if extra_output_sa is not None:
            extra_output += tuple(extra_output_sa)
        if extra_output_ca is not None:
            extra_output += tuple(extra_output_ca)
        if len(extra_output) > 0:
            return x, *extra_output
        return x


@export
@deprecated("")
class TransformerDecoder(ITransformerDecoder, L.LightningModule):
    def __init__(self, decoder_layer: TransformerDecoderBlock, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        target_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False
    ):
        extra_outputs = []
        output = target
        for layer in cast(Sequence[TransformerDecoderBlock], self.layers):
            output = layer(
                target=output,
                memory=memory,
                target_mask=target_mask,
                memory_mask=memory_mask,
                target_key_padding_mask=target_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                average_attention_weights=average_attention_weights,
                return_attention_weights=return_attention_weights)
            if isinstance(output, tuple):
                output, *extra_output = output
                if len(extra_output) > 0:
                    extra_outputs.append(extra_output)
        if len(extra_outputs) > 0:
            return output, *zip(*extra_outputs)
        return output

    @property
    def embed_dim(self):
        return self.layers[0].embed_dim

    def __len__(self):
        return len(self.layers)


@export
@deprecated("")
class ConditionedInducedSetAttentionBlock(ITransformerDecoder, L.LightningModule):
    def __init__(
        self,
        mha: MultiHeadAttention,
        num_inducing_points: int,
        feedforward_dim: int,
        feedforward_activation: Union[L.LightningModule, Callable] = F.gelu,
        norm_first: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_inducing_points = num_inducing_points
        self.inducing_point_predictor = nn.Linear(mha.embed_dim, mha.embed_dim*num_inducing_points)
        self.mab1 = MultiHeadAttentionBlock(copy.deepcopy(mha), feedforward_dim, feedforward_activation, norm_first, dropout)
        self.mab2 = MultiHeadAttentionBlock(copy.deepcopy(mha), feedforward_dim, feedforward_activation, norm_first, dropout)

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        target_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False
    ):
        if target_mask is not None:
            raise Exception(f"target_mask not supported for {self.__class__}")
        if memory_mask is not None:
            raise Exception(f"memory_mask not supported for {self.__class__}")
        if memory_key_padding_mask is not None:
            raise Exception(f"Memory key padding mask not supported for {self.__class__}")
        i = self.inducing_point_predictor(memory).view(*memory.shape[:-1], self.num_inducing_points, -1)
        h = self.mab1(i, target, key_padding_mask=target_key_padding_mask)
        return self.mab2(target, h, average_attention_weights=average_attention_weights, return_attention_weights=return_attention_weights)

    @property
    def embed_dim(self):
        return self.mab1.embed_dim

# Bidirectional Mamba3 Mechanism ------------------------------------------------------------------

class BiMamba3Block(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        d_state: int = 128,
        headdim: int = 64,
        expand: int = 2,
        chunk_size: int = 64,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        bidirectional: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        mamba_kwargs = dict(
            d_model=embed_dim,
            d_state=d_state,
            headdim=headdim,
            expand=expand,
            chunk_size=chunk_size,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank
        )

        self.bidirectional = bidirectional
        self.norm = nn.LayerNorm(embed_dim)
        self.forward_mamba = Mamba3(**mamba_kwargs)
        self.backward_mamba = Mamba3(**mamba_kwargs) if bidirectional else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, **kwargs):
        # kwargs absorbs the transformer-style arguments (src_mask, etc.) so
        # this block is substitutable wherever an encoder block is expected.
        x = self.norm(src)
        output = self.forward_mamba(x)
        if self.backward_mamba is not None:
            output = output + self.backward_mamba(x.flip(-2)).flip(-2)
        return src + self.dropout(output)

    @property
    def embed_dim(self) -> int:
        return self.forward_mamba.d_model

    @property
    def is_mimo(self) -> bool:
        return self.forward_mamba.is_mimo

    @property
    def chunk_size(self) -> int:
        return self.forward_mamba.chunk_size


class Mamba3Encoder(nn.Module):

    def __init__(
        self,
        encoder_layer: BiMamba3Block,
        num_layers: int,
        final_layer_norm: bool = False
    ):
        super().__init__()
        self.is_mimo = encoder_layer.is_mimo
        self.chunk_size = encoder_layer.chunk_size
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(encoder_layer.embed_dim) if final_layer_norm else nn.Identity()

    def forward(self, src: torch.Tensor, **kwargs):
        seq_length = src.shape[-2]
        if self.is_mimo:
            pad_size = (-seq_length) % self.chunk_size
            if pad_size != 0:
                src = F.pad(src, (0, 0, 0, pad_size))

        output = src
        for layer in self.layers:
            output = layer(output)

        if output.shape[-2] != seq_length:
            output = output[:, :seq_length, :]

        return self.norm(output)

    @property
    def embed_dim(self) -> int:
        return self.layers[0].embed_dim

    def __len__(self) -> int:
        return len(self.layers)