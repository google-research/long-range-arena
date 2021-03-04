# Copyright 2021 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implements Sparse Transformer's attention pattern.

(https://arxiv.org/pdf/1904.10509.pdf).

Note that all attention patterns are causal.
"""
import functools
from typing import Iterable

import attr
from flax import nn
from jax import lax
import jax.numpy as jnp
import numpy as np


@attr.s
class _Pattern(object):
  pass


@attr.s
class AllPattern(_Pattern):
  pass


@attr.s
class LocalPattern(_Pattern):
  bandwidth = attr.ib()


@attr.s
class StridedPattern(_Pattern):
  stride = attr.ib()


@attr.s
class Fixed1Pattern(_Pattern):
  """Corresponds to the first of two heads in the fixed scheme."""
  block_size = attr.ib()


@attr.s
class Fixed2Pattern(_Pattern):
  """Corresponds to the second of two heads in the fixed scheme."""
  block_size = attr.ib()
  c = attr.ib()


def build_mask(seq_len: int, patterns: Iterable[_Pattern]):
  """Merges multiple attention mask patterns into one."""
  merged_mask = functools.reduce(
      np.logical_or,
      (_build_mask(seq_len, pattern) for pattern in patterns))
  return jnp.array(merged_mask).astype(jnp.bool_)


def _build_mask(n: int, pattern: _Pattern) -> np.ndarray:
  """Helper to build sparse masks."""
  if isinstance(pattern, AllPattern):
    mask = np.tri(n, k=0)
  elif isinstance(pattern, LocalPattern):
    ctx = min(n - 1, pattern.bandwidth - 1)
    mask = sum(np.eye(n, k=-i) for i in range(ctx + 1))
  else:
    r = np.arange(n)
    q = r[:, np.newaxis]
    k = r[np.newaxis, :]
    lower_triangular = k <= q
    if isinstance(pattern, StridedPattern):
      mask = np.remainder(q - k, pattern.stride) == 0
    elif isinstance(pattern, Fixed1Pattern):
      mask = np.floor_divide(q, pattern.block_size) == np.floor_divide(
          k, pattern.block_size)
    elif isinstance(pattern, Fixed2Pattern):
      remainder = np.remainder(k, pattern.block_size)
      mask = np.logical_or(remainder == 0,
                           remainder >= pattern.block_size - pattern.c)
    else:
      raise ValueError('Attention Pattern {} not supported.'.format(pattern))
    mask = np.logical_and(lower_triangular, mask)
  return np.reshape(mask, [1, 1, n, n])


class SparseAttention(nn.Module):
  """Module implementing Sparse Transformer's attention."""

  def apply(self,
            inputs_q,
            inputs_kv,
            num_heads,
            attention_patterns=None,
            dtype=jnp.float32,
            qkv_features=None,
            out_features=None,
            padding_mask=None,
            key_padding_mask=None,
            segmentation=None,
            key_segmentation=None,
            broadcast_dropout=True,
            dropout_rng=None,
            dropout_rate=0.,
            deterministic=False,
            precision=None,
            kernel_init=nn.linear.default_kernel_init,
            bias_init=nn.initializers.zeros,
            bias=True,
            use_cls_token=False):
    """Applies sparse multi-head dot product attention on the input data.

    Args:
      inputs_q: input queries of shape `[bs, seq_len, features]`.
      inputs_kv: key/values of shape `[bs, seq_len, features]` or `None` for
        self-attention, in which case key/values will be derived from inputs_q.
      num_heads: number of attention heads (should divide number of features).
      attention_patterns: list of `_Pattern` objects representing the sparse
        `None`, we use the merged, fixed attention pattern used in the paper for
        EnWik8.
      dtype: the dtype of the computation (default: float32).
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection.
      padding_mask: boolean specifying query tokens that are pad token.
      key_padding_mask: boolean specifying key-value tokens that are pad token.
      segmentation: segment indices for packed inputs_q data.
      key_segmentation: segment indices for packed inputs_kv data.
      broadcast_dropout: use a broadcasted dropout along batch dims.
      dropout_rng: JAX PRNGKey to be use for dropout.
      dropout_rate: dropout rate.
      deterministic: if true, apply dropout, else don't.
      precision: numerical precision of the computation.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      bias: whether pointwise QKVO dense transforms use bias. query, key, value,
        and returns output of shape `[bs, seq_len, num_heads, value_channels]`.
      use_cls_token: boolean

    Returns:
      output of shape `[bs, seq_len, features]`.
    """
    if inputs_kv is None:
      inputs_kv = inputs_q

    attention_axis = (1,)

    features = out_features or inputs_q.shape[-1]
    qkv_features = qkv_features or inputs_q.shape[-1]
    seq_len = inputs_q.shape[attention_axis[0]]

    assert qkv_features % num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // num_heads

    dense = nn.DenseGeneral.partial(
        axis=-1,
        features=(num_heads, head_dim),
        kernel_init=kernel_init,
        bias_init=bias_init,
        bias=bias,
        precision=precision)

    query, key, value = (dense(inputs_q, dtype=dtype, name='query'),
                         dense(inputs_kv, dtype=dtype, name='key'),
                         dense(inputs_kv, dtype=dtype, name='value'))

    if attention_patterns is None:
      # This is the merged fixed attention pattern used in the paper for EnWik8.
      attention_patterns = [
          Fixed1Pattern(block_size=128),
          Fixed2Pattern(block_size=128, c=32)
      ]

    if use_cls_token:
      # don't mask cls token
      # reset all attention bias to 0 for first position
      mask_seq_len = seq_len - 1
      sparse_mask = build_mask(mask_seq_len, attention_patterns)
      pad_first = jnp.array([[0, 0], [0, 0], [1, 0], [1, 0]])
      sparse_mask = jnp.pad(sparse_mask, pad_first, constant_values=1.0)

    else:
      mask_seq_len = seq_len
      sparse_mask = build_mask(mask_seq_len, attention_patterns)

    mask_components = [
        sparse_mask
    ]

    if padding_mask is not None:
      if key_padding_mask is None:
        key_padding_mask = padding_mask
      padding_mask = nn.attention.make_padding_mask(
          padding_mask_query=padding_mask,
          padding_mask_key=key_padding_mask,
          query_shape=query.shape,
          key_shape=key.shape,
          attention_axis=attention_axis)
      mask_components.append(padding_mask)

    if segmentation is not None:
      if key_segmentation is None:
        key_segmentation = segmentation
      segmentation_mask = nn.attention.make_padding_mask(
          padding_mask_query=segmentation,
          padding_mask_key=key_segmentation,
          query_shape=query.shape,
          key_shape=key.shape,
          attention_axis=attention_axis,
          segmentation_mask=True)
      mask_components.append(segmentation_mask)

    if mask_components:
      attention_mask = mask_components[0]
      for component in mask_components[1:]:
        attention_mask = jnp.logical_and(attention_mask, component)

      # attention mask in the form of attention bias
      attention_bias = lax.select(
          attention_mask > 0,
          jnp.full(attention_mask.shape, 0.).astype(dtype),
          jnp.full(attention_mask.shape, -1e10).astype(dtype))
    else:
      attention_bias = None

    # apply attention
    x = nn.attention.dot_product_attention(
        query,
        key,
        value,
        dtype=dtype,
        axis=attention_axis,
        bias=attention_bias,
        precision=precision,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        broadcast_dropout=broadcast_dropout,
        deterministic=deterministic)

    # back to the original inputs dimensions
    out = nn.DenseGeneral(
        x,
        features=features,
        axis=(-2, -1),
        kernel_init=kernel_init,
        bias_init=bias_init,
        bias=bias,
        dtype=dtype,
        precision=precision,
        name='out')

    return out


SparseSelfAttention = SparseAttention.partial(inputs_kv=None)
