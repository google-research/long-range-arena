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
"""Custom Attention modules for Linear Transformer."""

from flax import nn
import jax.numpy as jnp


def elu_feature_map(x):
  return nn.elu(x) + 1


def linear_attention(query,
                     key,
                     value,
                     broadcast_dropout=True,
                     dropout_rng=None,
                     dropout_rate=0.,
                     deterministic=False,
                     feature_map=elu_feature_map,
                     eps=1e-6):
  """Computes linear attention given query, key, and value.


  Args:
    query: queries for calculating attention with shape of `[batch_size, len,
      num_heads, mem_channels]`.
    key: keys for calculating attention with shape of `[batch_size, len,
      num_heads, mem_channels]`.
    value: values to be used in attention with shape of `[batch_size, len,
      num_heads, value_channels]`.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    feature_map: function, to map query and key to a new feature space.
    eps: float, used to avoid division by zero.

  Returns:
    Output of shape `[bs, length, num_heads, value_channels]`.
  """
  del broadcast_dropout
  del dropout_rng
  del dropout_rate
  del deterministic
  # TODO(dehghani): figure out how to apply attention dropout!
  assert key.ndim == query.ndim == value.ndim == 4
  assert key.shape[:-1] == value.shape[:-1]
  assert (query.shape[0:1] == key.shape[0:1] and
          query.shape[-1] == key.shape[-1])

  query_mapped = feature_map(query)
  key_mapped = feature_map(key)
  kv = jnp.einsum('nshd,nshm->nhmd', key_mapped, value)

  z = 1 / (
      jnp.einsum('nlhd,nhd->nlh', query_mapped, jnp.sum(key_mapped, axis=1)) +
      eps)
  y = jnp.einsum('nlhd,nhmd,nlh->nlhm', query_mapped, kv, z)

  return y


class LinearAttention(nn.Module):
  """Linear Attention Architecture."""

  def apply(self,
            inputs_q,
            inputs_kv,
            num_heads,
            dtype=jnp.float32,
            qkv_features=None,
            out_features=None,
            causal_mask=False,
            padding_mask=None,
            key_padding_mask=None,
            segmentation=None,
            key_segmentation=None,
            cache=None,
            broadcast_dropout=True,
            dropout_rng=None,
            dropout_rate=0.,
            deterministic=False,
            precision=None,
            kernel_init=nn.linear.default_kernel_init,
            bias_init=nn.initializers.zeros,
            bias=True):
    """Applies linear attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies linear attention and project the results to an output vector.

    Args:
      inputs_q: input queries of shape `[bs, dim1, dim2, ..., dimN, features]`.
      inputs_kv: key/values of shape `[bs, dim1, dim2, ..., dimN, features]` or
        None for self-attention, inn which case key/values will be derived from
        inputs_q.
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation (default: float32)
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      causal_mask: boolean specifying whether to apply a causal mask on the
        attention weights. If True, the output at timestep `t` will not depend
        on inputs at timesteps strictly greater than `t`.
      padding_mask: boolean specifying query tokens that are pad token.
      key_padding_mask: boolean specifying key-value tokens that are pad token.
      segmentation: segment indices for packed inputs_q data.
      key_segmentation: segment indices for packed inputs_kv data.
      cache: an instance of `flax.nn.attention.Cache` used for efficient
        autoregressive decoding.
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rng: JAX PRNGKey: to be used for dropout
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      bias: bool: whether pointwise QKVO dense transforms use bias.

    Returns:
      output of shape `[bs, dim1, dim2, ..., dimN, features]`.
    """

    if padding_mask is not None:
      NotImplementedError(
          'Currently, we do not support autoregresive decoding.')

      assert causal_mask or not cache, (
          'Caching is only support for causal attention.')

    assert inputs_q.ndim == 3

    if inputs_kv is None:
      inputs_kv = inputs_q

    features = out_features or inputs_q.shape[-1]
    qkv_features = qkv_features or inputs_q.shape[-1]

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

    # project inputs_q to multi-headed q/k/v
    # dimensions are then [bs, dims..., n_heads, n_features_per_head]
    query, key, value = (dense(inputs_q, dtype=dtype, name='query'),
                         dense(inputs_kv, dtype=dtype, name='key'),
                         dense(inputs_kv, dtype=dtype, name='value'))

    if cache:
      raise NotImplementedError('Decoding not supported in LinearAttention.')

    # apply regular dot product attention
    x = linear_attention(
        query,
        key,
        value,
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


LinearSelfAttention = LinearAttention.partial(inputs_kv=None)
