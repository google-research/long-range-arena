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
"""Sinkhorn Attention modules."""

from collections.abc import Iterable  # pylint: disable=g-importing-member

from flax import nn
from flax.nn.attention import _CacheEntry
from flax.nn.attention import _make_causal_mask
from flax.nn.attention import Cache
from flax.nn.attention import make_padding_mask
from flax.nn.stochastic import make_rng
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as onp


logsumexp = jax.scipy.special.logsumexp


def sinkhorn_operator(log_alpha, n_iters=1, temp=1.0, clip=1.0, causal=True):
  """sinkhorn operator."""
  n = log_alpha.shape[1]
  log_alpha = jnp.reshape(log_alpha, (-1, n, n)) / temp

  def causal_logsumexp(log_alpha, axis=-1):
    # TODO(yitay) Verify causal sinkhorn ops
    log_alpha = jnp.exp(jnp.clip(log_alpha, -clip, clip))
    mask = _make_causal_mask(log_alpha)
    mask = jnp.reshape(mask(-1, n, n))
    if axis == 1:
      mask = jnp.transpose(mask, (0, 2, 1))
    log_alpha *= (1-mask)  # flip mask
    log_alpha = jnp.sum(log_alpha, axis=axis, keepdims=True)
    log_alpha = jnp.log(log_alpha + 1e-10)
    return log_alpha

  def reduce_logsumexp(log_alpha, axis=1):
    log_alpha = logsumexp(log_alpha, axis=axis, keepdims=True)
    return log_alpha

  for _ in range(n_iters):
    if causal:
      log_alpha -= causal_logsumexp(log_alpha, axis=2)
      log_alpha -= causal_logsumexp(log_alpha, axis=1)
    else:
      log_alpha -= jnp.reshape(reduce_logsumexp(log_alpha, axis=2), (-1, n, 1))
      log_alpha -= jnp.reshape(reduce_logsumexp(log_alpha, axis=1), (-1, 1, n))

  log_alpha = jnp.clip(log_alpha, -clip, clip)
  return jnp.exp(log_alpha)


def local_dot_product_attention(query,
                                key,
                                value,
                                dtype=jnp.float32,
                                bias=None,
                                axis=None,
                                broadcast_dropout=True,
                                dropout_rng=None,
                                dropout_rate=0.,
                                deterministic=False,
                                precision=None):
  """Computes dot-product attention given query, key, and value.

  Note: This is equivalent to the dot product attention in flax.nn.
  However, we do extra broadcasting of the bias in this function.
  I'm leaving this here incase we need to modify something later.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights. This
  function supports multi-dimensional inputs.


  Args:
    query: queries for calculating attention with shape of `[batch_size, dim1,
      dim2, ..., dimN, num_heads, mem_channels]`.
    key: keys for calculating attention with shape of `[batch_size, dim1, dim2,
      ..., dimN, num_heads, mem_channels]`.
    value: values to be used in attention with shape of `[batch_size, dim1,
      dim2,..., dimN, num_heads, value_channels]`.
    dtype: the dtype of the computation (default: float32)
    bias: bias for the attention weights. This can be used for incorporating
      autoregressive mask, padding mask, proximity bias.
    axis: axises over which the attention is applied.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    Output of shape `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]`.
  """
  assert key.shape[:-1] == value.shape[:-1]
  assert (query.shape[0:1] == key.shape[0:1] and
          query.shape[-1] == key.shape[-1])

  if axis is None:
    axis = tuple(range(1, key.ndim - 2))
  if not isinstance(axis, Iterable):
    axis = (axis,)
  assert key.ndim == query.ndim
  assert key.ndim == value.ndim
  for ax in axis:
    if not (query.ndim >= 3 and 1 <= ax < query.ndim - 2):
      raise ValueError('Attention axis must be between the batch '
                       'axis and the last-two axes.')
  depth = query.shape[-1]
  n = key.ndim
  # batch_dims is  <bs, <non-attention dims>, num_heads>
  batch_dims = tuple(onp.delete(range(n), axis + (n - 1,)))
  # q & k -> (bs, <non-attention dims>, num_heads, <attention dims>, channels)
  qk_perm = batch_dims + axis + (n - 1,)
  key = key.transpose(qk_perm)
  query = query.transpose(qk_perm)
  # v -> (bs, <non-attention dims>, num_heads, channels, <attention dims>)
  v_perm = batch_dims + (n - 1,) + axis
  value = value.transpose(v_perm)

  query = query / jnp.sqrt(depth).astype(dtype)
  batch_dims_t = tuple(range(len(batch_dims)))
  attn_weights = lax.dot_general(
      query,
      key, (((n - 1,), (n - 1,)), (batch_dims_t, batch_dims_t)),
      precision=precision)

  # apply attention bias: masking, droput, proximity bias, ect.
  if bias is not None:
    bias = bias[:, :, None, :, :]
    attn_weights = attn_weights + bias

  # normalize the attention weights
  norm_dims = tuple(range(attn_weights.ndim - len(axis), attn_weights.ndim))
  attn_weights = jax.nn.softmax(attn_weights, axis=norm_dims)
  attn_weights = attn_weights.astype(dtype)

  # apply dropout
  if not deterministic and dropout_rate > 0.:
    if dropout_rng is None:
      dropout_rng = make_rng()
    keep_prob = jax.lax.tie_in(attn_weights, 1.0 - dropout_rate)
    if broadcast_dropout:
      # dropout is broadcast across the batch+head+non-attention dimension
      dropout_dims = attn_weights.shape[-(2 * len(axis)):]
      dropout_shape = (tuple([1] * len(batch_dims_t)) + dropout_dims)
      keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    else:
      keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
    multiplier = (keep.astype(attn_weights.dtype) /
                  jnp.asarray(keep_prob, dtype=dtype))
    attn_weights = attn_weights * multiplier

  # compute the new values given the attention weights
  wv_contracting_dims = (norm_dims, range(value.ndim - len(axis), value.ndim))
  y = lax.dot_general(
      attn_weights,
      value, (wv_contracting_dims, (batch_dims_t, batch_dims_t)),
      precision=precision)

  # back to (bs, dim1, dim2, ..., dimN, num_heads, channels)
  perm_inv = _invert_perm(qk_perm)
  y = y.transpose(perm_inv)
  return y


def _invert_perm(perm):
  perm_inv = [0] * len(perm)
  for i, j in enumerate(perm):
    perm_inv[j] = i
  return tuple(perm_inv)


class SinkhornAttention(nn.Module):
  """Multi-head Sinkhorn Attention Architecture."""

  def apply(self,
            inputs_q,
            inputs_kv,
            num_heads,
            dtype=jnp.float32,
            qkv_features=None,
            out_features=None,
            attention_axis=None,
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
            bias=True,
            block_size=50,
            max_num_blocks=25,
            sort_activation='softmax'):
    """Applies multi-head sinkhorn attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    This can be used for encoder-decoder attention by specifying both `inputs_q`
    and `inputs_kv` orfor self-attention by only specifying `inputs_q` and
    setting `inputs_kv` to None.

    Args:
      inputs_q: input queries of shape `[bs, dim1, dim2, ..., dimN, features]`.
      inputs_kv: key/values of shape `[bs, dim1, dim2, ..., dimN, features]`
        or None for self-attention, inn which case key/values will be derived
        from inputs_q.
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation (default: float32)
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      attention_axis: axes over which the attention is applied ( 'None' means
        attention over all axes, but batch, heads, and features).
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
      block_size: int, block size.
      max_num_blocks:  int, max num blocks.
      sort_activation: str {softmax, sinkhorn, gumbel_sinkhorn}

    Returns:
      output of shape `[bs, dim1, dim2, ..., dimN, features]`.
    """

    assert causal_mask or not cache, (
        'Caching is only support for causal attention.')

    assert inputs_q.ndim == 3

    if inputs_kv is None:
      inputs_kv = inputs_q

    if attention_axis is None:
      attention_axis = tuple(range(1, inputs_q.ndim - 1))

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
    qlength = inputs_q.shape[-2]
    bs = inputs_q.shape[0]
    kvlength = inputs_kv.shape[-2]

    query, key, value = (dense(inputs_q, dtype=dtype, name='query'),
                         dense(inputs_kv, dtype=dtype, name='key'),
                         dense(inputs_kv, dtype=dtype, name='value'))

    if cache:
      assert isinstance(cache, Cache), 'cache must be an instance of Cache'
      if self.is_initializing():
        cache.store(onp.array((key.ndim,) + key.shape[-2:], dtype=onp.int32))
      else:
        cache_entry = cache.retrieve(None)
        expected_shape = list(cache_entry.key.shape[:-2])
        for attn_dim in attention_axis:
          expected_shape[attn_dim] = 1
        expected_shape = tuple(expected_shape) + inputs_q.shape[-1:]
        if expected_shape != inputs_q.shape:
          raise ValueError('Invalid shape provided, '
                           'expected shape %s instead got %s.' %
                           (expected_shape, inputs_q.shape))

        if not isinstance(cache_entry, _CacheEntry):
          raise ValueError('Cache is not initialized.')

        cshape = cache_entry.key.shape
        indices = [0] * len(cshape)
        i = cache_entry.i
        attn_size = onp.prod(onp.take(cshape, attention_axis))
        for attn_dim in attention_axis:
          attn_size //= cshape[attn_dim]
          indices[attn_dim] = i // attn_size
          i = i % attn_size

        key = lax.dynamic_update_slice(cache_entry.key, key, indices)
        value = lax.dynamic_update_slice(cache_entry.value, value, indices)
        one = jnp.array(1, jnp.uint32)
        cache_entry = cache_entry.replace(i=cache_entry.i + one,
                                          key=key,
                                          value=value)
        cache.store(cache_entry)

        key_padding_mask = jnp.broadcast_to(
            (jnp.arange(cshape[1]) < cache_entry.i), cshape[:2])
        key_padding_mask = key_padding_mask.astype(jnp.float32)[..., None]

    # block reshape before attention
    num_query_blocks = qlength // block_size
    num_kv_blocks = kvlength // block_size

    block_query = jnp.reshape(
        query, (bs, block_size, num_query_blocks, num_heads, head_dim))
    block_key = jnp.reshape(
        key, (bs, block_size, num_kv_blocks, num_heads, head_dim))
    block_value = jnp.reshape(
        value, (bs, block_size, num_kv_blocks, num_heads, head_dim))

    if causal_mask:
      # causal masking needs to not have blocks with mixed information.
      sum_key = jnp.cumsum(block_key, axis=1)
      sum_key = sum_key[:, 0, :, :, :]  # take first item
    else:
      sum_key = jnp.sum(block_key, axis=1)

    # sort net on head_dim dimensions
    sort_out = nn.DenseGeneral(sum_key, axis=-1,
                               features=(max_num_blocks),
                               kernel_init=kernel_init,
                               bias_init=bias_init,
                               bias=bias,
                               precision=precision)

    # (bs x num_key_blocks x num_heads x num_key_blocks
    sort_out = sort_out[:, :, :, :num_query_blocks]

    # simple softmax sorting first.

    if sort_activation == 'sinkhorn':
      permutation = sinkhorn_operator(
          jnp.reshape(sort_out, (-1, num_kv_blocks, num_query_blocks)),
          causal=causal_mask)
      permutation = jnp.reshape(permutation, (-1, num_kv_blocks, num_heads,
                                              num_query_blocks))
    else:
      if causal_mask:
        block_mask = _make_causal_mask(key, attention_axis)
        sort_out += block_mask
      permutation = jax.nn.softmax(sort_out, axis=-1)

    sorted_key = jnp.einsum('bskhd,bnhl->bsnhd', block_key, permutation)
    sorted_value = jnp.einsum('bskhd,bnhl->bsnhd', block_value, permutation)

    # create attention masks
    mask_components = []
    sorted_mask_components = []

    if causal_mask:
      # TODO(yitay): Test this causal masking.
      if cache and not self.is_initializing():
        bias_pre_shape = (1,) * (key.ndim - 1)
        attn_shape = tuple(onp.take(key.shape, attention_axis))
        attn_size = onp.prod(attn_shape)
        ii = jnp.arange(attn_size, dtype=jnp.uint32)
        mask = ii < cache_entry.i
        mask_components.append(mask.reshape(bias_pre_shape + attn_shape))
      else:
        mask_components.append(_make_causal_mask(key, attention_axis))

    if padding_mask is not None:
      # divide padding mask into block
      padding_mask = jnp.reshape(padding_mask,
                                 (bs * num_query_blocks, block_size, 1))
      if key_padding_mask is None:
        key_padding_mask = padding_mask

      padding_mask = make_padding_mask(
          padding_mask_query=padding_mask,
          padding_mask_key=key_padding_mask,
          query_shape=(bs * num_query_blocks, block_size, num_heads, head_dim),
          key_shape=(bs * num_kv_blocks, block_size, num_heads, head_dim),
          attention_axis=attention_axis)

      padding_mask = jnp.reshape(padding_mask,
                                 (bs, num_query_blocks, block_size, block_size))
      mask_components.append(padding_mask)
      sorted_padding_mask = jnp.einsum('bksj,bnhl->bnsj', padding_mask,
                                       permutation)
      sorted_mask_components.append(sorted_padding_mask)

    if segmentation is not None:
      if key_segmentation is None:
        key_segmentation = segmentation
      segmentation_mask = make_padding_mask(
          padding_mask_query=segmentation,
          padding_mask_key=key_segmentation,
          query_shape=(bs * num_query_blocks, block_size, num_heads, head_dim),
          key_shape=(bs * num_kv_blocks, block_size, num_heads, head_dim),
          attention_axis=attention_axis,
          segmentation_mask=True)
      segmentation_mask = jnp.reshape(segmentation_mask,
                                      (bs, num_query_blocks, block_size,
                                       block_size))
      mask_components.append(segmentation_mask)
      sorted_segmentation_mask = jnp.einsum('bksj,bnhl->bnsj',
                                            segmentation_mask,
                                            permutation)
      sorted_mask_components.append(sorted_segmentation_mask)

    if mask_components:
      attention_mask = mask_components[0]
      for component in mask_components[1:]:
        attention_mask = jnp.logical_and(attention_mask, component)

      # attention mask in the form of attention bias
      attention_bias = lax.select(
          attention_mask > 0, jnp.full(attention_mask.shape, 0.).astype(dtype),
          jnp.full(attention_mask.shape, -1e10).astype(dtype))
    else:
      attention_bias = None

    if sorted_mask_components:
      attention_mask = sorted_mask_components[0]
      for component in sorted_mask_components[1:]:
        attention_mask = jnp.logical_and(attention_mask, component)

      # attention mask in the form of attention bias
      sorted_attention_bias = lax.select(
          attention_mask > 0, jnp.full(attention_mask.shape, 0.).astype(dtype),
          jnp.full(attention_mask.shape, -1e10).astype(dtype))
    else:
      sorted_attention_bias = None

    # apply attention
    x = local_dot_product_attention(
        block_query,
        block_key,
        block_value,
        dtype=dtype,
        axis=attention_axis,
        bias=attention_bias,
        precision=precision,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        broadcast_dropout=broadcast_dropout,
        deterministic=deterministic)

    sorted_x = local_dot_product_attention(
        block_query,
        sorted_key,
        sorted_value,
        dtype=dtype,
        axis=attention_axis,
        bias=sorted_attention_bias,
        precision=precision,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        broadcast_dropout=broadcast_dropout,
        deterministic=deterministic)

    x = x + sorted_x

    x = jnp.reshape(x, (bs, qlength, num_heads, head_dim))

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

SinkhornSelfAttention = SinkhornAttention.partial(inputs_kv=None)
