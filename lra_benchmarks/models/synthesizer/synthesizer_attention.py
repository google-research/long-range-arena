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
"""Synthesizer Attention modules."""

from collections.abc import Iterable  # pylint: disable=g-importing-member

from absl import logging

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


def synthetic_attention(query,
                        key,
                        value,
                        synthetic,
                        dtype=jnp.float32,
                        bias=None,
                        axis=None,
                        broadcast_dropout=True,
                        dropout_rng=None,
                        dropout_rate=0.,
                        deterministic=False,
                        precision=None,
                        ignore_dot_product=False):
  """Computes dot-product attention given query, key, and value.

  Supports additional synthetic weights mixture.

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
    synthetic: list of weight matrices of [len, len].
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
    ignore_dot_product: bool, to ignore dot product or not.

  Returns:
    Output of shape `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]`.
  """
  if axis is None:
    axis = tuple(range(1, key.ndim - 2))
  if not isinstance(axis, Iterable):
    axis = (axis,)
  if not ignore_dot_product:
    assert key.shape[:-1] == value.shape[:-1]
    assert (query.shape[0:1] == key.shape[0:1] and
            query.shape[-1] == key.shape[-1])
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
  else:
    n = key.ndim
    batch_dims = tuple(onp.delete(range(n), axis + (n - 1,)))
    v_perm = batch_dims + (n - 1,) + axis
    qk_perm = batch_dims + axis + (n - 1,)
    value = value.transpose(v_perm)
    batch_dims_t = tuple(range(len(batch_dims)))
    attn_weights = 0

  if synthetic:
    # add synthetic attention
    for syn_weights in synthetic:
      attn_weights += syn_weights

  # apply attention bias: masking, droput, proximity bias, ect.
  if bias is not None:
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


class SynthesizerAttention(nn.Module):
  """Multi-head Synthesizer Architecture."""

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
            max_length=512,
            ignore_dot_product=True,
            synthesizer_mode='factorized_random',
            k=32):
    """Applies multi-head synthesizer attention on the input data.

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
      max_length: int, the maximum supported sequence length.
      ignore_dot_product: bool, to ignore the dot product attention or not.
      synthesizer_mode: str support 'dense' and 'random' or 'dense+random'
      k: int, low rank factorized attention.

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
    kvlength = inputs_kv.shape[-2]

    if ignore_dot_product:
      value = dense(inputs_kv, dtype=dtype, name='value')
      key = value
      query = inputs_q
    else:
      query, key, value = (dense(inputs_q, dtype=dtype, name='query'),
                           dense(inputs_kv, dtype=dtype, name='key'),
                           dense(inputs_kv, dtype=dtype, name='value'))

    syn_weights_list = []
    logging.info(synthesizer_mode)
    if 'random' in synthesizer_mode:
      if 'factorized_random' in synthesizer_mode:
        logging.info('Using factorized random')
        rand_syn_weights1 = self.param('random1', (num_heads, max_length, k),
                                       kernel_init)
        rand_syn_weights2 = self.param('random2', (num_heads, k, max_length),
                                       kernel_init)
        rand_syn_weights1 = rand_syn_weights1[:, :qlength, :]
        rand_syn_weights2 = rand_syn_weights2[:, :, :kvlength]
        rand_syn_weights = jnp.einsum('hlk,hkn->hln', rand_syn_weights1,
                                      rand_syn_weights2)
        rand_syn_weights = jax.lax.broadcast(rand_syn_weights,
                                             (inputs_q.shape[0],))
        syn_weights_list.append(rand_syn_weights)
      else:
        rand_syn_weights = self.param('random',
                                      (num_heads, max_length, max_length),
                                      kernel_init)
        rand_syn_weights = rand_syn_weights[:, :qlength, :kvlength]
        rand_syn_weights = jax.lax.broadcast(rand_syn_weights,
                                             (inputs_q.shape[0],))
        syn_weights_list.append(rand_syn_weights)
    if 'dense' in synthesizer_mode:
      dense_syn = nn.DenseGeneral.partial(axis=-1,
                                          features=(num_heads, head_dim),
                                          kernel_init=kernel_init,
                                          bias_init=bias_init,
                                          bias=bias,
                                          precision=precision,
                                          name='dense_syn',
                                          dtype=dtype)
      # TODO(yitay): Change this to nn.Dense and make sure it works
      dense_syn_length = nn.linear.DenseGeneral.partial(axis=-1,
                                                        features=(max_length),
                                                        kernel_init=kernel_init,
                                                        bias_init=bias_init,
                                                        bias=bias,
                                                        precision=precision,
                                                        name='dense_syn2',
                                                        dtype=dtype)
      proj = dense_syn(inputs_q, dtype=dtype, name='dense_syn')
      proj = jax.nn.relu(proj)
      proj = dense_syn_length(proj, dtype=dtype, name='dense_syn_len')
      # TODO(yitay) check if this reshape is needed
      dense_syn_weights = proj.reshape((inputs_q.shape[0], num_heads,
                                        qlength, max_length))
      dense_syn_weights = dense_syn_weights[:, :, :, :qlength]
      syn_weights_list.append(dense_syn_weights)
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

    # create attention masks
    mask_components = []

    if causal_mask:
      if cache and not self.is_initializing():
        bias_pre_shape = (1,) * (key.ndim - 1)
        attn_shape = tuple(onp.take(key.shape, attention_axis))
        attn_size = onp.prod(attn_shape)
        ii = jnp.arange(attn_size, dtype=jnp.uint32)
        mask = ii < cache_entry.i
        mask_components.append(mask.reshape(bias_pre_shape + attn_shape))
      else:
        mask_components.append(_make_causal_mask(key, attention_axis))

    if not ignore_dot_product:
      if padding_mask is not None:
        if key_padding_mask is None:
          key_padding_mask = padding_mask
        padding_mask = make_padding_mask(
            padding_mask_query=padding_mask,
            padding_mask_key=key_padding_mask,
            query_shape=query.shape,
            key_shape=key.shape,
            attention_axis=attention_axis)
        mask_components.append(padding_mask)

      if segmentation is not None:
        if key_segmentation is None:
          key_segmentation = segmentation
        segmentation_mask = make_padding_mask(
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
          attention_mask > 0, jnp.full(attention_mask.shape, 0.).astype(dtype),
          jnp.full(attention_mask.shape, -1e10).astype(dtype))
    else:
      attention_bias = None

    # apply attention
    x = synthetic_attention(
        query,
        key,
        value,
        syn_weights_list,
        dtype=dtype,
        axis=attention_axis,
        bias=attention_bias,
        precision=precision,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        broadcast_dropout=broadcast_dropout,
        deterministic=deterministic,
        ignore_dot_product=ignore_dot_product)

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

# TODO(flax-dev): Consider refactoring MultiHeadDotProductAttention and moving
# causal_mask and cache support into this class instead.
SynthesizerSelfAttention = SynthesizerAttention.partial(inputs_kv=None)
