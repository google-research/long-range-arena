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
"""Attention modules for Reformer model."""

from absl import logging
from flax import nn
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def look_one_back(x):
  """Looks back to previous chunk.

  Args:
    x: input tensor of shape [num_chunks, div_len, dim]
  Returns:
    output tensor of shape [num_chunks, div_len * 2, dim]
  """
  xlb = jnp.concatenate([x[-1:, ...], x[:-1, ...]], axis=0)
  return jnp.concatenate([x, xlb], axis=1)


def permute_via_gather(val, permutation, inverse_permutation, axis=0):
  """Permutation helper for LSH attention."""
  # It is *not* safe to use jax.custom_vjp here. The most likely cause is that
  # it can't close over values: https://github.com/google/jax/issues/2676
  # The error only occurs in some configurations (e.g. use_python_loop = True,
  # num_parallel_heads = 1) but not others.
  permutation = jax.lax.stop_gradient(permutation)
  inverse_permutation = jax.lax.stop_gradient(inverse_permutation)
  def permute_impl(val):
    return jnp.take(val, permutation, axis=axis)
  def permute_vjp(val):
    permuted = permute_impl(jax.lax.stop_gradient(val))
    def vjpfun(permuted_grad):
      # JAX autodiff would synthesize a scatter operation because it doesn't
      # know that the indices are a permutatation. However on TPU, gathers are
      # faster than scatters (at least in the regime the LSH attention uses).
      return (jnp.take(permuted_grad, inverse_permutation, axis=axis),)
    return permuted, vjpfun
  permute = jax.custom_transforms(permute_impl)
  jax.defvjp_all(permute, permute_vjp)
  return permute(val)


def permute_via_sort(val, keys, inverse_keys, axis=0):
  """Permutation helper for LSH attention."""
  # It is *not* safe to use jax.custom_vjp here (see permute_via_gather).
  keys = jax.lax.stop_gradient(keys)
  inverse_keys = jax.lax.stop_gradient(inverse_keys)
  def permute_impl(val):
    # On TPU, sorting scalars by key is faster than a gather.
    _, permuted = jax.lax.sort_key_val(keys, val, dimension=axis)
    return permuted
  def permute_vjp(val):
    permuted = permute_impl(jax.lax.stop_gradient(val))
    def vjpfun(permuted_grad):
      _, val_grad = jax.lax.sort_key_val(
          inverse_keys, permuted_grad, dimension=axis)
      return (val_grad,)
    return permuted, vjpfun
  permute = jax.custom_transforms(permute_impl)
  jax.defvjp_all(permute, permute_vjp)
  return permute(val)


def hash_vectors(vecs, rng, num_buckets, num_hashes):
  """Performs batched hashing.

  Args:
    vecs: input of [length, dim].
    rng: rng object.
    num_buckets: integer, number of buckets.
    num_hashes: integer, number of hashes.
  Returns:
    output of shape [batch_size, length]
  """

  # batch_size = vecs.shape[0]

  assert num_buckets % 2 == 0

  rot_size = num_buckets

  rotations_shape = (vecs.shape[-1], num_hashes, rot_size // 2)

  rng = jax.lax.stop_gradient(jax.lax.tie_in(vecs, rng))
  random_rotations = jax.random.normal(rng, rotations_shape).astype(jnp.float32)

  rotated_vecs = jnp.einsum('tf,fhi->hti', vecs, random_rotations)

  rotated_vecs = jnp.concatenate([rotated_vecs, -rotated_vecs], axis=-1)
  buckets = jnp.argmax(rotated_vecs, axis=-1)
  offsets = jax.lax.tie_in(buckets, jnp.arange(num_hashes))
  offsets = jnp.reshape(offsets * num_buckets, (-1, 1))
  buckets = jnp.reshape(buckets + offsets, (-1,))

  return buckets


def lsh_attention_single_batch(query, value, n_buckets, n_hashes,
                               causal_mask=True):
  """LSH attention for single batch."""
  del causal_mask
  attn = jax.vmap(lsh_attention_single_head, in_axes=(1, 1, None, None))
  out = attn(query, value, n_buckets, n_hashes)
  return out


def length_normalized(x, epsilon=1e-6):
  variance = jnp.mean(x**2, axis=-1, keepdims=True)
  norm_inputs = x / jnp.sqrt(variance + epsilon)
  return norm_inputs


def lsh_attention_single_head(query, value, n_buckets, n_hashes,
                              causal_mask=True,
                              length_norm=False):
  """Applies LSH attention on a single head and a single batch.

  Args:
    query: query tensor of shape [qlength, dims].
    value: value tensor of shape [vlength, dims].
    n_buckets: integer, number of buckets.
    n_hashes: integer, number of hashes.
    causal_mask: boolean, to use causal mask or not.
    length_norm: boolean, to normalize k or not.
  Returns:
    output tensor of shape [qlength, dims]
  """

  qdim, vdim = query.shape[-1], value.shape[-1]
  chunk_size = n_hashes * n_buckets

  seqlen = query.shape[0]

  with nn.stochastic(jax.random.PRNGKey(0)):
    rng = nn.make_rng()

  buckets = hash_vectors(
      query, rng, num_buckets=n_buckets, num_hashes=n_hashes)
  # buckets should be (seq_len)
  assert buckets.shape[-1] == n_hashes * seqlen

  total_hashes = n_hashes

  # create sort and unsort
  ticker = jax.lax.tie_in(query, jnp.arange(n_hashes * seqlen))
  buckets_and_t = seqlen * buckets + (ticker % seqlen)
  buckets_and_t = jax.lax.stop_gradient(buckets_and_t)
  # ticker = jnp.tile(jnp.reshape(ticker, [1, -1]), [batch_size, 1])
  sbuckets_and_t, sticker = jax.lax.sort_key_val(
      buckets_and_t, ticker, dimension=-1)
  _, undo_sort = jax.lax.sort_key_val(sticker, ticker, dimension=-1)
  sbuckets_and_t = jax.lax.stop_gradient(sbuckets_and_t)
  sticker = jax.lax.stop_gradient(sticker)
  undo_sort = jax.lax.stop_gradient(undo_sort)

  st = (sticker % seqlen)

  sqk = jnp.take(query, st, axis=0)
  sv = jnp.take(value, st, axis=0)

  bkv_t = jnp.reshape(st, (chunk_size, -1))
  bqk = jnp.reshape(sqk, (chunk_size, -1, qdim))
  bv = jnp.reshape(sv, (chunk_size, -1, vdim))
  bq = bqk
  bk = bqk

  if length_norm:
    bk = length_normalized(bk)

  # get previous chunks
  bk = look_one_back(bk)
  bv = look_one_back(bv)
  bkv_t = look_one_back(bkv_t)

  # compute dot product attention
  dots = jnp.einsum('hie,hje->hij', bq, bk) * (qdim ** 0.5)

  if causal_mask:
    # apply causal mask
    # TODO(yitay): This is not working yet
    # We don't need causal reformer for any task YET.
    pass

  dots_logsumexp = logsumexp(dots, axis=-1, keepdims=True)
  slogits = jnp.reshape(dots_logsumexp, [-1])
  dots = jnp.exp(dots - dots_logsumexp)

  x = jnp.matmul(dots, bv)
  x = jnp.reshape(x, [-1, qdim])

  # Unsort
  o = permute_via_gather(x, undo_sort, sticker, axis=0)
  logits = permute_via_sort(slogits, sticker, undo_sort, axis=0)
  logits = jnp.reshape(logits, [total_hashes, seqlen, 1])
  probs = jnp.exp(logits - logsumexp(logits, axis=0, keepdims=True))
  o = jnp.reshape(o, [n_hashes, seqlen, qdim])
  out = jnp.sum(o * probs, axis=0)
  out = jnp.reshape(out, [seqlen, qdim])

  return out


class ReformerAttention(nn.Module):
  """Multi-head Reformer Architecture."""

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
            bias=True,
            chunk_len=10,
            n_chunks_before=1,
            n_hashes=1,
            n_buckets=10
            ):
    """Applies multi-head reformer attention on the input data.

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
      chunk_len: int, chunk length.
      n_chunks_before: int, number of chunks before to attend to.
      n_hashes: int, number of hashes.
      n_buckets: int, number of buckets.

    Returns:
      output of shape `[bs, dim1, dim2, ..., dimN, features]`.
    """

    assert causal_mask or not cache, (
        'Caching is only support for causal attention.')

    assert inputs_q.ndim == 3

    if inputs_kv is None:
      inputs_kv = inputs_q

    qkv_features = inputs_q.shape[-1]
    qlength = inputs_q.shape[1]
    orig_seqlen = inputs_q.shape[1]
    batch_size = inputs_q.shape[0]

    # chunk_size = n_hashes * n_buckets

    extra_len = chunk_len - (qlength % chunk_len)
    pad_width = jnp.array([[0, 0], [0, extra_len], [0, 0]])

    inputs_q = jnp.pad(inputs_q, pad_width)
    inputs_kv = jnp.pad(inputs_kv, pad_width)

    logging.info(inputs_q)

    qlength = inputs_q.shape[1]

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
    query, value = (dense(inputs_q, dtype=dtype, name='query'),
                    dense(inputs_kv, dtype=dtype, name='value'))

    attn = jax.vmap(lsh_attention_single_batch, in_axes=(0, 0, None, None))
    out = attn(query, value, n_buckets, n_hashes)
    out = jnp.reshape(out, [batch_size, qlength, qkv_features])
    out = out[:, :orig_seqlen, :]
    return out

ReformerSelfAttention = ReformerAttention.partial(inputs_kv=None)
