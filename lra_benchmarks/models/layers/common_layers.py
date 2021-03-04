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
# Lint as: python3
"""Common layers used in models."""
from flax import nn
from jax import lax
import jax.numpy as jnp
import numpy as np


class Embed(nn.Module):
  """Embedding Module.

  A parameterized function from integers [0, n) to d-dimensional vectors.
  """

  def apply(self,
            inputs,
            num_embeddings,
            features,
            mode='input',
            emb_init=nn.initializers.normal(stddev=1.0)):
    """Applies Embed module.

    Args:
      inputs: input data
      num_embeddings: number of embedding
      features: size of the embedding dimension
      mode: either 'input' or 'output' -> to share input/output embedding
      emb_init: embedding initializer

    Returns:
      output which is embedded input data
    """
    embedding = self.param('embedding', (num_embeddings, features), emb_init)
    if mode == 'input':
      if inputs.dtype not in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64]:
        raise ValueError('Input type must be an integer or unsigned integer.')
      return jnp.take(embedding, inputs, axis=0)
    if mode == 'output':
      return jnp.einsum('bld,vd->blv', inputs, embedding)


def sinusoidal_init(max_len=2048):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_feature, 2) * -(np.log(10000.0) / d_feature))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs."""

  def apply(self,
            inputs,
            inputs_positions=None,
            max_len=512,
            posemb_init=None,
            cache=None):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init.

    Args:
      inputs: input data.
      inputs_positions: input position indices for packed sequences.
      max_len: maximum possible length for the input.
      posemb_init: positional embedding initializer, if None, then use a
        fixed (non-learned) sinusoidal embedding table.
      cache: flax attention cache for fast decoding.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    length = inputs.shape[1]
    pos_emb_shape = (1, max_len, inputs.shape[-1])
    if posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(
          max_len=max_len)(None, pos_emb_shape, None)
    else:
      pos_embedding = self.param('pos_embedding', pos_emb_shape, posemb_init)
    pe = pos_embedding[:, :length, :]
    # We abuse the same attention Cache mechanism to run positional embeddings
    # in fast predict mode. We could use state variables instead, but this
    # simplifies invocation with a single top-level cache context manager.
    # We only use the cache's position index for tracking decoding position.
    if cache:
      if self.is_initializing():
        cache.store(np.array((4, 1, 1), dtype=np.int32))
      else:
        cache_entry = cache.retrieve(None)
        i = cache_entry.i
        cache.store(cache_entry.replace(i=cache_entry.i + 1))
        _, _, df = pos_embedding.shape
        pe = lax.dynamic_slice(pos_embedding,
                               jnp.array((0, i, 0)),
                               jnp.array((1, 1, df)))
    if inputs_positions is None:
      # normal unpacked case:
      return inputs + pe
    else:
      # for packed data we need to use known position indices:
      return inputs + jnp.take(pe[0], inputs_positions, axis=0)


class MlpBlock(nn.Module):
  """Transformer MLP block."""

  def apply(self,
            inputs,
            mlp_dim,
            dtype=jnp.float32,
            out_dim=None,
            dropout_rate=0.1,
            deterministic=False,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6)):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if out_dim is None else out_dim
    x = nn.Dense(inputs, mlp_dim, dtype=dtype,
                 kernel_init=kernel_init, bias_init=bias_init)
    x = nn.gelu(x)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    output = nn.Dense(
        x, actual_out_dim, dtype=dtype, kernel_init=kernel_init,
        bias_init=bias_init)
    output = nn.dropout(output, rate=dropout_rate, deterministic=deterministic)
    return output


def classifier_head(encoded, num_classes, mlp_dim, pooling_mode='MEAN'):
  """Classifier head.

  We put this here just so that all models consistently call the same function.

  Args:
    encoded: tensor inputs are shape of [bs, len, dim].
    num_classes: int, number of classes
    mlp_dim: int, dim of intermediate MLP.
    pooling_mode: str, string dictating pooling op {MEAN}

  Returns:
    tensor of shape [bs, num_classes]

  """
  if pooling_mode == 'MEAN':
    encoded = jnp.mean(encoded, axis=1)
  elif pooling_mode == 'SUM':
    encoded = jnp.sum(encoded, axis=1)
  elif pooling_mode == 'FLATTEN':
    encoded = encoded.reshape((encoded.shape[0], -1))
  elif pooling_mode == 'CLS':
    encoded = encoded[:, 0]
  else:
    raise NotImplementedError('Pooling not supported yet.')
  encoded = nn.Dense(encoded, mlp_dim, name='mlp')
  encoded = nn.relu(encoded)
  encoded = nn.Dense(encoded, num_classes, name='logits')
  return encoded


def classifier_head_dual(encoded1, encoded2,
                         num_classes, mlp_dim, pooling_mode='MEAN',
                         interaction=None):
  """Classifier head for dual encoding or pairwise problem.

  We put this here just so that all models consistently call the same function.

  Args:
    encoded1: tensor inputs are shape of [bs, len, dim].
    encoded2: tensor inputs are shape of [bs, len, dim].
    num_classes: int, number of classes
    mlp_dim: int, dim of intermediate MLP.
    pooling_mode: str, string dictating pooling op {MEAN}
    interaction: str, string dictating interaction between e1, e2

  Returns:
    tensor of shape [bs, num_classes]

  """
  if pooling_mode == 'MEAN':
    encoded1 = jnp.mean(encoded1, axis=1)
    encoded2 = jnp.mean(encoded2, axis=1)
  elif pooling_mode == 'SUM':
    encoded1 = jnp.sum(encoded1, axis=1)
    encoded2 = jnp.sum(encoded2, axis=1)
  elif pooling_mode == 'FLATTEN':
    encoded1 = encoded1.reshape((encoded1.shape[0], -1))
    encoded2 = encoded2.reshape((encoded2.shape[0], -1))
  elif pooling_mode == 'CLS':
    encoded1 = encoded1[:, 0]
    encoded2 = encoded2[:, 0]
  else:
    raise NotImplementedError('Pooling not supported yet.')

  if interaction == 'NLI':
    # NLI interaction style
    encoded = jnp.concatenate([encoded1, encoded2, encoded1 * encoded2,
                               encoded1 - encoded2], 1)
  else:
    encoded = jnp.concatenate([encoded1, encoded2], 1)
  encoded = nn.Dense(encoded, mlp_dim, name='mlp')
  encoded = nn.relu(encoded)
  encoded = nn.Dense(encoded, int(mlp_dim // 2), name='mlp2')
  encoded = nn.relu(encoded)
  encoded = nn.Dense(encoded, num_classes, name='logits')
  return encoded
