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
"""Longformer modules."""
from flax import nn
import jax.numpy as jnp
from lra_benchmarks.models.layers import common_layers
from lra_benchmarks.models.longformer import longformer_attention


class LongformerBlock(nn.Module):
  """Longformer Layer."""

  def apply(self,
            inputs,
            qkv_dim,
            mlp_dim,
            num_heads,
            sliding_window_size=512,
            global_mask=None,
            causal_mask=False,
            dtype=jnp.float32,
            inputs_segmentation=None,
            padding_mask=None,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            deterministic=False):
    """Applies the LongformerBlock module.

    Args:
      inputs: input data of size `[bs, seq_len, features]`.
      qkv_dim: dimension of the query/key/value.
      mlp_dim: dimension of the mlp on top of attention block.
      num_heads: number of attention heads.
      sliding_window_size: size of sliding window attention to use.
      global_mask: boolean matrix of shape `[bs, seq_len]`, where `True`
        indicates that the position is globally attended. By default, no global
        attention is used.
      causal_mask: If true, apply causal attention mask.
      dtype: the dtype of the computation (default: float32).
      inputs_segmentation: input segmentation info for packed examples.
      padding_mask: bool, mask padding tokens.
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      deterministic: if true, apply dropout else don't.

    Returns:
      output of shape `[bs, seq_len, mlp_dim]`.
    """

    assert inputs.ndim == 3
    x = nn.LayerNorm(inputs)
    x = longformer_attention.LongformerSelfAttention(
        x,
        num_heads=num_heads,
        qkv_features=qkv_dim,
        sliding_window_size=sliding_window_size,
        global_mask=global_mask,
        causal_mask=causal_mask,
        dtype=dtype,
        segmentation=inputs_segmentation,
        padding_mask=padding_mask,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        bias=False,
        broadcast_dropout=False,
        dropout_rate=attention_dropout_rate,
        deterministic=deterministic)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    x = x + inputs

    y = nn.LayerNorm(x)
    y = common_layers.MlpBlock(
        y,
        mlp_dim=mlp_dim,
        dtype=dtype,
        dropout_rate=dropout_rate,
        deterministic=deterministic)

    return x + y


class LongformerEncoder(nn.Module):
  """Longformer Encoder."""

  def apply(self,
            inputs,
            vocab_size,
            sliding_window_size=512,
            global_mask=None,
            causal_mask=False,
            inputs_positions=None,
            inputs_segmentation=None,
            shared_embedding=None,
            use_bfloat16=False,
            emb_dim=512,
            num_heads=8,
            dtype=jnp.float32,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=512,
            train=True,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            learn_pos_emb=False,
            classifier=False,
            classifier_pool='CLS',
            num_classes=10):
    """Applies Longformer model on the inputs.

    Args:
      inputs: input data.
      vocab_size: size of the vocabulary.
      sliding_window_size: size of sliding window attention to use.
      global_mask: boolean matrix of shape `[bs, seq_len]`, where `True`
        indicates that the position is globally attended. By default, no global
        attention is used.
      causal_mask: If true, apply causal attention masking.
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      shared_embedding: a shared embedding layer to use.
      use_bfloat16: bool: whether use bfloat16.
      emb_dim: dimension of embedding
      num_heads: number of heads
      dtype: the dtype of the computation (default: float32)
      num_layers: number of layers
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      max_len: maximum length.
      train: if it is training,
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      learn_pos_emb: boolean, if learn the positional embedding or use the
        sinusoidal positional embedding.
      classifier: boolean, for classification mode (output N-class logits)
      classifier_pool: str, supports "MEAN", "MAX" pooling.
      num_classes: int, number of classification classes.

    Returns:
      output of the encoder or logits if classifier_mode is true.
    """
    assert inputs.ndim == 2  # (batch, len)

    # Padding Masks
    src_padding_mask = (inputs > 0)[..., None]

    # Input Embedding
    if shared_embedding is None:
      input_embed = nn.Embed.partial(
          num_embeddings=vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      input_embed = shared_embedding
    x = inputs.astype('int32')
    x = input_embed(x)

    if classifier and classifier_pool == 'CLS':
      cls = self.param('cls', (1, 1, emb_dim), nn.initializers.zeros)
      cls = jnp.tile(cls, [x.shape[0], 1, 1])
      x = jnp.concatenate([cls, x], axis=1)
      max_len += 1
      src_padding_mask = jnp.concatenate(
          [src_padding_mask[:, :1], src_padding_mask], axis=1)

    pe_init = nn.initializers.normal(stddev=0.02) if learn_pos_emb else None
    x = common_layers.AddPositionEmbs(
        x,
        inputs_positions=inputs_positions,
        posemb_init=pe_init,
        max_len=max_len,
        name='posembed_input')
    x = nn.dropout(x, rate=dropout_rate, deterministic=not train)

    if use_bfloat16:
      x = x.astype(jnp.bfloat16)
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float32

    # Input Encoder
    for lyr in range(num_layers):
      x = LongformerBlock(
          x,
          qkv_dim=qkv_dim,
          mlp_dim=mlp_dim,
          num_heads=num_heads,
          sliding_window_size=sliding_window_size,
          global_mask=global_mask,
          causal_mask=causal_mask,
          dtype=dtype,
          inputs_segmentation=inputs_segmentation,
          padding_mask=src_padding_mask,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          deterministic=not train,
          name=f'encoderblock_{lyr}')
    encoded = nn.LayerNorm(x, dtype=dtype, name='encoder_norm')

    if classifier:
      encoded = common_layers.classifier_head(
          encoded, num_classes, mlp_dim, pooling_mode=classifier_pool)
    return encoded


class LongformerDualEncoder(nn.Module):
  """Longformer Model for Matching (dual encoding) tasks."""

  def apply(self,
            inputs1,
            inputs2,
            vocab_size=None,
            inputs1_positions=None,
            inputs2_positions=None,
            inputs1_segmentation=None,
            inputs2_segmentation=None,
            use_bfloat16=False,
            emb_dim=512,
            num_heads=8,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=2048,
            train=False,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            classifier=True,
            classifier_pool='CLS',
            num_classes=2,
            interaction=None
            ):
    """Applies Transformer model on text similarity.

    A deliberate choice to distinguish this from NLI because
    we may want to do different things to the model later. Dual Encoding
    mode enforces that we do not do cross attention between pairs.

    Args:
      inputs1: input data.
      inputs2: target data.
      vocab_size: size of the input vocabulary.
      inputs1_positions: input subsequence positions for packed examples.
      inputs2_positions: target subsequence positions for packed examples.
      inputs1_segmentation: input segmentation info for packed examples.
      inputs2_segmentation: target segmentation info for packed examples.
      use_bfloat16: bool: whether use bfloat16.
      emb_dim: dimension of embedding.
      num_heads: number of heads.
      num_layers: number of layers.
      qkv_dim: dimension of the query/key/value.
      mlp_dim: dimension of the mlp on top of attention block.
      max_len: maximum length.
      train: whether it is training.
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout rate for attention weights.
      classifier: boolean, to use classifier.
      classifier_pool: str, supports "MEAN", "MAX" pooling.
      num_classes: int, number of classification classes.
      interaction: str

    Returns:
      output of a transformer decoder.
    """
    encoder = LongformerEncoder.shared(
        inputs_positions=inputs1_positions,
        inputs_segmentation=inputs1_segmentation,
        vocab_size=vocab_size,
        use_bfloat16=use_bfloat16,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim,
        max_len=max_len,
        train=train,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        name='encoder')
    inputs1_encoded = encoder(inputs1)
    inputs2_encoded = encoder(inputs2)

    encoded = common_layers.classifier_head_dual(
        inputs1_encoded,
        inputs2_encoded,
        num_classes,
        mlp_dim,
        pooling_mode=classifier_pool,
        interaction=interaction)

    return encoded


class LongformerDecoder(nn.Module):
  """Longformer Decoder."""

  def apply(self,
            inputs,
            vocab_size,
            sliding_window_size=512,
            global_mask=None,
            emb_dim=512,
            num_heads=8,
            dtype=jnp.float32,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=2048,
            train=False,
            shift=True,
            dropout_rate=0.1,
            attention_dropout_rate=0.1):
    """Applies Longformer model on the inputs, using causal masking.

    Args:
      inputs: input data
      vocab_size: size of the vocabulary
      sliding_window_size: size of sliding window attention to use.
      global_mask: boolean matrix of shape `[bs, seq_len]`, where `True`
        indicates that the position is globally attended. By default, no global
        attention is used.
      emb_dim: dimension of embedding
      num_heads: number of heads
      dtype: the dtype of the computation (default: float32)
      num_layers: number of layers
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      max_len: maximum length.
      train: bool: if model is training.
      shift: bool: if we right-shift input - this is only disabled for
        fast, looped single-token autoregressive decoding.
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights

    Returns:
      output of a transformer decoder.
    """
    padding_mask = jnp.where(inputs > 0, 1, 0).astype(jnp.float32)[..., None]
    assert inputs.ndim == 2  # (batch, len)
    x = inputs
    if shift:
      x = common_layers.shift_right(x)
    x = x.astype('int32')
    x = common_layers.Embed(
        x, num_embeddings=vocab_size, features=emb_dim, name='embed')
    x = common_layers.AddPositionEmbs(
        x,
        max_len=max_len,
        posemb_init=common_layers.sinusoidal_init(max_len=max_len),
        cache=None)
    x = nn.dropout(x, rate=dropout_rate, deterministic=not train)
    for _ in range(num_layers):
      x = LongformerBlock(
          x,
          qkv_dim=qkv_dim,
          mlp_dim=mlp_dim,
          num_heads=num_heads,
          sliding_window_size=sliding_window_size,
          global_mask=global_mask,
          causal_mask=True,
          padding_mask=padding_mask,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          deterministic=not train,
          cache=None,
      )
    x = nn.LayerNorm(x)
    logits = nn.Dense(
        x,
        vocab_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))
    return logits
