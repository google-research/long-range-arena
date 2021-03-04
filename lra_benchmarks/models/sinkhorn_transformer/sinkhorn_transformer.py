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
"""Sinkhorn Attention Transformer models."""
from flax import nn
import jax.numpy as jnp
from lra_benchmarks.models.layers import common_layers
from lra_benchmarks.models.sinkhorn_transformer import sinkhorn_attention


class SinkhornTransformerBlock(nn.Module):
  """Transformer layer (https://openreview.net/forum?id=H1e5GJBtDr)."""

  def apply(self,
            inputs,
            qkv_dim,
            mlp_dim,
            num_heads,
            dtype=jnp.float32,
            inputs_segmentation=None,
            causal_mask=False,
            padding_mask=None,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            deterministic=False,
            cache=None,
            block_size=50):
    """Applies TransformerBlock module.

    Args:
      inputs: input data
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: number of heads
      dtype: the dtype of the computation (default: float32).
      inputs_segmentation: input segmentation info for packed examples.
      causal_mask: bool, mask future or not
      padding_mask: bool, mask padding tokens
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      deterministic: bool, deterministic or not (to apply dropout)
      cache: flax autoregressive cache for fast decoding.
      block_size: int, block size.

    Returns:
      output after transformer block.

    """

    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(inputs)
    x = sinkhorn_attention.SinkhornSelfAttention(
        x,
        num_heads=num_heads,
        dtype=dtype,
        qkv_features=qkv_dim,
        attention_axis=(1,),
        causal_mask=causal_mask,
        segmentation=inputs_segmentation,
        padding_mask=padding_mask,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        bias=False,
        broadcast_dropout=False,
        dropout_rate=attention_dropout_rate,
        deterministic=deterministic,
        cache=cache,
        block_size=block_size)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(x)
    y = common_layers.MlpBlock(
        y,
        mlp_dim=mlp_dim,
        dtype=dtype,
        dropout_rate=dropout_rate,
        deterministic=deterministic)

    return x + y


class SinkhornTransformerEncoder(nn.Module):
  """Sinkhorn Transformer Encoder."""

  def apply(self,
            inputs,
            vocab_size,
            inputs_positions=None,
            inputs_segmentation=None,
            shared_embedding=None,
            use_bfloat16=False,
            emb_dim=512,
            num_heads=8,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=512,
            train=True,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            block_size=50,
            learn_pos_emb=False,
            classifier=False,
            classifier_pool='MEAN',
            num_classes=10):
    """Applies Local Transformer model on the inputs.

    Args:
      inputs: input data
      vocab_size: size of the vocabulary
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      shared_embedding: a shared embedding layer to use.
      use_bfloat16: bool: whether use bfloat16.
      emb_dim: dimension of embedding
      num_heads: number of heads
      num_layers: number of layers
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      max_len: maximum length.
      train: if it is training,
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      block_size: int, block size.
      learn_pos_emb: boolean, if learn the positional embedding or use the
        sinusoidal positional embedding.
      classifier: boolean, for classification mode (output N-class logits)
      classifier_pool: str, supports "MEAN", "MAX" pooling.
      num_classes: int, number of classification classes.

    Returns:
      output of a transformer encoder.
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
      x = SinkhornTransformerBlock(
          x,
          qkv_dim=qkv_dim,
          mlp_dim=mlp_dim,
          num_heads=num_heads,
          dtype=dtype,
          padding_mask=src_padding_mask,
          inputs_segmentation=inputs_segmentation,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          deterministic=not train,
          name=f'encoderblock_{lyr}',
          block_size=block_size)
    encoded = nn.LayerNorm(x, dtype=dtype, name='encoder_norm')

    if classifier:
      if classifier_pool == 'MEAN':
        encoded = jnp.mean(encoded, axis=1)
        encoded = nn.Dense(encoded, num_classes, name='logits')
      else:
        # TODO: Add other pooling methods.
        raise ValueError('Pooling method not supported yet.')
    return encoded


class LocalTransformerDecoder(nn.Module):
  """Local Transformer Decoder."""

  def apply(self,
            inputs,
            vocab_size,
            emb_dim=512,
            num_heads=8,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=2048,
            train=False,
            shift=True,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            cache=None,
            block_size=50):
    """Applies Local Attention model on the inputs.

    Args:
      inputs: input data
      vocab_size: size of the vocabulary
      emb_dim: dimension of embedding
      num_heads: number of heads
      num_layers: number of layers
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      max_len: maximum length.
      train: bool: if model is training.
      shift: bool: if we right-shift input - this is only disabled for
        fast, looped single-token autoregressive decoding.
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      cache: flax autoregressive cache for fast decoding.
      block_size: int, block size

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
        cache=cache)
    x = nn.dropout(x, rate=dropout_rate, deterministic=not train)
    for _ in range(num_layers):
      x = SinkhornTransformerBlock(
          x,
          qkv_dim=qkv_dim,
          mlp_dim=mlp_dim,
          num_heads=num_heads,
          causal_mask=True,
          padding_mask=padding_mask,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          deterministic=not train,
          cache=cache,
          block_size=block_size)
    x = nn.LayerNorm(x)
    logits = nn.Dense(
        x,
        vocab_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))
    return logits


class SinkhornTransformerSeq2SeqBlock(nn.Module):
  """Local Transformer encoder-decoder layer."""

  def apply(self,
            targets,
            encoded,
            qkv_dim,
            mlp_dim,
            num_heads,
            dtype=jnp.float32,
            inputs_segmentation=None,
            targets_segmentation=None,
            padding_mask=None,
            key_padding_mask=None,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            deterministic=False,
            cache=None,
            block_size=50):
    """Applies TransformerSeq2Seq module.

    Args:
      targets: input data for decoder
      encoded: input data from encoder
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: number of heads
      dtype: the dtype of the computation (default: float32)
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.
      padding_mask: bool, mask padding tokens
      key_padding_mask: bool, mask padding tokens
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      deterministic: bool, deterministic or not (to apply dropout)
      cache: flax attention cache for fast decoding.
      block_size: int block size.

    Returns:
      output after transformer encoder-decoder block.
    """

    # Decoder block.
    assert targets.ndim == 3
    x = nn.LayerNorm(targets, dtype=dtype)

    x = sinkhorn_attention.SinkhornSelfAttention(
        x,
        num_heads=num_heads,
        dtype=dtype,
        inputs_kv=x,
        qkv_features=qkv_dim,
        attention_axis=(1,),
        causal_mask=True,
        padding_mask=padding_mask,
        segmentation=targets_segmentation,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        bias=False,
        broadcast_dropout=False,
        dropout_rate=attention_dropout_rate,
        deterministic=deterministic,
        cache=cache,
        block_size=block_size)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    x = x + targets

    # Encoder-Decoder block.
    # Don't use block attention for cross attention.
    y = nn.LayerNorm(x, dtype=dtype)
    y = nn.SelfAttention(
        y,
        num_heads=num_heads,
        dtype=dtype,
        inputs_kv=encoded,
        qkv_features=qkv_dim,
        attention_axis=(1,),
        causal_mask=False,
        padding_mask=padding_mask,
        key_padding_mask=key_padding_mask,
        segmentation=targets_segmentation,
        key_segmentation=inputs_segmentation,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        bias=False,
        broadcast_dropout=False,
        dropout_rate=attention_dropout_rate,
        deterministic=deterministic)
    y = nn.dropout(y, rate=dropout_rate, deterministic=deterministic)
    y = y + x

    # MLP block.
    z = nn.LayerNorm(y, dtype=dtype)
    z = common_layers.MlpBlock(
        z,
        mlp_dim=mlp_dim,
        dtype=dtype,
        dropout_rate=dropout_rate,
        deterministic=deterministic)

    return y + z


class SinkhornTransformerDualEncoder(nn.Module):
  """Sinkhorn Transformer Model for Matching (dual encoding) tasks."""

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
            interaction=None):
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
      interaction: str.

    Returns:
      output of a transformer decoder.
    """
    encoder = SinkhornTransformerEncoder.shared(
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


class SinkhornTransformerSeq2SeqDecoder(nn.Module):
  """Local Transformer Model Decoder for sequence to sequence translation."""

  def apply(self,
            encoded,
            src_padding_mask,
            targets,
            output_vocab_size,
            targets_positions=None,
            inputs_segmentation=None,
            targets_segmentation=None,
            tgt_padding_mask=None,
            shared_embedding=None,
            logits_via_embedding=False,
            shift=True,
            use_bfloat16=False,
            emb_dim=512,
            num_heads=8,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=512,
            train=True,
            cache=None,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            block_size=50):
    """Applies Sinkhorn Transformer model on the inputs.

    Args:
      encoded: encoded input data from encoder.
      src_padding_mask: padding mask for inputs.
      targets: target inputs.
      output_vocab_size: size of the vocabulary.
      targets_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.
      tgt_padding_mask: target tokens padding mask.
      shared_embedding: a shared embedding layer to use.
      logits_via_embedding: bool: whether final logit transform shares
        embedding weights.
      shift: whether to shift or not (for fast decoding).
      use_bfloat16: bool: whether use bfloat16.
      emb_dim: dimension of embedding.
      num_heads: number of heads.
      num_layers: number of layers.
      qkv_dim: dimension of the query/key/value.
      mlp_dim: dimension of the mlp on top of attention block.
      max_len: maximum length.
      train: whether it is training.
      cache: flax attention cache for fast decoding.
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout rate for attention weights.
      block_size: int block size.

    Returns:
      output of a transformer decoder.
    """
    assert encoded.ndim == 3  # (batch, len, depth)
    assert targets.ndim == 2  # (batch, len)

    # Padding Masks
    if tgt_padding_mask is None:
      tgt_padding_mask = (targets > 0)[..., None]

    # Target Embedding
    if shared_embedding is None:
      output_embed = nn.Embed.partial(
          num_embeddings=output_vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      output_embed = shared_embedding

    y = targets.astype('int32')
    if shift:
      y = common_layers.shift_right(y)
    y = output_embed(y)
    y = common_layers.AddPositionEmbs(
        y,
        inputs_positions=targets_positions,
        max_len=max_len,
        cache=cache,
        name='posembed_output')
    y = nn.dropout(y, rate=dropout_rate, deterministic=not train)

    if use_bfloat16:
      y = y.astype(jnp.bfloat16)
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float32

    # Target-Input Decoder
    for lyr in range(num_layers):
      y = SinkhornTransformerSeq2SeqBlock(
          y,
          encoded,
          qkv_dim=qkv_dim,
          mlp_dim=mlp_dim,
          num_heads=num_heads,
          dtype=dtype,
          padding_mask=tgt_padding_mask,
          key_padding_mask=src_padding_mask,
          inputs_segmentation=inputs_segmentation,
          targets_segmentation=targets_segmentation,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          deterministic=not train,
          cache=cache,
          name=f'encoderdecoderblock_{lyr}',
          block_size=block_size)
    y = nn.LayerNorm(y, dtype=dtype, name='encoderdecoder_norm')

    # Decoded Logits
    if logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = output_embed.attend(y.astype(jnp.float32))
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    else:
      logits = nn.Dense(
          y,
          output_vocab_size,
          dtype=dtype,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6),
          name='logitdense')
    return logits


class SinkhornTransformerSeq2Seq(nn.Module):
  """Transformer Model for sequence to sequence translation."""

  def apply(self,
            inputs,
            targets,
            vocab_size=None,
            output_vocab_size=None,
            inputs_positions=None,
            targets_positions=None,
            inputs_segmentation=None,
            targets_segmentation=None,
            tgt_padding_mask=None,
            share_embeddings=False,
            logits_via_embedding=False,
            use_bfloat16=False,
            emb_dim=512,
            num_heads=8,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=2048,
            train=False,
            shift=True,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            cache=None,
            block_size=50):
    """Applies Sinkhorn Transformer model on the inputs.

    Args:
      inputs: input data.
      targets: target data.
      vocab_size: size of the input vocabulary.
      output_vocab_size: size of the output vocabulary. If None, the output
        vocabulary size is assumed to be the same as vocab_size.
      inputs_positions: input subsequence positions for packed examples.
      targets_positions: target subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.
      tgt_padding_mask: target tokens padding mask.
      share_embeddings: bool: share embedding layer for inputs and targets.
      logits_via_embedding: bool: whether final logit transform shares
        embedding weights.
      use_bfloat16: bool: whether use bfloat16.
      emb_dim: dimension of embedding.
      num_heads: number of heads.
      num_layers: number of layers.
      qkv_dim: dimension of the query/key/value.
      mlp_dim: dimension of the mlp on top of attention block.
      max_len: maximum length.
      train: whether it is training.
      shift: whether to right-shift targets.
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout rate for attention weights.
      cache: flax autoregressive cache for fast decoding.
      block_size: int, block_size.

    Returns:
      output of a transformer decoder.
    """
    src_padding_mask = (inputs > 0)[..., None]

    if share_embeddings:
      if output_vocab_size is not None:
        assert output_vocab_size == vocab_size, (
            "can't share embedding with different vocab sizes.")
      shared_embedding = nn.Embed.shared(
          num_embeddings=vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      shared_embedding = None

    encoded = SinkhornTransformerEncoder(
        inputs,
        inputs_positions=inputs_positions,
        inputs_segmentation=inputs_segmentation,
        vocab_size=vocab_size,
        shared_embedding=shared_embedding,
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
        name='encoder',
        block_size=block_size)

    logits = SinkhornTransformerSeq2SeqDecoder(
        encoded,
        src_padding_mask,
        targets,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        output_vocab_size=output_vocab_size,
        shared_embedding=shared_embedding,
        logits_via_embedding=logits_via_embedding,
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
        cache=cache,
        name='decoder',
        block_size=block_size)
    return logits.astype(jnp.float32) if use_bfloat16 else logits

  # The following two methods allow us to run the trained Transformer in
  # two parts during fast decoding.  First, we call the encoder branch to
  # encode the inputs, then we call the decoder branch while providing a
  # cache object for iteratively storing keys and values during the decoding
  # process.

  @nn.module_method
  def encode(self,
             inputs,
             vocab_size=None,
             output_vocab_size=None,
             inputs_positions=None,
             inputs_segmentation=None,
             targets_positions=None,
             targets_segmentation=None,
             tgt_padding_mask=None,
             share_embeddings=False,
             logits_via_embedding=False,
             use_bfloat16=False,
             emb_dim=512,
             num_heads=8,
             num_layers=6,
             qkv_dim=512,
             mlp_dim=2048,
             max_len=2048,
             train=True,
             shift=True,
             dropout_rate=0.1,
             attention_dropout_rate=0.1,
             cache=None,
             block_size=50):
    del (output_vocab_size, shift, targets_positions,
         targets_segmentation, tgt_padding_mask, logits_via_embedding,
         cache)

    if share_embeddings:
      shared_embedding = nn.Embed.shared(
          num_embeddings=vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      shared_embedding = None

    encoded = SinkhornTransformerEncoder(
        inputs,
        inputs_positions=inputs_positions,
        inputs_segmentation=inputs_segmentation,
        vocab_size=vocab_size,
        shared_embedding=shared_embedding,
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
        name='encoder',
        block_size=block_size)

    return encoded

  @nn.module_method
  def decode(self,
             encoded,
             src_padding_mask,
             targets,
             inputs_positions=None,
             vocab_size=None,
             output_vocab_size=None,
             targets_positions=None,
             inputs_segmentation=None,
             targets_segmentation=None,
             tgt_padding_mask=None,
             share_embeddings=False,
             logits_via_embedding=False,
             use_bfloat16=False,
             emb_dim=512,
             num_heads=8,
             num_layers=6,
             qkv_dim=512,
             mlp_dim=2048,
             max_len=2048,
             train=True,
             shift=True,
             dropout_rate=0.1,
             attention_dropout_rate=0.1,
             cache=None,
             block_size=50):
    del inputs_positions

    if share_embeddings:
      shared_embedding = nn.Embed.shared(
          num_embeddings=vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      shared_embedding = None

    logits = SinkhornTransformerSeq2SeqDecoder(
        encoded,
        src_padding_mask,
        targets,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        tgt_padding_mask=tgt_padding_mask,
        output_vocab_size=output_vocab_size,
        shared_embedding=shared_embedding,
        logits_via_embedding=logits_via_embedding,
        use_bfloat16=use_bfloat16,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim,
        max_len=max_len,
        train=train,
        shift=shift,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        cache=cache,
        name='decoder',
        block_size=block_size)

    return logits
