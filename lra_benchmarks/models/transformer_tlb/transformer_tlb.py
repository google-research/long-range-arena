# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transformer-based stateful lra models."""
from flax.deprecated import nn
import jax
import jax.numpy as jnp
from lra_benchmarks.models.layers import common_layers
from lra_benchmarks.models.transformer import transformer


class CrossTransformerBlock(nn.Module):
  """Cross Transformer layer."""

  def apply(self,
            inputs_q,
            inputs_kv,
            qkv_dim,
            mlp_dim,
            num_heads,
            dtype=jnp.float32,
            inputs_segmentation=None,
            causal_mask=False,
            padding_mask=None,
            key_padding_mask=None,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            deterministic=False,
            cache=None,
            residual=True):
    """Applies CrossTransformerBlock module.

    Args:
      inputs_q: input query
      inputs_kv: input key-value
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: number of heads
      dtype: the dtype of the computation (default: float32).
      inputs_segmentation: input segmentation info for packed examples.
      causal_mask: bool, mask future or not
      padding_mask: bool, mask padding tokens
      key_padding_mask: bool, mask padding tokens
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      deterministic: bool, deterministic or not (to apply dropout)
      cache: flax autoregressive cache for fast decoding.
      residual: Boolean, to use residual connectors or not.

    Returns:
      output after transformer block.

    """

    # Attention block.
    assert inputs_q.ndim == 3
    x = nn.LayerNorm(inputs_q)
    s = nn.LayerNorm(inputs_kv)
    x = nn.MultiHeadDotProductAttention(
        x, s,
        num_heads=num_heads,
        dtype=dtype,
        qkv_features=qkv_dim,
        attention_axis=(1,),
        causal_mask=causal_mask,
        segmentation=inputs_segmentation,
        padding_mask=padding_mask,
        key_padding_mask=key_padding_mask,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        bias=False,
        broadcast_dropout=False,
        dropout_rate=attention_dropout_rate,
        deterministic=deterministic,
        cache=cache)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    x = x + inputs_q

    # MLP block.
    y = nn.LayerNorm(x)
    y = common_layers.MlpBlock(
        y,
        mlp_dim=mlp_dim,
        dtype=dtype,
        dropout_rate=dropout_rate,
        deterministic=deterministic)

    if residual:
      output = x + y
    else:
      output = x
    if padding_mask is not None:
      corner_case = (jnp.sum(padding_mask,
                             axis=1) == 0)[..., None]
      output = jnp.where(corner_case, inputs_q, output)
    elif key_padding_mask is not None:
      corner_case = (jnp.sum(key_padding_mask,
                             axis=1) == 0)[..., None]
      output = jnp.where(corner_case, inputs_q, output)
    return output


class StatefulTransformerEncoder(nn.Module):
  """Stateful Transformer Model Encoder (https://arxiv.org/abs/2205.14794)."""

  def apply(self,
            inputs,
            vocab_size,
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
            num_classes=10,
            tied_weights=False,
            meta_network=False,
            meta_layers=1,
            meta_pool='last',
            use_residual=True,
            meta_partition=3,
            meta_layer_output=False,
            self_to_cross_ratio_input_updater=2,
            num_cross_layers_input_updater=1,
            num_cross_layers_state_updater=1,
            num_state_tokens=20,
            block_size=20,
            use_global_pos_encoding=False):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      vocab_size: size of the vocabulary
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
      tied_weights: bool, to tie weights or not.
      meta_network: boolean, experimental extreme self-attention.
      meta_layers: int, number of meta_layers
      meta_pool: str, the type of meta pooling.
      use_residual: boolean, turn off transformer residuals.
      meta_partition: int.
      meta_layer_output: boolean.
      self_to_cross_ratio_input_updater: number of self-attention layers before
        each cross attention layer in the input-update direction
      num_cross_layers_input_updater: number of cross-attention layers
        in the input update direction
      num_cross_layers_state_updater: number of cross-attention layers
        in the state update direction
      num_state_tokens: number of state tokens
      block_size: chunk size of inputs
      use_global_pos_encoding: Whether the input position embedding is global
        or local

    Returns:
      output of a transformer encoder or logits if classifier_mode is true.
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

    # Input Positional Encoding
    pe_init = nn.initializers.normal(stddev=0.02) if learn_pos_emb else None
    if use_global_pos_encoding:
      x = common_layers.AddPositionEmbs(
          x,
          inputs_positions=inputs_positions,
          posemb_init=pe_init,
          max_len=max_len,
          name='global_posembed_input')
    pe = common_layers.AddPositionEmbs(
        jnp.zeros((x.shape[0], block_size, x.shape[2]),
                  dtype=x.dtype),
        inputs_positions=inputs_positions,
        posemb_init=pe_init,
        max_len=max_len,
        name='posembed_input')

    if use_bfloat16:
      x = x.astype(jnp.bfloat16)
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float32

    # Create layers
    horizontal_blocks = []
    for block_idx in range(num_cross_layers_input_updater):
      horizontal_layers = []
      for layer_idx in range(self_to_cross_ratio_input_updater):
        horizontal_layers.append(
            transformer.TransformerBlock.shared(
                qkv_dim=qkv_dim,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                dtype=dtype,
                inputs_segmentation=inputs_segmentation,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                deterministic=not train,
                name=f'horizontal_block_{block_idx}_self_{layer_idx}'))
      horizontal_layers.append(
          CrossTransformerBlock.shared(
              qkv_dim=qkv_dim,
              mlp_dim=mlp_dim,
              num_heads=num_heads,
              dtype=dtype,
              inputs_segmentation=inputs_segmentation,
              dropout_rate=dropout_rate,
              attention_dropout_rate=attention_dropout_rate,
              deterministic=not train,
              name=f'horizontal_block_{block_idx}_cross',
              residual=use_residual))
      horizontal_blocks.append(horizontal_layers)
    vertical_layers = []
    for layer_idx in range(num_cross_layers_state_updater):
      vertical_layers.append(
          CrossTransformerBlock.shared(
              qkv_dim=qkv_dim,
              mlp_dim=mlp_dim,
              num_heads=num_heads,
              dtype=dtype,
              inputs_segmentation=inputs_segmentation,
              dropout_rate=dropout_rate,
              attention_dropout_rate=attention_dropout_rate,
              deterministic=not train,
              name=f'vertical_block_{block_idx}_cross',
              residual=use_residual))

    num_tokens = x.shape[1]
    num_chunks = num_tokens // block_size
    init_state = jnp.zeros((x.shape[0], num_state_tokens, x.shape[2]))
    x_with_pad = jnp.concatenate([x, src_padding_mask], axis=2)
    # Split inputs into chunks of block_size.
    x_with_pad = jnp.stack(
        jnp.split(x_with_pad, num_chunks, axis=1), axis=0)

    # State positional encoding
    state_pos_embed = common_layers.AddPositionEmbs(
        init_state,
        inputs_positions=None,
        posemb_init=None,
        max_len=num_state_tokens,
        name='posembed_state')

    # Processing function for each chunk
    def scan_inner(cur_state, cur_x_with_pad):
      padding_mask_cur = cur_x_with_pad[:, :, -1][:, :, None]
      x_cur = cur_x_with_pad[:, :, :-1]
      if not use_global_pos_encoding:
        x_cur = x_cur + pe
      x_cur = nn.dropout(x_cur, rate=dropout_rate, deterministic=not train)
      cur_state = cur_state + state_pos_embed
      for block_idx in range(num_cross_layers_input_updater):
        for layer_idx in range(self_to_cross_ratio_input_updater):
          x_cur = horizontal_blocks[block_idx][layer_idx](
              x_cur, padding_mask=padding_mask_cur)
        x_cur = horizontal_blocks[block_idx][-1](
            x_cur, cur_state, padding_mask=padding_mask_cur)
      for layer_idx in range(num_cross_layers_state_updater):
        cur_state = vertical_layers[layer_idx](
            cur_state, x_cur, key_padding_mask=padding_mask_cur)
      return cur_state, None

    # Scan
    cur_state, _ = jax.lax.scan(scan_inner, init_state, x_with_pad, unroll=40)

    assert cur_state.shape == init_state.shape

    encoded = nn.LayerNorm(cur_state, dtype=dtype, name='encoder_norm')

    if classifier:
      encoded = common_layers.classifier_head(
          encoded, num_classes, mlp_dim, pooling_mode='MEAN')
    return encoded


class StatefulTransformerDualEncoder(nn.Module):
  """Stateful Transformer Model Encoder (https://arxiv.org/abs/2205.14794)."""

  def apply(self,
            inputs1,
            inputs2,
            vocab_size,
            inputs1_positions=None,
            inputs2_positions=None,
            inputs1_segmentation=None,
            inputs2_segmentation=None,
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
            num_classes=10,
            tied_weights=False,
            meta_network=False,
            meta_layers=1,
            meta_pool='last',
            use_residual=True,
            meta_partition=3,
            meta_layer_output=False,
            self_to_cross_ratio_input_updater=2,
            num_cross_layers_input_updater=1,
            num_cross_layers_state_updater=1,
            num_state_tokens=20,
            block_size=20,
            use_global_pos_encoding=False,
            interaction=None):
    """Applies Transformer model on the inputs."""
    encoder = StatefulTransformerEncoder.shared(
        vocab_size=vocab_size,
        shared_embedding=shared_embedding,
        use_bfloat16=use_bfloat16,
        emb_dim=emb_dim,
        num_heads=num_heads,
        dtype=dtype,
        num_layers=num_layers,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim,
        max_len=max_len,
        train=train,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        learn_pos_emb=learn_pos_emb,
        classifier=False,
        classifier_pool=classifier_pool,
        num_classes=num_classes,
        tied_weights=tied_weights,
        meta_network=meta_network,
        meta_layers=meta_layers,
        meta_pool=meta_pool,
        use_residual=use_residual,
        meta_partition=meta_partition,
        meta_layer_output=meta_layer_output,
        self_to_cross_ratio_input_updater=self_to_cross_ratio_input_updater,
        num_cross_layers_input_updater=num_cross_layers_input_updater,
        num_cross_layers_state_updater=num_cross_layers_state_updater,
        num_state_tokens=num_state_tokens,
        block_size=block_size,
        use_global_pos_encoding=use_global_pos_encoding,
        name='stateful_encoder')
    inputs1_encoded = encoder(
        inputs=inputs1,
        inputs_positions=inputs1_positions,
        inputs_segmentation=inputs1_segmentation)
    inputs2_encoded = encoder(
        inputs=inputs2,
        inputs_positions=inputs2_positions,
        inputs_segmentation=inputs2_segmentation)

    encoded = common_layers.classifier_head_dual(
        inputs1_encoded,
        inputs2_encoded,
        num_classes,
        mlp_dim,
        pooling_mode='MEAN',
        interaction=interaction)
    return encoded
