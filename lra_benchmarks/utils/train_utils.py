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
"""This contains utility functions for model training and evaluation."""

from flax import nn
from flax.training import common_utils
import jax.numpy as jnp
from lra_benchmarks.models.bigbird import bigbird
from lra_benchmarks.models.linear_transformer import linear_transformer
from lra_benchmarks.models.linformer import linformer
from lra_benchmarks.models.local import local
from lra_benchmarks.models.longformer import longformer
from lra_benchmarks.models.performer import performer
from lra_benchmarks.models.reformer import reformer
from lra_benchmarks.models.sinkhorn_transformer import sinkhorn_transformer
from lra_benchmarks.models.sparse_transformer import sparse_attention
from lra_benchmarks.models.sparse_transformer import sparse_transformer
from lra_benchmarks.models.synthesizer import synthesizer
from lra_benchmarks.models.transformer import transformer
import numpy as onp


def get_model(model_type, create_model_fn, model_kwargs, *create_model_args):
  """Create and initialize the model.

  Args:
    model_type: str; Type of Transformer model to create.
    create_model_fn: fn: Function that is used for creating the model.
    model_kwargs: keyword argument to the model.
    *create_model_args: positional argument to the create_model_args.

  Returns:
    Initialized model.
  """
  if model_type == 'transformer':
    return create_model_fn(transformer.TransformerEncoder, model_kwargs,
                           *create_model_args)
  elif model_type == 'synthesizer':
    return create_model_fn(synthesizer.SynthesizerEncoder, model_kwargs,
                           *create_model_args)
  elif model_type == 'reformer':
    return create_model_fn(reformer.ReformerEncoder, model_kwargs,
                           *create_model_args)
  elif model_type == 'performer':
    return create_model_fn(performer.PerformerEncoder, model_kwargs,
                           *create_model_args)
  elif model_type == 'linformer':
    return create_model_fn(linformer.LinformerEncoder, model_kwargs,
                           *create_model_args)
  elif model_type == 'local':
    return create_model_fn(local.LocalTransformerEncoder, model_kwargs,
                           *create_model_args)
  elif model_type == 'bigbird':
    return create_model_fn(bigbird.BigBirdEncoder, model_kwargs,
                           *create_model_args)
  elif model_type == 'sinkhorn':
    return create_model_fn(sinkhorn_transformer.SinkhornTransformerEncoder,
                           model_kwargs, *create_model_args)
  elif model_type == 'linear_transformer':
    return create_model_fn(linear_transformer.LinearTransformerEncoder,
                           model_kwargs, *create_model_args)
  elif model_type == 'sparse_transformer':
    model_kwargs['attention_patterns'] = [
        sparse_attention.Fixed1Pattern(block_size=50),
        sparse_attention.Fixed2Pattern(block_size=50, c=10)
    ]
    return create_model_fn(sparse_transformer.SparseTransformerEncoder,
                           model_kwargs, *create_model_args)
  elif model_type == 'longformer':
    return create_model_fn(longformer.LongformerEncoder, model_kwargs,
                           *create_model_args)
  else:
    raise ValueError('Model type not supported')


def create_learning_rate_scheduler(
    factors='constant * linear_warmup * rsqrt_decay',
    base_learning_rate=0.5,
    warmup_steps=1000,
    decay_factor=0.5,
    steps_per_decay=20000,
    steps_per_cycle=100000):
  """Creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * rsqrt_normalized_decay: divide by square root of max(step/warmup_steps, 1)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

  Args:
    factors: string, factors separated by '*' that defines the schedule.
    base_learning_rate: float, the starting constant for the lr schedule.
    warmup_steps: int, how many steps to warm up for in the warmup schedule.
    decay_factor: float, the amount to decay the learning rate by.
    steps_per_decay: int, how often to decay the learning rate.
    steps_per_cycle: int, steps per cycle when using cosine decay.

  Returns:
    a function learning_rate(step): float -> {'learning_rate': float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split('*')]

  def step_fn(step):
    """Step to learning rate function."""
    ret = 1.0
    for name in factors:
      if name == 'constant':
        ret *= base_learning_rate
      elif name == 'linear_warmup':
        ret *= jnp.minimum(1.0, step / warmup_steps)
      elif name == 'rsqrt_decay':
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'rsqrt_normalized_decay':
        ret *= jnp.sqrt(warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'decay_every':
        ret *= (decay_factor**(step // steps_per_decay))
      elif name == 'cosine_decay':
        progress = jnp.maximum(0.0,
                               (step - warmup_steps) / float(steps_per_cycle))
        ret *= jnp.maximum(0.0,
                           0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
      else:
        raise ValueError('Unknown factor %s.' % name)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn


def compute_weighted_cross_entropy(logits, targets, num_classes, weights=None):
  """Compute weighted cross entropy and entropy for log probs and targets.

  Args:
   logits: [batch, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   num_classes: int, num classes of problem.
   weights: None or array of shape [batch x length]

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  onehot_targets = common_utils.onehot(targets, num_classes)
  loss = -jnp.sum(onehot_targets * nn.log_softmax(logits), axis=-1)
  normalizing_factor = onehot_targets.sum()
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def compute_weighted_accuracy(logits, targets, weights=None):
  """Compute weighted accuracy for log probs and targets.

  Args:
   logits: [batch, num_classes] float array.
   targets: categorical targets [batch] int array.
   weights: None or array of shape [batch]

  Returns:
    Tuple of scalar accuracy and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  loss = jnp.equal(jnp.argmax(logits, axis=-1), targets)
  normalizing_factor = onp.prod(logits.shape[:-1])
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor
