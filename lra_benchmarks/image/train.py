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
"""Main training script for the image classification task."""
import functools
import itertools
import json
import os
import time

from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax import nn
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils
import jax
from jax import random
import jax.nn
import jax.numpy as jnp
from lra_benchmarks.image import task_registry
from lra_benchmarks.models.transformer import transformer
from lra_benchmarks.utils import train_utils
from ml_collections import config_flags
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=True)
flags.DEFINE_string(
    'model_dir', default=None, help='Directory to store model data.')
flags.DEFINE_string('task_name', default='mnist', help='Name of the task')
flags.DEFINE_bool(
    'eval_only', default=False, help='Run the evaluation on the test data.')


def create_model(key, flax_module, input_shape, model_kwargs):
  """Creates and initializes the model."""

  @functools.partial(jax.jit, backend='cpu')
  def _create_model(key):
    module = flax_module.partial(**model_kwargs)
    with nn.stateful() as init_state:
      with nn.stochastic(key):
        _, initial_params = module.init_by_shape(key,
                                                 [(input_shape, jnp.float32)])
        model = nn.Model(module, initial_params)
    return model, init_state

  return _create_model(key)


def create_optimizer(model, learning_rate, weight_decay):
  optimizer_def = optim.Adam(
      learning_rate, beta1=0.9, beta2=0.98, eps=1e-9, weight_decay=weight_decay)
  optimizer = optimizer_def.create(model)
  return optimizer


def compute_metrics(logits, labels, num_classes, weights):
  """Compute summary metrics."""
  loss, weight_sum = train_utils.compute_weighted_cross_entropy(
      logits, labels, num_classes, weights=weights)
  acc, _ = train_utils.compute_weighted_accuracy(logits, labels, weights)
  metrics = {
      'loss': loss,
      'accuracy': acc,
      'denominator': weight_sum,
  }
  metrics = jax.lax.psum(metrics, 'batch')
  return metrics


def get_model(init_rng, input_shape, model_type, model_kwargs):
  """Create and initialize the model.

  Args:
    init_rng: float; Jax PRNG key.
    input_shape: tuple; Tuple indicating input shape.
    model_type: str; Type of Transformer model to create.
    model_kwargs: keyword argument to the model.

  Returns:
    Initialized model.
  """
  if model_type == 'transformer':
    return create_model(init_rng, transformer.TransformerEncoder, input_shape,
                        model_kwargs)
  else:
    raise ValueError('Model type not supported')


def train_step(optimizer,
               state,
               batch,
               learning_rate_fn,
               num_classes,
               flatten_input=True,
               grad_clip_norm=None,
               dropout_rng=None):
  """Perform a single training step."""
  train_keys = ['inputs', 'targets']
  (inputs, targets) = [batch.get(k, None) for k in train_keys]
  if flatten_input:
    inputs = inputs.reshape(inputs.shape[0], -1)

  # We handle PRNG splitting inside the top pmap, rather
  # than handling it outside in the training loop - doing the
  # latter can add some stalls to the devices.
  dropout_rng, new_dropout_rng = random.split(dropout_rng)

  def loss_fn(model):
    """Loss function used for training."""
    with nn.stateful(state) as new_state:
      with nn.stochastic(dropout_rng):
        logits = model(inputs, train=True)
    loss, weight_sum = train_utils.compute_weighted_cross_entropy(
        logits, targets, num_classes=num_classes, weights=None)
    mean_loss = loss / weight_sum
    return mean_loss, (new_state, logits)

  step = optimizer.state.step
  lr = learning_rate_fn(step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, (new_state, logits)), grad = grad_fn(optimizer.target)
  grad = jax.lax.pmean(grad, 'batch')
  if grad_clip_norm:
    # Optionally resize the global gradient to a maximum norm.
    gradients, _ = jax.tree_flatten(grad)
    g_l2 = jnp.sqrt(sum([jnp.vdot(p, p) for p in gradients]))
    g_factor = jnp.minimum(1.0, grad_clip_norm / g_l2)
    grad = jax.tree_map(lambda p: g_factor * p, grad)
  new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)
  metrics = compute_metrics(logits, targets, num_classes, weights=None)
  metrics['learning_rate'] = lr
  return new_optimizer, new_state, metrics, new_dropout_rng


def eval_step(model, state, batch, num_classes, flatten_input=True):
  eval_keys = ['inputs', 'targets']
  (inputs, targets) = [batch.get(k, None) for k in eval_keys]
  if flatten_input:
    inputs = inputs.reshape(inputs.shape[0], -1)
  if jax.tree_leaves(state):
    state = jax.lax.pmean(state, 'batch')
  with nn.stateful(state, mutable=False):
    logits = model(inputs, train=False)
  return compute_metrics(logits, targets, num_classes, weights=None)


def test(optimizer, state, p_eval_step, step, test_ds, summary_writer,
         model_dir):
  """Test the flax module in optimizer on test_ds.

  Args:
    optimizer: flax optimizer (contains flax module).
    state: model state, e.g. batch statistics.
    p_eval_step: fn; Pmapped evaluation step function.
    step: int; Number of training steps passed so far.
    test_ds: tf.dataset; Test dataset.
    summary_writer: tensorflow summary writer.
    model_dir: model directory.
  """
  # Test Metrics
  test_metrics = []
  test_iter = iter(test_ds)
  for _, test_batch in zip(itertools.repeat(1), test_iter):
    # pylint: disable=protected-access
    test_batch = common_utils.shard(
        jax.tree_map(lambda x: x._numpy(), test_batch))
    # pylint: enable=protected-access
    metrics = p_eval_step(optimizer.target, state, test_batch)
    test_metrics.append(metrics)
  test_metrics = common_utils.get_metrics(test_metrics)
  test_metrics_sums = jax.tree_map(jnp.sum, test_metrics)
  test_denominator = test_metrics_sums.pop('denominator')
  test_summary = jax.tree_map(
      lambda x: x / test_denominator,  # pylint: disable=cell-var-from-loop
      test_metrics_sums)
  logging.info('test in step: %d, loss: %.4f, acc: %.4f', step,
               test_summary['loss'], test_summary['accuracy'])
  if jax.host_id() == 0:
    for key, val in test_summary.items():
      summary_writer.scalar(f'test_{key}', val, step)
    summary_writer.flush()
  with tf.io.gfile.GFile(os.path.join(model_dir, 'results.json'), 'w') as f:
    json.dump(jax.tree_map(lambda x: x.tolist(), test_summary), f)


def train_loop(config, dropout_rngs, eval_ds, eval_freq, num_eval_steps,
               num_train_steps, optimizer, state, p_eval_step, p_train_step,
               start_step, train_iter, summary_writer):
  """Training loop.

  Args:
    config: experiment config.
    dropout_rngs: float array; Jax PRNG key.
    eval_ds: tf.dataset; Evaluation dataset.
    eval_freq: int; Evaluation frequency;
    num_eval_steps: int; Number of evaluation steps.
    num_train_steps: int; Number of training steps.
    optimizer: flax optimizer.
    state: model state, e.g. batch statistics.
    p_eval_step: fn; Pmapped evaluation step function.
    p_train_step: fn; Pmapped train step function.
    start_step: int; global training step.
    train_iter: iter(tf.dataset); Training data iterator.
    summary_writer: tensorflow summary writer.

  Returns:
    optimizer, global training step
  """
  metrics_all = []
  tick = time.time()
  logging.info('Starting training')
  logging.info('====================')

  step = 0
  for step, batch in zip(range(start_step, num_train_steps), train_iter):
    batch = common_utils.shard(jax.tree_map(lambda x: x._numpy(), batch))  # pylint: disable=protected-access
    optimizer, state, metrics, dropout_rngs = p_train_step(
        optimizer, state, batch, dropout_rng=dropout_rngs)
    metrics_all.append(metrics)
    # Save a Checkpoint
    if ((step % config.checkpoint_freq == 0 and step > 0) or
        step == num_train_steps - 1):
      if jax.host_id() == 0 and config.save_checkpoints:
        # Save unreplicated optimizer + model state.
        checkpoints.save_checkpoint(
            FLAGS.model_dir,
            (jax_utils.unreplicate(optimizer), jax_utils.unreplicate(state)),
            step)

    # Periodic metric handling.
    if step % eval_freq == 0 and step > 0:
      metrics_all = common_utils.get_metrics(metrics_all)
      lr = metrics_all.pop('learning_rate').mean()
      metrics_sums = jax.tree_map(jnp.sum, metrics_all)
      denominator = metrics_sums.pop('denominator')
      summary = jax.tree_map(lambda x: x / denominator, metrics_sums)  # pylint: disable=cell-var-from-loop
      summary['learning_rate'] = lr
      # Calculate (clipped) perplexity after averaging log-perplexities:
      logging.info('train in step: %d, loss: %.4f, acc: %.4f', step,
                   summary['loss'], summary['accuracy'])
      if jax.host_id() == 0:
        tock = time.time()
        steps_per_sec = eval_freq / (tock - tick)
        tick = tock
        summary_writer.scalar('examples_per_second',
                              steps_per_sec * config.batch_size, step)
        for key, val in summary.items():
          summary_writer.scalar(f'train_{key}', val, step)
        summary_writer.flush()
      # Reset metric accumulation for next evaluation cycle.
      metrics_all = []

      # Eval Metrics
      eval_metrics = []
      eval_iter = iter(eval_ds)
      if num_eval_steps == -1:
        num_iter = itertools.repeat(1)
      else:
        num_iter = range(num_eval_steps)
      for _, eval_batch in zip(num_iter, eval_iter):
        # pylint: disable=protected-access
        eval_batch = common_utils.shard(
            jax.tree_map(lambda x: x._numpy(), eval_batch))
        # pylint: enable=protected-access
        metrics = p_eval_step(optimizer.target, state, eval_batch)
        eval_metrics.append(metrics)
      eval_metrics = common_utils.get_metrics(eval_metrics)
      eval_metrics_sums = jax.tree_map(jnp.sum, eval_metrics)
      eval_denominator = eval_metrics_sums.pop('denominator')
      eval_summary = jax.tree_map(
          lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
          eval_metrics_sums)
      logging.info('eval in step: %d, loss: %.4f, acc: %.4f', step,
                   eval_summary['loss'], eval_summary['accuracy'])
      if jax.host_id() == 0:
        for key, val in eval_summary.items():
          summary_writer.scalar(f'val_{key}', val, step)
        summary_writer.flush()
  return optimizer, state, step


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.enable_v2_behavior()

  config = FLAGS.config
  logging.info('===========Config Dict============')
  logging.info(config)
  batch_size = config.batch_size
  learning_rate = config.learning_rate
  num_train_steps = config.num_train_steps
  num_eval_steps = config.num_eval_steps
  eval_freq = config.eval_frequency
  random_seed = config.random_seed
  model_type = config.model_type

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(
        os.path.join(FLAGS.model_dir, 'summary'))
  else:
    summary_writer = None

  if batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')

  logging.info('Training on %s', FLAGS.task_name)

  if model_type in ['wideresnet', 'resnet', 'simple_cnn']:
    normalize = True
  else:  # transformer-based models
    normalize = False
  (train_ds, eval_ds, test_ds, num_classes, vocab_size,
   input_shape) = task_registry.TASK_DATA_DICT[FLAGS.task_name](
       n_devices=jax.local_device_count(),
       batch_size=batch_size,
       normalize=normalize)
  train_iter = iter(train_ds)
  model_kwargs = {}
  flatten_input = True

  if model_type in ['wideresnet', 'resnet', 'simple_cnn']:
    model_kwargs.update({
        'num_classes': num_classes,
    })
    flatten_input = False

  else:  # transformer models
    # we will flatten the input
    bs, h, w, c = input_shape
    assert c == 1
    input_shape = (bs, h * w * c)
    model_kwargs.update({
        'vocab_size': vocab_size,
        'max_len': input_shape[1],
        'classifier': True,
        'num_classes': num_classes,
    })

  model_kwargs.update(config.model)

  rng = random.PRNGKey(random_seed)
  rng = jax.random.fold_in(rng, jax.host_id())
  rng, init_rng = random.split(rng)
  # We init the first set of dropout PRNG keys, but update it afterwards inside
  # the main pmap'd training update for performance.
  dropout_rngs = random.split(rng, jax.local_device_count())

  model, state = get_model(init_rng, input_shape, model_type, model_kwargs)

  optimizer = create_optimizer(model, learning_rate, config.weight_decay)
  del model  # Don't keep a copy of the initial model.

  start_step = 0
  if config.restore_checkpoints:
    # Restore unreplicated optimizer + model state from last checkpoint.
    optimizer, state = checkpoints.restore_checkpoint(FLAGS.model_dir,
                                                      (optimizer, state))
    # Grab last step.
    start_step = int(optimizer.state.step)

  # Replicate optimizer and state
  optimizer = jax_utils.replicate(optimizer)
  state = jax_utils.replicate(state)

  learning_rate_fn = train_utils.create_learning_rate_scheduler(
      factors=config.factors,
      base_learning_rate=learning_rate,
      warmup_steps=config.warmup,
      steps_per_cycle=config.get('steps_per_cycle', None),
  )
  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          learning_rate_fn=learning_rate_fn,
          num_classes=num_classes,
          grad_clip_norm=config.get('grad_clip_norm', None),
          flatten_input=flatten_input),
      axis_name='batch')

  p_eval_step = jax.pmap(
      functools.partial(
          eval_step, num_classes=num_classes, flatten_input=flatten_input),
      axis_name='batch',
  )

  optimizer, state, step = train_loop(config, dropout_rngs, eval_ds, eval_freq,
                                      num_eval_steps, num_train_steps,
                                      optimizer, state, p_eval_step,
                                      p_train_step, start_step, train_iter,
                                      summary_writer)

  logging.info('Starting testing')
  logging.info('====================')
  test(optimizer, state, p_eval_step, step, test_ds, summary_writer,
       FLAGS.model_dir)


if __name__ == '__main__':
  app.run(main)
