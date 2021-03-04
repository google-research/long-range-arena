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
"""Base Configuration."""

import ml_collections

NUM_EPOCHS = 200
TRAIN_EXAMPLES = 160000
VALID_EXAMPLES = 20000


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.batch_size = 256
  config.eval_frequency = TRAIN_EXAMPLES // config.batch_size
  config.num_train_steps = (TRAIN_EXAMPLES // config.batch_size) * NUM_EPOCHS
  config.num_eval_steps = VALID_EXAMPLES // config.batch_size
  config.weight_decay = 0.
  config.grad_clip_norm = None

  config.save_checkpoints = True
  config.restore_checkpoints = True
  config.checkpoint_freq = (TRAIN_EXAMPLES //
                            config.batch_size) * NUM_EPOCHS // 2
  config.random_seed = 0

  config.learning_rate = .001
  config.factors = 'constant * linear_warmup * cosine_decay'
  config.warmup = (TRAIN_EXAMPLES // config.batch_size) * 1
  config.steps_per_cycle = (TRAIN_EXAMPLES // config.batch_size) * NUM_EPOCHS

  # model params
  config.model = ml_collections.ConfigDict()
  config.model.emb_dim = 32
  config.model.num_heads = 4
  config.model.num_layers = 2
  config.model.qkv_dim = 32
  config.model.mlp_dim = 32
  config.model.dropout_rate = 0.1
  config.model.attention_dropout_rate = 0.1
  config.model.classifier_pool = 'CLS'
  config.model.learn_pos_emb = True

  config.trial = 0  # dummy for repeated runs.
  return config
