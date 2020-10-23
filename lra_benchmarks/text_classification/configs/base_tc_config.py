"""Base Configuration."""

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.batch_size = 32
  config.eval_frequency = 100
  config.num_train_steps = 20000
  config.num_eval_steps = -1
  config.learning_rate = 0.05
  config.weight_decay = 1e-1
  config.max_target_length = 200
  config.max_eval_target_length = 200
  config.sampling_temperature = 0.6
  config.sampling_top_k = 20
  config.max_predict_token_length = 50
  config.save_checkpoints = True
  config.restore_checkpoints = True
  config.checkpoint_freq = 10000
  config.random_seed = 0
  config.prompt = ""
  config.factors = "constant * linear_warmup * rsqrt_decay"
  config.warmup = 8000
  config.classifier_pool = "CLS"

  config.max_length = 1000

  config.emb_dim = 256
  config.num_heads = 4
  config.num_layers = 4
  config.qkv_dim = 256
  config.mlp_dim = 1024

  config.trial = 0  # dummy for repeated runs.
  return config
