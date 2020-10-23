"""Base Configuration."""

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.batch_size = 32
  config.eval_frequency = 20
  config.num_train_steps = 5000
  config.num_eval_steps = 20
  config.learning_rate = 0.05
  config.weight_decay = 1e-1
  config.max_target_length = 200  # ignored
  config.max_eval_target_length = 200  # ignored
  config.sampling_temperature = 0.6
  config.sampling_top_k = 20
  config.max_predict_token_length = 50
  config.save_checkpoints = True
  config.restore_checkpoints = True
  config.checkpoint_freq = 10000
  config.random_seed = 0
  config.prompt = ""
  config.factors = "constant * linear_warmup * rsqrt_decay"
  config.warmup = 1000
  config.max_length = 2000
  config.tied_weights = True

  config.pooling_mode = "CLS"
  # config.interaction = "NLI"

  config.emb_dim = 512
  config.num_heads = 8
  config.num_layers = 6
  config.qkv_dim = 512
  config.mlp_dim = 2048

  config.trial = 0  # dummy for repeated runs.
  return config
