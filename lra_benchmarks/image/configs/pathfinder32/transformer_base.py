"""Configuration and hyperparameter sweeps."""

from lra_benchmarks.image.configs.pathfinder32 import base_pathfinder32_config


def get_config():
  """Get the hyperparameter configuration."""
  config = base_pathfinder32_config.get_config()
  config.model_type = "transformer"

  config.model.num_layers = 1
  config.model.num_heads = 4
  config.model.emb_dim = 128
  config.model.dropout_rate = 0.2
  config.model.qkv_dim = config.model.emb_dim // 2
  config.model.mlp_dim = config.model.qkv_dim * 2

  return config


def get_hyper(hyper):
  return hyper.product([])
