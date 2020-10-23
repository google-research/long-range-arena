"""Configuration and hyperparameter sweeps."""

from lra_benchmarks.image.configs.pathfinder128 import base_pathfinder128_config


def get_config():
  """Get the hyperparameter configuration."""
  config = base_pathfinder128_config.get_config()
  config.model_type = "transformer"
  return config


def get_hyper(hyper):
  return hyper.product([])
