"""Configuration and hyperparameter sweeps."""

from lra_benchmarks.matching.configs import base_match_config


def get_config():
  """Get the default hyperparameter configuration."""
  config = base_match_config.get_config()
  config.model_type = "transformer"
  return config


def get_hyper(hyper):
  return hyper.product([])
