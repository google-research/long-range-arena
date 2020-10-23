"""Configuration and hyperparameter sweeps."""

from lra_benchmarks.listops.configs import base_listops_config


def get_config():
  """Get the default hyperparameter configuration."""
  config = base_listops_config.get_config()
  config.model_type = "transformer"
  return config


def get_hyper(hyper):
  return hyper.product([])
