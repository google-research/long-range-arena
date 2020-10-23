"""Configuration and hyperparameter sweeps."""

from lra_benchmarks.text_classification.configs import base_tc_config


def get_config():
  """Get the default hyperparameter configuration."""
  config = base_tc_config.get_config()
  config.model_type = "transformer"
  return config


def get_hyper(hyper):
  return hyper.product([])
