"""Configuration and hyperparameter sweeps."""

from lra_benchmarks.image.configs.cifar10 import base_cifar10_config


def get_config():
  """Get the hyperparameter configuration."""
  config = base_cifar10_config.get_config()
  config.model_type = "transformer"
  return config


def get_hyper(hyper):
  return hyper.product([])
