# Long-Range Arena

Repairing the LRA repo... A work in progress.

## Description

Long-range arena is an effort toward systematic evaluation of efficient
transformer models. The project aims at establishing benchmark tasks/dtasets
using which we can evaluate transformer-based models in a systematic way, by
assessing their generalization power, computational efficiency, memory
foot-print, etc.

Long-range arena implements different variants of Transformers using 
[JAX](https://github.com/google/jax) and [Flax](https://github.com/google/flax).

## Installation

We recommend installing the python dependencies using a virtual environment such as venv, pipenv, or miniconda.
After the virtual environment is activated, install pip3, then run: 
```
pip3 install --upgrade pip;

### CPU
pip3 install -e . \
    -f https://storage.googleapis.com/jax-releases/jax_releases.html;

### TPU
pip3 install -e '.[tpu]' \
    -f https://storage.googleapis.com/jax-releases/jax_releases.html \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html;
```


## Dataset Setup

To prepare the data, run:

```
source ./dataset_setup.sh;
```

## Example Usage

To run a task, run the train.py file in the corresponding task directory.
For example:

```
python3 lra_benchmarks/listops/train.py \
      --config=lra_benchmarks/listops/configs/transformer_base.py \
      --model_dir=/tmp/listops \
      --task_name=basic \
      --data_dir=./lra_data/listops/ \
      --config.checkpoint_freq=100 \
      --config.num_eval_steps=100 
```
