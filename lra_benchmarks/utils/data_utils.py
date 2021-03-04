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
"""This script contains utility functions for data preprocessing and output post-processing."""

from collections import defaultdict  ## pylint: disable=g-importing-member
import tempfile
import time

from absl import logging
import jax
import nltk

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_text as tftxt

from sentencepiece import SentencePieceTrainer

PAD_TOKEN = {"index": 0, "token": "<pad>"}
UNK_TOKEN = {"index": 1, "token": "<unk>"}
EOS_TOKEN = {"index": 2, "token": "<eos>"}
BOS_TOKEN = {"index": 3, "token": "<bos>"}

nltk.download("punkt")


def filter_non_ascii(s):
  """Filter non-ascii characters from a string."""
  if isinstance(s, bytes):
    s = s.decode("utf-8")
  return s.encode("ascii", errors="ignore").decode("utf-8")


def nltk_tokenize(s: str):
  """Tokenize a string to a sequence of tokens with nltk tokenizer.

  Args:
    s: str: input string.

  Returns:
    A tokenized string.
  """
  return nltk.word_tokenize(s)


def add_bos_token(s: str):
  return BOS_TOKEN["token"] + " " + s


def add_eos_token(s: str):
  return s + " " + EOS_TOKEN["token"]


def build_vocab(train_dataset, fields, vocab_size=None, min_freq=0):
  """Build word vocab from the train dataset.

  Assume the dataset has been preprocessed , tokenized, lowercased properly.
  Args:
    train_dataset: tf.Dataset: the training dataset.
    fields: List[str]: the data fields for building the vocab.
    vocab_size: None or int.
    min_freq: int: minimum token frequency to be kept in the vocab.

  Returns:
    A vocab dictionary.
  """
  vocab = {
      PAD_TOKEN["token"]: PAD_TOKEN["index"],
      UNK_TOKEN["token"]: UNK_TOKEN["index"],
      BOS_TOKEN["token"]: BOS_TOKEN["index"],
      EOS_TOKEN["token"]: EOS_TOKEN["index"]
  }
  word_freqs = defaultdict(int)
  for example in train_dataset:
    for field in fields:
      s = example[field].numpy().decode("utf-8")
      for token in s.split():
        word_freqs[token] += 1
  # filter vocab by min_freq and vocab size.
  sorted_word_freqs = sorted(
      word_freqs.items(), key=lambda item: item[1], reverse=True)
  if vocab_size:
    sorted_word_freqs = sorted_word_freqs[:vocab_size]
  for (token, freq) in sorted_word_freqs:
    if freq >= min_freq:
      if token not in vocab:
        vocab[token] = len(vocab)
  logging.info("Vocab size: before filtering (%d), after(%d)", len(word_freqs),
               len(vocab))
  # logging.info("Top 10 frequent tokens: ", sorted_word_freqs[:10])
  # logging.info("Bottom 10 frequent tokens: ", sorted_word_freqs[-10:])
  return vocab


# -----------------------------------------------------------------------------
# Train and Load SentencePiece Tokenizer.
# -----------------------------------------------------------------------------
def dump_chars_to_textfile(dataset,
                           maxchars=1e9,
                           data_keys=("inputs", "targets")):
  """Write part of a TFDS sentence dataset to lines in a text file.

  Args:
    dataset: tf.dataset containing string-data.
    maxchars: int: approximate number of characters to save from dataset.
    data_keys: Tuple[str]: what keys in dataset to dump from.

  Returns:
    name of temp file with dataset bytes, exact number of characters dumped.
  """
  char_count = 0
  processed_examples = 0
  ds_iter = dataset.as_numpy_iterator()
  with tempfile.NamedTemporaryFile(
      delete=False, prefix="/tmp/ds_chars") as outfp:
    while char_count < maxchars:
      example = next(ds_iter, None)
      processed_examples += 1
      if example is None:
        break
      for k in data_keys:
        line = example[k] + b"\n"
        char_count += len(line)
        outfp.write(line)
  logging.info("%d examples processed for training sentencepiece tokenizer.",
               processed_examples)
  return outfp.name, char_count


def train_sentencepiece(dataset,
                        vocab_size,
                        maxchars=1e9,
                        character_coverage=1.0,
                        model_path="model",
                        model_type="unigram",
                        data_keys=("inputs", "targets")):
  """Train SentencePiece tokenizer from subset of tf dataset.

  Args:
    dataset: tf.dataset
    vocab_size: int: size of vocab tokens to train.
    maxchars: int: number of characters to use for sentencepiece training.
    character_coverage: amount of characters covered by the model, good defaults
      are 0.9995 for languages with rich character set like Japanese or Chinese
      and 1.0 for other languages with small character set.
    model_path: str: path of model file to save vocab model to.
    model_type: str: type of sentencepiece vocab to train.
    data_keys: Tuple[str]: keys of dataset to use for training.

  Returns:
    path to the trained sentencepiece vocabulary model.
  """
  fname, _ = dump_chars_to_textfile(
      dataset, maxchars=maxchars, data_keys=data_keys)
  with tempfile.NamedTemporaryFile(
      delete=False, prefix="/tmp/sp_tmp") as model_fp:
    pass  # we just want a prefix'd tmp-filename
  argstr = " ".join([
      f"--input={fname}", f"--vocab_size={vocab_size}",
      f"--character_coverage={character_coverage}",
      f"--model_prefix={model_fp.name}", f"--model_type={model_type}"
  ])
  SentencePieceTrainer.Train(argstr)
  if jax.host_id() == 0:
    # Use an intermediate filename that is renamed to the target name to address
    # create and fill delays.
    copy_rename_path = model_path + ".rntmp"
    tf.io.gfile.copy(model_fp.name + ".model", copy_rename_path, overwrite=True)
    tf.io.gfile.rename(copy_rename_path, model_path, overwrite=True)
    tf.io.gfile.copy(
        model_fp.name + ".vocab", copy_rename_path + ".vocab", overwrite=True)
    tf.io.gfile.rename(
        copy_rename_path + ".vocab", model_path + ".vocab", overwrite=True)
    logging.info("copied %s to %s", model_fp.name + ".model", model_path)
  else:
    while not tf.io.gfile.exists(model_path):
      time.sleep(1)
    time.sleep(1)
  return model_path


def load_sentencepiece_tokenizer(model_path,
                                 add_bos=True,
                                 add_eos=True,
                                 reverse=False):
  """Load a tf-text SentencePiece tokenizer from given model filepath."""
  with tf.io.gfile.GFile(model_path, "rb") as model_fp:
    sp_model = model_fp.read()
  sp_tokenizer = tftxt.SentencepieceTokenizer(
      model=sp_model, add_bos=add_bos, add_eos=add_eos, reverse=reverse)
  return sp_tokenizer


def load_tfds_dataset(data_dir, dataset_name, split, shuffle=True):
  """Return train and evaluation datasets, feature info and supervised keys.

  Args:
    data_dir: directory where the data is located.
    dataset_name: a string, the name of the TFDS dataset.
    split: string: the split of the dataset, e.g., {train, validation, test}
    shuffle: Boolean determining whether or not to shuffle the train files at
      startup. Set to False if you want data determinism.

  Returns:
    a 3-tuple consisting of:
     * the train tf.Dataset
     * information about features: a python dictionary with feature names
         as keys and an object as value that provides .shape and .n_classes.
     * supervised_keys: information what's the input and what's the target,
         ie., a pair of lists with input and target feature names.
  """
  dataset_builder = tfds.builder(dataset_name, data_dir=data_dir)
  info = dataset_builder.info
  splits = dataset_builder.info.splits
  if split not in splits:
    raise ValueError(
        f"{split} not exists in the dataset {data_dir}/{dataset_name}/{splits}."
    )
  dataset = tfds.load(
      name=dataset_name, split=split, data_dir=data_dir, shuffle_files=shuffle)
  keys = None
  if info.supervised_keys:
    keys = info.supervised_keys
  return dataset, info.features, keys


def bin_and_batch(
    dataset,
    length_fn,  # the length function of an input sample
    training,
    n_devices,
    target_batch_size=256,
    target_bucket_length=32,
    buckets=None,
    max_eval_length=None,
    drop_remainder=False):
  """Batching function, can specify batch size directly or per-device.

  Args:
    dataset: tf dataset containing individual sequences.
    length_fn: a function to determine the sample length.
    training: bool: is this a train or eval dataset.
    n_devices: number of devices this dataset will be run on.
    target_batch_size: int: the target batch size for binned batches.
    target_bucket_length: int: the target sequence length for binned batches.
    buckets: (List[int], List[int]): manually specified length buckets and batch
      sizes for bins.
    max_eval_length: int: for eval set allow a extra long-sequence bin.
    drop_remainder: bool: if true drop last batch if not divisible by batch
      sizes. (e.g. not divisible by n_devices).

  Returns:
    Dynamically binned batches of sequence that roughly keep the total
    number of tokens (target_batch_size * target_bucket_length) the same, while
    insuring batch sizes are divisible by n_devices for distributed training.
  """
  # Create heuristic buckets is none are specified.
  if buckets is None:
    logging.info("Heuristically bucketing based on shapes of examples.")
    bucket_boundaries = [
        target_bucket_length // 4, target_bucket_length // 2,
        target_bucket_length, target_bucket_length * 2,
        target_bucket_length * 4, target_bucket_length * 8,
        target_bucket_length * 16
    ]
    bucket_batch_sizes = [
        target_batch_size * 4, target_batch_size * 2, target_batch_size,
        target_batch_size // 2, target_batch_size // 4, target_batch_size // 8,
        target_batch_size // 16
    ]
    # allow for different evaluation max-length bucket and batchsize
    if not training:
      max_eval_length = max_eval_length or target_bucket_length * 32
      bucket_boundaries[-1] = max_eval_length
      bucket_batch_sizes[-1] = (
          target_batch_size // (max_eval_length // target_bucket_length))
    # We will pad to boundaries which pads to bucket_boundary-1: add 1 here.
    bucket_boundaries = [b + 1 for b in bucket_boundaries]
    # Make batch sizes divisible by n_devices.
    bucket_batch_sizes = [
        max(b // n_devices, 1) * n_devices for b in bucket_batch_sizes
    ]
    buckets = (bucket_boundaries, bucket_batch_sizes)

  logging.info("Bucketing with buckets %s.", str(buckets))

  boundaries, batch_sizes = buckets
  # bucket_by_sequence_length expects a final dummy 1 batch_size
  batch_sizes.append(1)
  dataset = dataset.apply(
      tf.data.experimental.bucket_by_sequence_length(
          length_fn,
          boundaries,
          batch_sizes,
          pad_to_bucket_boundary=True,
          drop_remainder=drop_remainder))
  return dataset
