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
"""Input pipeline for the imdb dataset."""

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.experimental.AUTOTUNE


def preprocess_dataset(file_path, batch_size):
  """Preprocess dataset."""
  tf.logging.info(file_path)
  sel_cols = ['Source', 'Target']
  col_defaults = [tf.string, tf.int32]
  ds = tf.data.experimental.make_csv_dataset([file_path],
                                             batch_size,
                                             column_defaults=col_defaults,
                                             select_columns=sel_cols,
                                             field_delim=',',
                                             header=True,
                                             shuffle=False,
                                             num_epochs=1)
  ds = ds.unbatch()
  return ds


def get_imdb_dataset():
  """Get dataset from  imdb tfds. converts into src/tgt pairs."""
  data = tfds.load('imdb_reviews')
  train_raw = data['train']
  valid_raw = data['test']
  test_raw = data['test']
  # use test set for validation because IMDb doesn't have val set.
  # Print an example.
  logging.info('Data sample: %s', next(iter(tfds.as_numpy(train_raw.skip(4)))))

  def adapt_example(example):
    return {'Source': example['text'], 'Target': example['label']}

  train = train_raw.map(adapt_example)
  valid = valid_raw.map(adapt_example)
  test = test_raw.map(adapt_example)

  return train, valid, test


def get_yelp_dataset():
  """Get dataset from yelp tfds. converts into src/tgt pairs."""
  data = tfds.load('yelp_polarity_reviews')
  train_raw = data['train']
  valid_raw = data['test']
  test_raw = data['test']
  # use test set for validation because yelp doesn't have val set.
  # Print an example.
  logging.info('Data sample: %s', next(iter(tfds.as_numpy(train_raw.skip(4)))))

  def adapt_example(example):
    return {'Source': example['text'], 'Target': example['label']}

  train = train_raw.map(adapt_example)
  valid = valid_raw.map(adapt_example)
  test = test_raw.map(adapt_example)

  return train, valid, test


def get_agnews_dataset():
  """Get dataset from  agnews tfds. converts into src/tgt pairs."""
  data = tfds.load('ag_news_subset')
  train_raw = data['train']
  valid_raw = data['test']
  test_raw = data['test']
  # use test set for validation because agnews doesn't have val set.
  # Print an example.
  logging.info('Data sample: %s', next(iter(tfds.as_numpy(train_raw.skip(4)))))

  def adapt_example(example):
    return {'Source': example['description'], 'Target': example['label']}

  train = train_raw.map(adapt_example)
  valid = valid_raw.map(adapt_example)
  test = test_raw.map(adapt_example)

  return train, valid, test


def get_tc_datasets(n_devices,
                    task_name,
                    data_dir=None,
                    batch_size=256,
                    fixed_vocab=None,
                    max_length=512,
                    tokenizer='char'):
  """Get text classification datasets."""
  if batch_size % n_devices:
    raise ValueError("Batch size %d isn't divided evenly by n_devices %d" %
                     (batch_size, n_devices))

  if task_name == 'imdb_reviews':
    train_dataset, val_dataset, test_dataset = get_imdb_dataset()
  elif task_name == 'yelp_reviews':
    train_dataset, val_dataset, test_dataset = get_yelp_dataset()
  elif task_name == 'agnews':
    train_dataset, val_dataset, test_dataset = get_agnews_dataset()
  else:
    train_path = data_dir + task_name + '_train.tsv'
    val_path = data_dir + task_name + '_val.tsv'
    test_path = data_dir + task_name + '_test.tsv'

    train_dataset = preprocess_dataset(train_path, batch_size)
    val_dataset = preprocess_dataset(val_path, batch_size)
    test_dataset = preprocess_dataset(test_path, batch_size)

  tf.logging.info('Finished preprocessing')

  tf.logging.info(val_dataset)

  if tokenizer == 'char':
    logging.info('Using char/byte level vocab')
    encoder = tfds.deprecated.text.ByteTextEncoder()
  else:
    if fixed_vocab is None:
      tf.logging.info('Building vocab')
      # build vocab
      vocab_set = set()
      tokenizer = tfds.deprecated.text.Tokenizer()
      for i, data in enumerate(train_dataset):
        examples = data['Source']
        examples = tokenizer.tokenize(examples.numpy())
        examples = np.reshape(examples, (-1)).tolist()
        vocab_set.update(examples)
        if i % 1000 == 0:
          tf.logging.info('Processed {}'.format(i))
      tf.logging.info(len(vocab_set))
      vocab_set = list(set(vocab_set))
      tf.logging.info('Finished processing vocab size={}'.format(
          len(vocab_set)))
    else:
      vocab_set = list(set(fixed_vocab))
    encoder = tfds.deprecated.text.TokenTextEncoder(vocab_set)

  def tf_encode(x):
    result = tf.py_function(lambda s: tf.constant(encoder.encode(s.numpy())), [
        x,
    ], tf.int32)
    result.set_shape([None])
    return result

  def tokenize(d):
    return {
        'inputs': tf_encode(d['Source'])[:max_length],
        'targets': d['Target']
    }

  train_dataset = train_dataset.map(tokenize, num_parallel_calls=AUTOTUNE)
  val_dataset = val_dataset.map(tokenize, num_parallel_calls=AUTOTUNE)
  test_dataset = test_dataset.map(tokenize, num_parallel_calls=AUTOTUNE)

  max_shape = {'inputs': [max_length], 'targets': []}
  train_dataset = train_dataset.shuffle(
      buffer_size=256, reshuffle_each_iteration=True).padded_batch(
          batch_size, padded_shapes=max_shape)
  val_dataset = val_dataset.padded_batch(batch_size, padded_shapes=max_shape)
  test_dataset = test_dataset.padded_batch(batch_size, padded_shapes=max_shape)

  return train_dataset, val_dataset, test_dataset, encoder
