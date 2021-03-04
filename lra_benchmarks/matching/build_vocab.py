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
"""Build vocab and cache it so we don't have to keep running."""
import collections

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

flags.DEFINE_string('vocab_file_path', '/tmp/lra_data/aan',
                    'Path for vocab file output.')

FLAGS = flags.FLAGS
DATASET_PATHS = '/tmp/dataset'


def whitespace_tokenize(text):
  """Splits an input into tokens by whitespace."""
  return text.strip().split()


def build_vocab(datasets,
                special_tokens=(b'<pad>', b'<unk>', b'<s>', b'</s>'),
                min_freq=10,
                text_keys=None):
  """Returns a vocabulary of tokens with optional minimum frequency."""
  # Count the tokens in the datasets.
  logging.info('Building Vocab...')
  counter = collections.Counter()
  num_processed = 0
  for dataset in datasets:
    for example in tfds.as_numpy(dataset):
      # logging.info(example)
      for k in text_keys[:1]:
        # logging.info(example[k])
        counter.update(whitespace_tokenize(example[k][:100]))
      num_processed += 1
      if num_processed % 100 == 0:
        logging.info('Processed %d', num_processed)

  # Add special tokens to the start of vocab.
  vocab = collections.OrderedDict()
  for token in special_tokens:
    vocab[token] = len(vocab)

  # Add all other tokens to the vocab if their frequency is >= min_freq.
  for token in sorted(list(counter.keys())):
    if counter[token] >= min_freq:
      vocab[token] = len(vocab)

  logging.info('Number of unfiltered tokens: %d', len(counter))
  logging.info('Vocabulary size: %d', len(vocab))

  return vocab


def get_tsv_dataset(file_path, batch_size):
  """Preprocess dataset."""
  tf.logging.info(file_path)
  # sel_cols = ['label', 'id1', 'id2']
  col_defaults = [tf.string, tf.string, tf.string, tf.string, tf.string]
  col_names = ['label', 'id1', 'id2', 'text1', 'text2']
  ds = tf.data.experimental.make_csv_dataset([file_path],
                                             batch_size,
                                             column_names=col_names,
                                             column_defaults=col_defaults,
                                             use_quote_delim=False,
                                             field_delim='\t',
                                             shuffle=False,
                                             header=False,
                                             num_epochs=1)
  ds = ds.unbatch()
  return ds


def get_dataset(batch_size):
  """Get dataset from matching datasets converts into src/tgt pairs."""
  train_fps = DATASET_PATHS + '.train.tsv'
  train = get_tsv_dataset(train_fps, batch_size)

  def adapt_example(example):
    return {
        'Source1': example['text1'],
        'Source2': example['text2'],
        'Target': example['label']
    }

  train = train.map(adapt_example)

  train = train.prefetch(tf.data.experimental.AUTOTUNE)

  return train


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train = get_dataset(1)
  logging.info('Building/loading subword tokenizer')
  encoder = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
      (en['Source1'].numpy() for en in train), target_vocab_size=2**13)
  encoder.save_to_file(FLAGS.vocab_file_path)
  logging.info('Saved')


if __name__ == '__main__':
  app.run(main)
