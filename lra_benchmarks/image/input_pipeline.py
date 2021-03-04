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
"""Functions to get dataset pipeline for the image cls tasks."""

from lra_benchmarks.data import pathfinder
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.experimental.AUTOTUNE



def get_mnist_datasets(n_devices, batch_size=256, normalize=False):
  """Get MNIST dataset splits."""
  if batch_size % n_devices:
    raise ValueError("Batch size %d isn't divided evenly by n_devices %d" %
                     (batch_size, n_devices))

  train_dataset = tfds.load('mnist', split='train[:90%]')
  val_dataset = tfds.load('mnist', split='train[90%:]')
  test_dataset = tfds.load('mnist', split='test')

  def decode(x):
    decoded = {
        'inputs': tf.cast(x['image'], dtype=tf.int32),
        'targets': x['label']
    }
    if normalize:
      decoded['inputs'] = decoded['inputs'] / 255
    return decoded

  train_dataset = train_dataset.map(decode, num_parallel_calls=AUTOTUNE)
  val_dataset = val_dataset.map(decode, num_parallel_calls=AUTOTUNE)
  test_dataset = test_dataset.map(decode, num_parallel_calls=AUTOTUNE)

  train_dataset = train_dataset.repeat()
  train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
  val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
  test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

  train_dataset = train_dataset.shuffle(
      buffer_size=256, reshuffle_each_iteration=True)

  return train_dataset, val_dataset, test_dataset, 10, 256, (batch_size, 28, 28,
                                                             1)


def get_cifar10_datasets(n_devices, batch_size=256, normalize=False):
  """Get CIFAR-10 dataset splits."""
  if batch_size % n_devices:
    raise ValueError("Batch size %d isn't divided evenly by n_devices %d" %
                     (batch_size, n_devices))

  train_dataset = tfds.load('cifar10', split='train[:90%]')
  val_dataset = tfds.load('cifar10', split='train[90%:]')
  test_dataset = tfds.load('cifar10', split='test')

  def decode(x):
    decoded = {
        'inputs':
            tf.cast(tf.image.rgb_to_grayscale(x['image']), dtype=tf.int32),
        'targets':
            x['label']
    }
    if normalize:
      decoded['inputs'] = decoded['inputs'] / 255
    return decoded

  train_dataset = train_dataset.map(decode, num_parallel_calls=AUTOTUNE)
  val_dataset = val_dataset.map(decode, num_parallel_calls=AUTOTUNE)
  test_dataset = test_dataset.map(decode, num_parallel_calls=AUTOTUNE)

  train_dataset = train_dataset.repeat()
  train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
  val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
  test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

  train_dataset = train_dataset.shuffle(
      buffer_size=256, reshuffle_each_iteration=True)

  return train_dataset, val_dataset, test_dataset, 10, 256, (batch_size, 32, 32,
                                                             1)


def get_pathfinder_orig_datasets(n_devices, batch_size=256, normalize=False):
  """Get Pathfinder dataset splits."""
  if batch_size % n_devices:
    raise ValueError("Batch size %d isn't divided evenly by n_devices %d" %
                     (batch_size, n_devices))

  builder = pathfinder.Pathfinder(data_dir=_PATHFINER_TFDS_PATH)

  def get_split(split):
    ds_p = builder.as_dataset(split=f'positive{split}')
    ds_n = builder.as_dataset(split=f'negetive{split}')
    ds = tf.data.experimental.sample_from_datasets([ds_p, ds_n],
                                                   weights=None,
                                                   seed=None)
    return ds

  train_dataset = get_split('[:80%]')
  val_dataset = get_split('[80%:90%]')
  test_dataset = get_split('[90%:]')

  def decode(x):
    decoded = {
        'inputs':
            tf.cast(tf.image.rgb_to_grayscale(x['image']), dtype=tf.int32),
        'targets':
            x['label']
    }
    if normalize:
      decoded['inputs'] = decoded['inputs'] / 255
    return decoded

  train_dataset = train_dataset.map(decode, num_parallel_calls=AUTOTUNE)
  val_dataset = val_dataset.map(decode, num_parallel_calls=AUTOTUNE)
  test_dataset = test_dataset.map(decode, num_parallel_calls=AUTOTUNE)

  train_dataset = train_dataset.repeat()
  train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
  val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
  test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

  train_dataset = train_dataset.shuffle(
      buffer_size=256, reshuffle_each_iteration=True)

  return train_dataset, val_dataset, test_dataset, 2, 256, (batch_size, 300,
                                                            300, 1)


def get_pathfinder_base_datasets(n_devices,
                                 batch_size=256,
                                 resolution=32,
                                 normalize=False,
                                 split='easy'):
  """Get Pathfinder dataset splits."""
  if batch_size % n_devices:
    raise ValueError("Batch size %d isn't divided evenly by n_devices %d" %
                     (batch_size, n_devices))

  if split not in ['easy', 'intermediate', 'hard']:
    raise ValueError("split must be in ['easy', 'intermediate', 'hard'].")

  if resolution == 32:
    builder = pathfinder.Pathfinder32(data_dir=_PATHFINER_TFDS_PATH)
    inputs_shape = (batch_size, 32, 32, 1)
  elif resolution == 64:
    builder = pathfinder.Pathfinder64(data_dir=_PATHFINER_TFDS_PATH)
    inputs_shape = (batch_size, 64, 64, 1)
  elif resolution == 128:
    builder = pathfinder.Pathfinder128(data_dir=_PATHFINER_TFDS_PATH)
    inputs_shape = (batch_size, 128, 128, 1)
  elif resolution == 256:
    builder = pathfinder.Pathfinder256(data_dir=_PATHFINER_TFDS_PATH)
    inputs_shape = (batch_size, 256, 256, 1)
  else:
    raise ValueError('Resolution must be in [32, 64, 128, 256].')

  def get_split(split):
    ds = builder.as_dataset(
        split=split, decoders={'image': tfds.decode.SkipDecoding()})

    # Filter out examples with empty images:
    ds = ds.filter(lambda x: tf.strings.length((x['image'])) > 0)

    return ds

  train_dataset = get_split(f'{split}[:80%]')
  val_dataset = get_split(f'{split}[80%:90%]')
  test_dataset = get_split(f'{split}[90%:]')

  def decode(x):
    decoded = {
        'inputs': tf.cast(tf.image.decode_png(x['image']), dtype=tf.int32),
        'targets': x['label']
    }
    if normalize:
      decoded['inputs'] = decoded['inputs'] / 255
    return decoded

  train_dataset = train_dataset.map(decode, num_parallel_calls=AUTOTUNE)
  val_dataset = val_dataset.map(decode, num_parallel_calls=AUTOTUNE)
  test_dataset = test_dataset.map(decode, num_parallel_calls=AUTOTUNE)

  train_dataset = train_dataset.repeat()
  train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
  val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
  test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

  train_dataset = train_dataset.shuffle(
      buffer_size=256 * 8, reshuffle_each_iteration=True)

  return train_dataset, val_dataset, test_dataset, 2, 256, inputs_shape
