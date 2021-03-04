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
"""Generators for custom listops tasks."""

import csv
import random

from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

flags.DEFINE_string(
    'task', default='basic',
    help='Name of task to create.')
flags.DEFINE_integer(
    'num_train_samples', default=96000,
    help=('Number of train samples.'))
flags.DEFINE_integer(
    'num_valid_samples', default=2000,
    help=('Number of test samples.'))
flags.DEFINE_integer(
    'num_test_samples', default=2000,
    help=('Number of test samples.'))
flags.DEFINE_integer(
    'max_depth', default=10,
    help=('maximum tree depth of training sequences.'))
flags.DEFINE_integer(
    'max_args', default=10,
    help=('maximum number of arguments per operator in training sequences.'))
flags.DEFINE_integer(
    'max_length', default=2000,
    help=('maximum length per sequence in training sequences.'))
flags.DEFINE_integer(
    'min_length', default=500,
    help=('minimum length per sequence in training sequences.'))
flags.DEFINE_string(
    'output_dir', default='output_dir',
    help='Directory to output files.')

FLAGS = flags.FLAGS

MIN = '[MIN'
MAX = '[MAX'
MED = '[MED'
FIRST = '[FIRST'
LAST = '[LAST'
SUM_MOD = '[SM'
END = ']'

OPERATORS = [MIN, MAX, MED, SUM_MOD]  # , FIRST, LAST]
VALUES = range(10)

VALUE_P = 0.25


def generate_tree(depth, max_depth, max_args):
  """Generate tree-like equations.

  Args:
    depth: current depth of the node, int.
    max_depth: maximum depth of the tree, int.
    max_args: maximum number of arguments per operator, int.

  Returns:
    The root node of a tree structure.
  """
  if depth < max_depth:
    r = random.random()
  else:
    r = 1

  if r > VALUE_P:
    value = random.choice(VALUES)
    return value, 1
  else:
    length = 2
    num_values = random.randint(2, max_args)
    values = []
    for _ in range(num_values):
      sub_t, sub_l = generate_tree(depth + 1, max_depth, max_args)
      values.append(sub_t)
      length += sub_l

    op = random.choice(OPERATORS)
    t = (op, values[0])
    for value in values[1:]:
      t = (t, value)
    t = (t, END)
  return t, length


def to_string(t, parens=True):
  if isinstance(t, str):
    return t
  elif isinstance(t, int):
    return str(t)
  else:
    if parens:
      return '( ' + to_string(t[0]) + ' ' + to_string(t[1]) + ' )'


def to_value(t):
  """Compute the output of equation t.

  Args:
    t: a tree structure that represents equation t, list.

  Returns:
    The result of equation t, int.
  """
  if not isinstance(t, tuple):
    return t
  l = to_value(t[0])
  r = to_value(t[1])
  if l in OPERATORS:  # Create an unsaturated function.
    return (l, [r])
  elif r == END:  # l must be an unsaturated function.
    if l[0] == MIN:
      return min(l[1])
    elif l[0] == MAX:
      return max(l[1])
    elif l[0] == FIRST:
      return l[1][0]
    elif l[0] == LAST:
      return l[1][-1]
    elif l[0] == MED:
      return int(np.median(l[1]))
    elif l[0] == SUM_MOD:
      return np.sum(l[1]) % 10
  elif isinstance(l, tuple):
    # We've hit an unsaturated function and an argument.
    return (l[0], l[1] + [r])


def write_to_file(data, fp):
  """Write to file output."""
  tf.logging.info(type(data))
  tf.logging.info('Writing {} samples to {}'.format(len(data), fp + '.tsv'))
  with tf.io.gfile.GFile(fp + '.tsv', 'w+') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['Source', 'Target'])
    writer.writerows(data)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.logging.info('Start dataset construction')

  data = set()
  num_samples = FLAGS.num_train_samples \
      + FLAGS.num_test_samples + FLAGS.num_valid_samples
  while len(data) < num_samples:
    tree, length = generate_tree(1, FLAGS.max_depth, FLAGS.max_args)
    if length > FLAGS.min_length and length < FLAGS.max_length:
      data.add(tree)
      if len(data) % 1000 == 0:
        tf.logging.info('Processed {}'.format(len(data)))
        print('Processed {}'.format(len(data)))
  train = []
  for example in data:
    train.append([to_string(example), to_value(example)])

  tf.logging.info('Finished running dataset construction')

  val = train[FLAGS.num_train_samples:]
  test = val[FLAGS.num_valid_samples:]
  val = val[:FLAGS.num_valid_samples]
  train = train[:FLAGS.num_train_samples]

  tf.logging.info('Dataset size: %d/%d/%d' % (len(train), len(val), len(test)))

  write_to_file(train, FLAGS.output_dir + '/{}_train'.format(FLAGS.task))
  write_to_file(val, FLAGS.output_dir + '/{}_val'.format(FLAGS.task))
  write_to_file(test, FLAGS.output_dir + '/{}_test'.format(FLAGS.task))
  tf.logging.info('Finished writing all to file')


if __name__ == '__main__':
  app.run(main)
