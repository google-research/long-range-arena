"""Generators for preparing arxiv text classification dataset."""

import csv
import random

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf



flags.DEFINE_string(
    'output_dir', default=ARXIV_TC_PATH, help='Directory to output files.')

flags.DEFINE_integer('class_id', default=0, help=('which class id to merge'))

flags.DEFINE_string(
    'operation',
    default='generate',
    help='Type of operation: "generate" or "merge".')

FLAGS = flags.FLAGS


def generate(classes):
  """Used for generating the data."""
  data = []
  for cls_id, cls in enumerate(classes):
    fp = RAW_DATA_PATH + '{}/'.format(cls)
    all_files = tf.io.gfile.listdir(fp)
    logging.info('Number of files %d', len(all_files))
    for idx, txt_file in enumerate(all_files):

      logging.info('[cls %d] processed % d', cls_id, idx)
      with tf.io.gfile.GFile(fp+'/'+txt_file) as f:
        lines = f.readlines()
        lines = [x.rstrip() for x in lines]
        lines = ' '.join(lines)
        data.append([lines, cls_id])
    logging.info('%d data points collected', len(data))

  random.shuffle(data)
  val_split = int(len(data) * 0.1)
  val_data = data[:val_split]
  test_data = data[val_split:val_split*2]
  train_data = data[val_split*2:]

  with tf.io.gfile.GFile(FLAGS.output_dir+'arxiv_train.csv', 'w+') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(train_data)

  with tf.io.gfile.GFile(FLAGS.output_dir+'arxiv_val.csv', 'w+') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(val_data)

  with tf.io.gfile.GFile(FLAGS.output_dir+'arxiv_test.csv', 'w+') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(test_data)

  logging.info('Finished generating!')


def merge(classes):
  """Used for merging the data."""
  data = []
  cls = classes[FLAGS.class_id]
  cls_id = FLAGS.class_id
  fp = RAW_DATA_PATH + '/{}/'.format(cls)
  all_files = tf.io.gfile.listdir(fp)
  logging.info('Number of files %d', len(all_files))
  for idx, txt_file in enumerate(all_files):
    logging.info('[cls %d] processed % d', cls_id, idx)
    with tf.io.gfile.GFile(fp + '/' + txt_file) as f:
      lines = f.readlines()
      lines = [x.rstrip() for x in lines]
      lines = ' '.join(lines)
      data.append([lines, cls_id])
  logging.info('%d data points collected', len(data))

  with tf.io.gfile.GFile(RAW_DATA_PATH + '/{}.csv'.format(cls), 'w+') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(data)

  logging.info('Finished running')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  classes = [
      'csai', 'csce', 'cscv', 'csds', 'csit', 'csne', 'cspl', 'cssy', 'maac',
      'magr', 'mast'
  ]

  if FLAGS.operation == 'generate':
    generate(classes)
  elif FLAGS.operation == 'merge':
    merge(classes)


if __name__ == '__main__':
  app.run(main)
