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
"""TFDS builder for pathfinder challenge."""

import os

import tensorflow as tf
import tensorflow_datasets as tfds



class Pathfinder32(tfds.core.BeamBasedBuilder):
  """Pathfinder TFDS builder (where the resolution is 32).

  The data for this dataset was generated using the script in
  https://github.com/drewlinsley/pathfinder with the default parameters, while
  followings being customized:
  ```
    args.paddle_margin_list = [1]
    args.window_size = [32, 32]
    args.padding= 1
    args.paddle_length = 2
    args.marker_radius = 1.5
    args.contour_length = 14
    args.paddle_thickness = 0.5
    args.antialias_scale = 2
    args.seed_distance= 7
    args.continuity = 1.0
    args.distractor_length = args.contour_length // 3
    args.num_distractor_snakes = 20 // args.distractor_length
    args.snake_contrast_list = [2]
    args.paddle_contrast_list = [0.75]
  ```
  """

  VERSION = tfds.core.Version('1.0.0')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=('This is a builder for pathfinder challenge dataset'),
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(),
            'label': tfds.features.ClassLabel(num_classes=2)
        }),
        supervised_keys=('image', 'label'),
        homepage='',
        citation="""@inproceedings{
                    Kim*2020Disentangling,
                    title={Disentangling neural mechanisms for perceptual grouping},
                    author={Junkyung Kim* and Drew Linsley* and Kalpit Thakkar and Thomas Serre},
                    booktitle={International Conference on Learning Representations},
                    year={2020},
                    url={https://openreview.net/forum?id=HJxrVA4FDS}
                    }""",
    )

  def _split_generators(self, dl_manager):
    """Downloads the data and defines the splits."""

    return [
        tfds.core.SplitGenerator(
            name='easy', gen_kwargs={'file_pattern': 'curv_baseline'}),
        tfds.core.SplitGenerator(
            name='intermediate',
            gen_kwargs={'file_pattern': 'curv_contour_length_9'}),
        tfds.core.SplitGenerator(
            name='hard', gen_kwargs={'file_pattern': 'curv_contour_length_14'})
    ]

  def _build_pcollection(self, pipeline, file_pattern):
    """Generate examples as dicts."""
    beam = tfds.core.lazy_imports.apache_beam

    def _generate_examples(file_path):
      """Read the input data out of the source files."""
      example_id = 0
      meta_examples = tf.io.read_file(file_path).numpy().decode('utf-8').split(
          '\n')[:-1]
      print(meta_examples)
      for m_example in meta_examples:
        m_example = m_example.split(' ')
        image_path = os.path.join(ORIGINAL_DATA_DIR_32, file_pattern,
                                  m_example[0], m_example[1])
        example_id += 1
        yield '_'.join([m_example[0], m_example[1],
                        str(example_id)]), {
                            'image': image_path,
                            'label': int(m_example[3]),
                        }

    meta_file_pathes = tf.io.gfile.glob(
        os.path.join(ORIGINAL_DATA_DIR_32, file_pattern, 'metadata/*.npy'))
    print(len(meta_file_pathes))
    return (pipeline
            | 'Create' >> beam.Create(meta_file_pathes)
            | 'Generate' >> beam.ParDo(_generate_examples))


class Pathfinder64(tfds.core.BeamBasedBuilder):
  """Pathfinder TFDS builder (where the resolution is 64).

  The data for this dataset was generated using the script in
  https://github.com/drewlinsley/pathfinder with the default parameters, while
  followings being customized:
  ```
    args.padding = 1
    args.antialias_scale = 4
    args.paddle_margin_list = [1]
    args.seed_distance = 12
    args.window_size = [64,64]
    args.marker_radius = 2.5
    args.contour_length = 14
    args.paddle_thickness = 1
    args.antialias_scale = 2
    args.continuity = 1.8  # from 1.8 to 0.8, with steps of 66%
    args.distractor_length = args.contour_length / 3
    args.num_distractor_snakes = 22 / args.distractor_length
    args.snake_contrast_list = [0.8]
  ```
  """

  VERSION = tfds.core.Version('1.0.0')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=('This is a builder for pathfinder challenge dataset'),
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(),
            'label': tfds.features.ClassLabel(num_classes=2)
        }),
        supervised_keys=('image', 'label'),
        homepage='',
        citation="""@inproceedings{
                    Kim*2020Disentangling,
                    title={Disentangling neural mechanisms for perceptual grouping},
                    author={Junkyung Kim* and Drew Linsley* and Kalpit Thakkar and Thomas Serre},
                    booktitle={International Conference on Learning Representations},
                    year={2020},
                    url={https://openreview.net/forum?id=HJxrVA4FDS}
                    }""",
    )

  def _split_generators(self, dl_manager):
    """Downloads the data and defines the splits."""

    return [
        tfds.core.SplitGenerator(
            name='easy', gen_kwargs={'file_pattern': 'curv_baseline'}),
        tfds.core.SplitGenerator(
            name='intermediate',
            gen_kwargs={'file_pattern': 'curv_contour_length_9'}),
        tfds.core.SplitGenerator(
            name='hard', gen_kwargs={'file_pattern': 'curv_contour_length_14'})
    ]

  def _build_pcollection(self, pipeline, file_pattern):
    """Generate examples as dicts."""
    beam = tfds.core.lazy_imports.apache_beam

    def _generate_examples(file_path):
      """Read the input data out of the source files."""
      example_id = 0
      meta_examples = tf.io.read_file(file_path).numpy().decode('utf-8').split(
          '\n')[:-1]
      print(meta_examples)
      for m_example in meta_examples:
        m_example = m_example.split(' ')
        image_path = os.path.join(ORIGINAL_DATA_DIR_64, file_pattern,
                                  m_example[0], m_example[1])
        example_id += 1
        yield '_'.join([m_example[0], m_example[1],
                        str(example_id)]), {
                            'image': image_path,
                            'label': int(m_example[3]),
                        }

    meta_file_pathes = tf.io.gfile.glob(
        os.path.join(ORIGINAL_DATA_DIR_64, file_pattern, 'metadata/*.npy'))
    print(len(meta_file_pathes))
    return (pipeline
            | 'Create' >> beam.Create(meta_file_pathes)
            | 'Generate' >> beam.ParDo(_generate_examples))


class Pathfinder128(tfds.core.BeamBasedBuilder):
  """Pathfinder TFDS builder (where the resolution is 128).

  The data for this dataset was generated using the script in
  https://github.com/drewlinsley/pathfinder with the default parameters, while
  followings being customized:
  ```
    args.padding = 1
    args.antialias_scale = 4
    args.paddle_margin_list = [2,3]
    args.seed_distance = 20
    args.window_size = [128,128]
    args.marker_radius = 3
    args.contour_length = 14
    args.paddle_thickness = 1.5
    args.antialias_scale = 2
    args.continuity = 1.8  # from 1.8 to 0.8, with steps of 66%
    args.distractor_length = args.contour_length / 3
    args.num_distractor_snakes = 35 / args.distractor_length
    args.snake_contrast_list = [0.9]
  ```
  """

  VERSION = tfds.core.Version('1.0.0')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=('This is a builder for pathfinder challenge dataset'),
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(),
            'label': tfds.features.ClassLabel(num_classes=2)
        }),
        supervised_keys=('image', 'label'),
        homepage='',
        citation="""@inproceedings{
                    Kim*2020Disentangling,
                    title={Disentangling neural mechanisms for perceptual grouping},
                    author={Junkyung Kim* and Drew Linsley* and Kalpit Thakkar and Thomas Serre},
                    booktitle={International Conference on Learning Representations},
                    year={2020},
                    url={https://openreview.net/forum?id=HJxrVA4FDS}
                    }""",
    )

  def _split_generators(self, dl_manager):
    """Downloads the data and defines the splits."""

    return [
        tfds.core.SplitGenerator(
            name='easy', gen_kwargs={'file_pattern': 'curv_baseline'}),
        tfds.core.SplitGenerator(
            name='intermediate',
            gen_kwargs={'file_pattern': 'curv_contour_length_9'}),
        tfds.core.SplitGenerator(
            name='hard', gen_kwargs={'file_pattern': 'curv_contour_length_14'})
    ]

  def _build_pcollection(self, pipeline, file_pattern):
    """Generate examples as dicts."""
    beam = tfds.core.lazy_imports.apache_beam
    def _generate_examples(file_path):
      """Read the input data out of the source files."""
      example_id = 0
      meta_examples = tf.io.read_file(
          file_path).numpy().decode('utf-8').split('\n')[:-1]
      print(meta_examples)
      for m_example in meta_examples:
        m_example = m_example.split(' ')
        image_path = os.path.join(ORIGINAL_DATA_DIR_128, file_pattern,
                                  m_example[0], m_example[1])
        example_id += 1
        yield '_'.join([m_example[0], m_example[1], str(example_id)]), {
            'image': image_path,
            'label': int(m_example[3]),
        }

    meta_file_pathes = tf.io.gfile.glob(
        os.path.join(ORIGINAL_DATA_DIR_128, file_pattern, 'metadata/*.npy'))
    print(len(meta_file_pathes))
    return (
        pipeline
        | 'Create' >> beam.Create(meta_file_pathes)
        | 'Generate' >> beam.ParDo(_generate_examples)
    )


class Pathfinder256(tfds.core.BeamBasedBuilder):
  """Pathfinder TFDS builder (where the resolution is 256).

  The data for this dataset was generated using the script in
  https://github.com/drewlinsley/pathfinder with the default parameters, while
  followings being customized:
  ```
    args.antialias_scale = 4
    args.paddle_margin_list = [3]
    args.window_size = [256,256]
    args.marker_radius = 5
    args.contour_length = 14
    args.paddle_thickness = 2
    args.antialias_scale = 2
    args.continuity = 1.8
    args.distractor_length = args.contour_length / 3
    args.num_distractor_snakes = 30 / args.distractor_length
    args.snake_contrast_list = [1.0]
  ```
  """

  VERSION = tfds.core.Version('1.0.0')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=('This is a builder for pathfinder challenge dataset'),
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(),
            'label': tfds.features.ClassLabel(num_classes=2)
        }),
        supervised_keys=('image', 'label'),
        homepage='',
        citation="""@inproceedings{
                    Kim*2020Disentangling,
                    title={Disentangling neural mechanisms for perceptual grouping},
                    author={Junkyung Kim* and Drew Linsley* and Kalpit Thakkar and Thomas Serre},
                    booktitle={International Conference on Learning Representations},
                    year={2020},
                    url={https://openreview.net/forum?id=HJxrVA4FDS}
                    }""",
    )

  def _split_generators(self, dl_manager):
    """Downloads the data and defines the splits."""

    return [
        tfds.core.SplitGenerator(
            name='easy', gen_kwargs={'file_pattern': 'curv_baseline'}),
        tfds.core.SplitGenerator(
            name='intermediate',
            gen_kwargs={'file_pattern': 'curv_contour_length_9'}),
        tfds.core.SplitGenerator(
            name='hard', gen_kwargs={'file_pattern': 'curv_contour_length_14'})
    ]

  def _build_pcollection(self, pipeline, file_pattern):
    """Generate examples as dicts."""
    beam = tfds.core.lazy_imports.apache_beam

    def _generate_examples(file_path):
      """Read the input data out of the source files."""
      example_id = 0
      meta_examples = tf.io.read_file(file_path).numpy().decode('utf-8').split(
          '\n')[:-1]
      print(meta_examples)
      for m_example in meta_examples:
        m_example = m_example.split(' ')
        image_path = os.path.join(ORIGINAL_DATA_DIR_256, file_pattern,
                                  m_example[0], m_example[1])
        example_id += 1
        yield '_'.join([m_example[0], m_example[1],
                        str(example_id)]), {
                            'image': image_path,
                            'label': int(m_example[3]),
                        }

    meta_file_pathes = tf.io.gfile.glob(
        os.path.join(ORIGINAL_DATA_DIR_256, file_pattern, 'metadata/*.npy'))
    print(len(meta_file_pathes))
    return (pipeline
            | 'Create' >> beam.Create(meta_file_pathes)
            | 'Generate' >> beam.ParDo(_generate_examples))
