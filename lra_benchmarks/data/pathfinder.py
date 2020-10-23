"""TFDS builder for pathfinder challenge."""

import os

import tensorflow as tf
import tensorflow_datasets as tfds



class Pathfinder32(tfds.core.BeamBasedBuilder):
  """Pathfinder TFDS builder (where the resolution is 32)."""

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
  """Pathfinder TFDS builder (where the resolution is 32)."""

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
  """Pathfinder TFDS builder (where the resolution is 32)."""

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
  """Pathfinder TFDS builder (where the resolution is 32)."""

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
