"""joint_ft dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import random

# TODO(joint_ft): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(joint_ft): BibTeX citation
_CITATION = """
"""


class JointFt(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for joint_ft dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(joint_ft): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'source': tfds.features.Text(),
            'target': tfds.features.Text(),
        }),
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(joint_ft): Downloads the data and defines the splits
    bn = dl_manager.download_and_extract('https://anuvaad-parallel-corpus.s3-us-west-2.amazonaws.com/train-2021-v1-en-hi.zip')
    hi = dl_manager.download_and_extract('https://anuvaad-parallel-corpus.s3-us-west-2.amazonaws.com/train-2021-v1-en-bn.zip')
    ta = dl_manager.download_and_extract('https://anuvaad-parallel-corpus.s3-us-west-2.amazonaws.com/train-2021-v1-en-ta.zip')

    devtest = dl_manager.download_and_extract('https://anuvaad-parallel-corpus.s3-us-west-2.amazonaws.com/devtest-2021-v1.zip')

    # TODO(joint_ft): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(bn_src=bn/'train.bn', bn_tgt=bn/'train.en', hi_src=hi/'train.hi', hi_tgt=hi/'train.en', ta_src=ta/'train.ta', ta_tgt=ta/'train.en'),
        'validation': self._generate_examples(bn_src=devtest/'all/en-bn/dev.bn', bn_tgt=devtest/'all/en-bn/dev.en', hi_src=devtest/'all/en-hi/dev.hi', hi_tgt=devtest/'all/en-hi/dev.en', ta_src=devtest/'all/en-ta/dev.ta', ta_tgt=devtest/'all/en-ta/dev.en'),
        'test': self._generate_examples(bn_src=devtest/'all/en-bn/test.bn', bn_tgt=devtest/'all/en-bn/test.en', hi_src=devtest/'all/en-hi/test.hi', hi_tgt=devtest/'all/en-hi/test.en', ta_src=devtest/'all/en-ta/test.ta', ta_tgt=devtest/'all/en-ta/test.en'),
    }

  def _generate_examples(self, bn_src, bn_tgt, hi_src, hi_tgt, ta_src, ta_tgt):
    """Yields examples."""
    # TODO(joint_ft): Yields (key, example) tuples from the dataset
    bn_src = tf.io.gfile.GFile(bn_src, mode='r').readlines()
    bn_tgt = tf.io.gfile.GFile(bn_tgt, mode='r').readlines()

    hi_src = tf.io.gfile.GFile(hi_src, mode='r').readlines()
    hi_tgt = tf.io.gfile.GFile(hi_tgt, mode='r').readlines()

    ta_src = tf.io.gfile.GFile(ta_src, mode='r').readlines()
    ta_tgt = tf.io.gfile.GFile(ta_tgt, mode='r').readlines()

    src = []
    tgt = []

    src.extend(bn_src)
    src.extend(hi_src)
    src.extend(ta_src)

    tgt.extend(bn_tgt)
    tgt.extend(hi_tgt)
    tgt.extend(ta_tgt)

    temp = list(zip(a,b))
    random.shuffle(temp)

    src, tgt = zip(*temp)

    for idx, row in enumerate(zip(src, tgt)):
      yield idx, {
        'source': row[0],
        'target': row[1]
      }
