"""joint_ft dataset."""

import tensorflow_datasets as tfds
from google.cloud import storage

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
    # TODO(hi_en): Specifies the tfds.core.DatasetInfo object
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
    # TODO(hi_en): Downloads the data and defines the splits
    train = dl_manager.download_and_extract('https://storage.googleapis.com/ai4b-anuvaad-nmt/joint-ft/data/j-train.zip')
    val = dl_manager.download_and_extract('https://storage.googleapis.com/ai4b-anuvaad-nmt/joint-ft/data/j-dev.zip')
    test = dl_manager.download_and_extract('https://storage.googleapis.com/ai4b-anuvaad-nmt/joint-ft/data/j-test.zip')

    # TODO(hi_en): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(source=train/'joint-train.txt', target=train/'en-train.txt'),
        'validation': self._generate_examples(source=val/'joint-dev.txt', target=val/'en-dev.txt'),
        'test': self._generate_examples(source=test/'joint-test.txt', target=test/'en-test.txt')
    }

  def _generate_examples(self, source, target):
    """Yields examples."""
    # TODO(hi_en): Yields (key, example) tuples from the dataset
    client = storage.Client()
    bucket = client.get_bucket('ai4b-anuvaad-nmt')
    blob = bucket.get_blob(path)

    src = blob.download_as_string()
    src = src.decode('utf-8')
    src = src.split('\n')[:-1]

    tgt = blob.download_as_string()
    tgt = tgt.decode('utf-8')
    tgt = tgt.split('\n')[:-1]

    for idx, row in zip(src, tgt):
        yield idx, {
          'source': row[0],
          'target': row[1]
        }
