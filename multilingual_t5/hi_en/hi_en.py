"""hi_en dataset."""

import tensorflow_datasets as tfds
import pandas as pd

# TODO(hi_en): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(hi_en): BibTeX citation
_CITATION = """
"""


class HiEn(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for hi_en dataset."""

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
    train = dl_manager.download_and_extract('https://storage.googleapis.com/ai4b-anuvaad-nmt/hi_en/data/hi_en_test.csv')
    val = dl_manager.download_and_extract('https://storage.googleapis.com/ai4b-anuvaad-nmt/hi_en/data/hi_en_test.csv')

    # TODO(hi_en): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(train),
        'val': self._generate_examples(val)
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(hi_en): Yields (key, example) tuples from the dataset
    df = pd.read_csv(path)
    for idx, row in df.iterrows():
      yield idx, {
        'source': row['src'],
        'target': row['tgt']
      }
