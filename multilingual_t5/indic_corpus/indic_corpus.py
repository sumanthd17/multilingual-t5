"""indic_corpus dataset."""
import tensorflow_datasets as tfds
from google.cloud import storage

_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

_CITATION = """
@inproceedings{kakwani2020indicnlpsuite,
    title={{IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages}},
    author={Divyanshu Kakwani and Anoop Kunchukuttan and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
    year={2020},
    booktitle={Findings of EMNLP},
}
"""

VERSION = tfds.core.Version("1.0.0")

RELEASE_NOTES = {
    "1.0.0": "Initial release.",
}

INDIC_LANGS = ["as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]


class IndicCorpusConfig(tfds.core.BuilderConfig):
    """BuilderConfig for IndicCorpus"""

    def __init__(self, name, languages, **kwargs):
        super().__init__(name=name, version=VERSION, **kwargs)

        self.languages = languages


class IndicCorpus(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for indic_corpus dataset."""

    BUILDER_CONFIGS = [
        IndicCorpusConfig(
            "indic_corpus",
            languages=INDIC_LANGS,
            description="Indic Corp has 12 languages and is generated from web sources",
        )
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        features = {"text": tfds.features.Text()}
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage="https://indicnlp.ai4bharat.org/home/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(indic_corpus): Downloads the data and defines the splits
        path = dl_manager.download_and_extract(
            "https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/indiccorp/all_langs/monoling.zip"
        )

        # TODO(indic_corpus): Returns the Dict[split names, Iterator[Key, Example]]
        splits = []
        for lang in self.builder_config.languages:
            splits.extend(
                [
                    tfds.core.SplitGenerator(
                        name=lang, gen_kwargs=dict(path=path / f"monoling/{lang}.txt", split=lang)
                    ),
                    tfds.core.SplitGenerator(
                        name=f"{lang}-validation",
                        gen_kwargs=dict(path=path / f"monoling/{lang}-validation.txt", split=f'{lang}-validation'),
                    ),
                ]
            )
        return splits

    def _generate_examples(self, path, split):
        """Yields examples."""
        client = storage.Client()
        bucket = client.get_bucket('pre-train')
        path = f't5-tfds/downloads/extracted/ZIP.ai4b-publ-indi-nlp-corp_indi_all_lang_monoACZKQgNF83mDxE_okLuFZRPPtb-SQxlLxv_f_U1NZUg.zip/monoling/{split}.txt'
        blob = bucket.get_blob(path)

        lines = blob.download_as_string()
        lines = lines.decode('utf-8')
        lines = lines.split('\n')[:-1]

        for id_, line in enumerate(lines):
          yield id_, {'text': line}

