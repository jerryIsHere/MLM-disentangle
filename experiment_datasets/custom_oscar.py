import datasets
from . import oscar_corpus

class Oscar(datasets.GeneratorBasedBuilder):


    def _info(self):

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description="custom oscar datasets",
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                 datasets.Features(
                {
                    "tokens": datasets.Sequence(datasets.Value("int64")),
                    "masked_tokens": datasets.Sequence(datasets.Value("int64")),
                    "language_id":datasets.Value("int8"),
                    "genus_label":datasets.Value("int8"),
                    "family_label":datasets.Value("int8"),
                }
            )

            ),
        )

    def _split_generators(self, dl_manager):

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
            )
        ]

    def _generate_examples(self, filepath):
        
