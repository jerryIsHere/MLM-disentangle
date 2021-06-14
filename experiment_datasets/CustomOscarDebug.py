import datasets
from experiment_datasets import xtreme_ds
from experiment_datasets.oscar_corpus import (
    get_corpus,
    get_token_frequency,
    vocab_token,
    punct_chars_2_newline,
    tokenizer,
    mask_token_id,
    s_pre_token,
    s_end_token,
    languages,
    genus,
    family,
)
import collections
import numpy as np


class CustomOscarDebug(datasets.GeneratorBasedBuilder):
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
                        "language_id": datasets.Value("int64"),
                        "genus_label": datasets.Value("int64"),
                        "family_label": datasets.Value("int64"),
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

    def _generate_examples(self):
        self.corpuses = {}
        self.tokenFrequency = {}
        self.vocab_prob = {}
        for lan in xtreme_ds.xtreme_lan:
            self.corpuses[lan] = get_corpus(lan)
            self.tokenFrequency[lan] = get_token_frequency(lan)
            self.vocab_prob[lan] = np.array(
                [
                    0
                    if token_id not in self.tokenFrequency[lan]
                    else self.tokenFrequency[lan][token_id]
                    for token_id in vocab_token
                ]
            )
            self.vocab_prob[lan] = self.vocab_prob[lan] / np.sum(self.vocab_prob[lan])
        idx = {}
        id_all = 0
        for lan in xtreme_ds.xtreme_lan:
            idx[lan] = 0
        lan_idx = 0
        while True:
            lan = xtreme_ds.xtreme_lan[lan_idx]
            word_frequency = self.tokenFrequency[lan]
            txt_tokens = np.array([])
            txt_masked_tokens = np.array([])
            while True:
                i = idx[lan] // len(xtreme_ds.xtreme_lan)
                idx[lan] = idx[lan] + 1
                i = i % len(self.corpuses[lan])
                txt = self.corpuses[lan][i]["text"]
                for each in punct_chars_2_newline:
                    txt = txt.replace(each, punct_chars_2_newline[each])
                for line in txt.split("\n"):
                    tokens = tokenizer.encode(text=line)
                    if len(tokens) < 3:
                        break
                    masked_tokens = np.array(tokens)
                    masked_tokens = masked_tokens[1 : len(masked_tokens) - 1]
                    tf = word_frequency + collections.Counter(masked_tokens)
                    prob = np.array([(1 / tf[token]) ** 0.5 for token in masked_tokens])
                    prob = prob / np.sum(prob)
                    chosen_len = int(len(masked_tokens) * 0.15)
                    chosen_len = chosen_len + (
                        1
                        if np.random.random() < len(masked_tokens) * 0.15 - chosen_len
                        else 0
                    )
                    mask_len = int(chosen_len * 0.8)
                    mask_len = mask_len + (
                        1 if np.random.random() < chosen_len * 0.8 - mask_len else 0
                    )
                    random_len = int(chosen_len * 0.9)
                    random_len = random_len + (
                        1 if np.random.random() < chosen_len * 0.9 - random_len else 0
                    )
                    chosen = np.random.choice(
                        len(masked_tokens), size=chosen_len, replace=False, p=prob
                    )
                    for indice in chosen[0:mask_len]:  # quantization problem
                        masked_tokens[indice] = mask_token_id
                    for indice in chosen[mask_len:random_len]:
                        masked_tokens[indice] = np.random.choice(
                            vocab_token, p=self.vocab_prob[lan]
                        )
                    masked_tokens = np.concatenate(
                        ([s_pre_token], masked_tokens, [s_end_token])
                    )
                    txt_tokens = np.concatenate((txt_tokens, tokens))
                    txt_masked_tokens = np.concatenate(
                        (txt_masked_tokens, masked_tokens)
                    )
                if len(txt_tokens) >= tokenizer.model_max_length:
                    txt_tokens = txt_tokens[0 : tokenizer.model_max_length]
                    txt_masked_tokens = txt_masked_tokens[
                        0 : tokenizer.model_max_length
                    ]
                    yield id_all, {
                        "tokens": txt_tokens,
                        "masked_tokens": txt_masked_tokens,
                        "language_id": lan_idx,
                        "genus_label": np.nonzero(
                            genus == languages[languages.iso639_1 == lan].genus.iloc[0]
                        )[0][0],
                        "family_label": np.nonzero(
                            family
                            == languages[languages.iso639_1 == lan].family.iloc[0]
                        )[0][0],
                    }
                    id_all += 1
                    del masked_tokens
                    del prob
                    del chosen
                    del txt_masked_tokens
                    del txt_tokens
                    break
            lan_idx = (lan_idx + 1) % len(xtreme_ds.xtreme_lan)
            if id_all > len(xtreme_ds.xtreme_lan):# * 700 * 4:
                break
