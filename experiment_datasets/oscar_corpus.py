import pandas as pd

languages = pd.DataFrame.from_dict(
    {
        "family": {
            0: "Indo-European",
            1: "Afro-Asiatic",
            2: "Basque",
            3: "Indo-European",
            4: "Indo-European",
            5: "Sino-Tibetan",
            6: "Sino-Tibetan",
            7: "Indo-European",
            8: "Indo-European",
            9: "Uralic",
            10: "Uralic",
            11: "Indo-European",
            12: "Kartvelian",
            13: "Indo-European",
            14: "Indo-European",
            15: "Afro-Asiatic",
            16: "Indo-European",
            17: "Uralic",
            18: "Austronesian",
            19: "Indo-European",
            20: "Japanese",
            21: "Austronesian",
            22: "Altaic",
            23: "Korean",
            24: "Dravidian",
            25: "Austronesian",
            26: "Indo-European",
            27: "Indo-European",
            28: "Indo-European",
            29: "Indo-European",
            30: "Indo-European",
            31: "Niger-Congo",
            32: "Austronesian",
            33: "Dravidian",
            34: "Dravidian",
            35: "Altaic",
            36: "Indo-European",
            37: "Austro-Asiatic",
            38: "Niger-Congo",
            39: "Tai-Kadai",
        },
        "genus": {
            0: "Germanic",
            1: "Semitic",
            2: "Basque",
            3: "Indic",
            4: "Slavic",
            5: "Burmese-Lolo",
            6: "Chinese",
            7: "Germanic",
            8: "Germanic",
            9: "Finnic",
            10: "Finnic",
            11: "Romance",
            12: "Kartvelian",
            13: "Germanic",
            14: "Greek",
            15: "Semitic",
            16: "Indic",
            17: "Ugric",
            18: "Malayo-Sumbawan",
            19: "Romance",
            20: "Japanese",
            21: "Javanese",
            22: "Turkic",
            23: "Korean",
            24: "Southern Dravidian",
            25: "Malayo-Sumbawan",
            26: "Indic",
            27: "Iranian",
            28: "Romance",
            29: "Slavic",
            30: "Romance",
            31: "Bantoid",
            32: "Greater Central Philippine",
            33: "Southern Dravidian",
            34: "South-Central Dravidian",
            35: "Turkic",
            36: "Indic",
            37: "Viet-Muong",
            38: "Defoid",
            39: "Kam-Tai",
        },
        "iso639_1": {
            0: "af",
            1: "ar",
            2: "eu",
            3: "bn",
            4: "bg",
            5: "my",
            6: "zh",
            7: "nl",
            8: "en",
            9: "et",
            10: "fi",
            11: "fr",
            12: "ka",
            13: "de",
            14: "el",
            15: "he",
            16: "hi",
            17: "hu",
            18: "id",
            19: "it",
            20: "ja",
            21: "jv",
            22: "kk",
            23: "ko",
            24: "ml",
            25: "ms",
            26: "mr",
            27: "fa",
            28: "pt",
            29: "ru",
            30: "es",
            31: "sw",
            32: "tl",
            33: "ta",
            34: "te",
            35: "tr",
            36: "ur",
            37: "vi",
            38: "yo",
            39: "th",
        },
        "iso639_3": {
            0: "afr",
            1: "apc",
            2: "eus",
            3: "ben",
            4: "bul",
            5: "mya",
            6: "cmn",
            7: "nld",
            8: "eng",
            9: "est",
            10: "fin",
            11: "fra",
            12: "kat",
            13: "deu",
            14: "ell",
            15: "heb",
            16: "hin",
            17: "hun",
            18: "ind",
            19: "ita",
            20: "jpn",
            21: "jav",
            22: "kaz",
            23: "kor",
            24: "mal",
            25: "zsm",
            26: "mar",
            27: "pes",
            28: "por",
            29: "rus",
            30: "spa",
            31: "swh",
            32: "tgl",
            33: "tam",
            34: "tel",
            35: "tur",
            36: "urd",
            37: "vie",
            38: "yor",
            39: "tha",
        },
        "language": {
            0: "Afrikaans",
            1: "Arabic",
            2: "Basque",
            3: "Bengali",
            4: "Bulgarian",
            5: "Burmese",
            6: "Chinese",
            7: "Dutch",
            8: "English",
            9: "Estonian",
            10: "Finnish",
            11: "French",
            12: "Georgian",
            13: "German",
            14: "Greek",
            15: "Hebrew",
            16: "Hindi",
            17: "Hungarian",
            18: "Indonesian",
            19: "Italian",
            20: "Japanese",
            21: "Javanese",
            22: "Kazakh",
            23: "Korean",
            24: "Malayalam",
            25: "Malay",
            26: "Marathi",
            27: "Persian (Farsi)",
            28: "Portuguese",
            29: "Russian",
            30: "Spanish",
            31: "Swahili",
            32: "Tagalog",
            33: "Tamil",
            34: "Telugu",
            35: "Turkish",
            36: "Urdu",
            37: "Vietnamese",
            38: "Yoruba",
            39: "Thai",
        },
    }
)
genus = languages.genus.unique()
family = languages.family.unique()


def get_corpus(lan):
    from datasets import load_dataset

    return load_dataset(
        "oscar",
        "unshuffled_deduplicated_" + lan,
        cache_dir="/gpfs1/scratch/ckchan666/oscar",
    )["train"]


def get_token_frequency(lan):
    import pickle

    try:
        path = "/gpfs1/scratch/ckchan666/oscar_token_frequency/" + lan + ".pickle"
        filehandler = open(path, "rb")
        obj = pickle.load(filehandler)
        return obj
    except:
        return None


def put_token_frequency(lan, word_frequency):
    import pickle

    filehandler = open(
        "/gpfs1/scratch/ckchan666/oscar_token_frequency/" + lan + ".pickle", "wb"
    )
    pickle.dump(word_frequency, filehandler)


import torch
from torch.utils.data import Dataset
import spacy
from . import xtreme_ds
from transformers import XLMRobertaTokenizer
import collections
import numpy as np

tokenizer = XLMRobertaTokenizer.from_pretrained(
    "xlm-roberta-large",
    cache_dir="/gpfs1/scratch/ckchan666/transformer_model_cache",
)
punct_chars = [
    "!",
    ".",
    "?",
    "։",
    "؟",
    "۔",
    "܀",
    "܁",
    "܂",
    "߹",
    "।",
    "॥",
    "၊",
    "။",
    "።",
    "፧",
    "፨",
    "᙮",
    "᜵",
    "᜶",
    "᠃",
    "᠉",
    "᥄",
    "᥅",
    "᪨",
    "᪩",
    "᪪",
    "᪫",
    "᭚",
    "᭛",
    "᭞",
    "᭟",
    "᰻",
    "᰼",
    "᱾",
    "᱿",
    "‼",
    "‽",
    "⁇",
    "⁈",
    "⁉",
    "⸮",
    "⸼",
    "꓿",
    "꘎",
    "꘏",
    "꛳",
    "꛷",
    "꡶",
    "꡷",
    "꣎",
    "꣏",
    "꤯",
    "꧈",
    "꧉",
    "꩝",
    "꩞",
    "꩟",
    "꫰",
    "꫱",
    "꯫",
    "﹒",
    "﹖",
    "﹗",
    "！",
    "．",
    "？",
    "𐩖",
    "𐩗",
    "𑁇",
    "𑁈",
    "𑂾",
    "𑂿",
    "𑃀",
    "𑃁",
    "𑅁",
    "𑅂",
    "𑅃",
    "𑇅",
    "𑇆",
    "𑇍",
    "𑇞",
    "𑇟",
    "𑈸",
    "𑈹",
    "𑈻",
    "𑈼",
    "𑊩",
    "𑑋",
    "𑑌",
    "𑗂",
    "𑗃",
    "𑗉",
    "𑗊",
    "𑗋",
    "𑗌",
    "𑗍",
    "𑗎",
    "𑗏",
    "𑗐",
    "𑗑",
    "𑗒",
    "𑗓",
    "𑗔",
    "𑗕",
    "𑗖",
    "𑗗",
    "𑙁",
    "𑙂",
    "𑜼",
    "𑜽",
    "𑜾",
    "𑩂",
    "𑩃",
    "𑪛",
    "𑪜",
    "𑱁",
    "𑱂",
    "𖩮",
    "𖩯",
    "𖫵",
    "𖬷",
    "𖬸",
    "𖭄",
    "𛲟",
    "𝪈",
    "｡",
    "。",
]
punct_chars_2_newline = {each: each + "\n" for each in punct_chars}
tokenizer_vocab = tokenizer.get_vocab()
vocab_token = np.array(list(tokenizer_vocab.values()))

mask_token_id = tokenizer_vocab[tokenizer.mask_token]
s_pre_token = tokenizer_vocab[tokenizer.cls_token]
s_end_token = tokenizer_vocab[tokenizer.sep_token]


def get_custom_corpus_debug():
    from datasets import load_dataset
    import os

    return load_dataset(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "CustomOscarDebug.py"),
        split="train",
        cache_dir="/gpfs1/scratch/ckchan666/custom_oscar_debug",
    )


def get_custom_corpus():
    from datasets import load_dataset
    import os

    return load_dataset(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "CustomOscar.py"),
        split="train",
        cache_dir="/gpfs1/scratch/ckchan666/custom_oscar",
    )


class MLMDisentangleDataset(torch.utils.data.IterableDataset):
    def __init__(self):
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

    def __iter__(self):
        idx = {}
        for lan in xtreme_ds.xtreme_lan:
            idx[lan] = 0
        lan_idx = 0
        while True:
            lan = xtreme_ds.xtreme_lan[lan_idx]
            word_frequency = self.tokenFrequency[lan]
            txt_tokens = np.array([])
            txt_masked_tokens = np.array([])
            while True:
                i = idx[lan] % len(self.corpuses[lan])
                idx[lan] += 1
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
                    yield {
                        "tokens": torch.from_numpy(txt_tokens).long(),
                        "masked_tokens": torch.from_numpy(txt_masked_tokens).long(),
                        "language_id": torch.tensor(lan_idx),
                        "genus_label": torch.tensor(
                            np.nonzero(
                                genus
                                == languages[languages.iso639_1 == lan].genus.iloc[0]
                            )[0]
                        ),
                        "family_label": torch.tensor(
                            np.nonzero(
                                family
                                == languages[languages.iso639_1 == lan].family.iloc[0]
                            )[0]
                        ),
                    }
                    del masked_tokens
                    del prob
                    del chosen
                    del txt_masked_tokens
                    del txt_tokens
                    break
            lan_idx = (lan_idx + 1) % len(xtreme_ds.xtreme_lan)
