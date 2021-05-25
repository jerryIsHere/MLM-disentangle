def get_corpus(lan):
    from datasets import load_dataset

    return load_dataset(
        "oscar",
        "unshuffled_deduplicated_" + lan,
        cache_dir="/gpfs1/scratch/ckchan666/oscar",
    )['"train"']


def get_token_frequency(lan):
    import pickle

    filehandler = open(
        "/gpfs1/scratch/ckchan666/oscar_token_frequency/" + lan + ".pickle", "r"
    )
    object = pickle.load(filehandler)


def put_token_frequency(lan, word_frequency):
    import pickle

    filehandler = open(
        "/gpfs1/scratch/ckchan666/oscar_token_frequency/" + lan + ".pickle", "w"
    )
    pickle.dump(word_frequency, filehandler)


import torch
from torch.utils.data import Dataset
import spacy
import xtreme_ds
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
                    if token_id not in self.tokenFrequency
                    else self.tokenFrequency[token_id]
                    for token_id in vocab_token
                ]
            )
            self.vocab_prob[lan] = vocab_prob / np.sum(vocab_prob[lan])

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
                    txt_masked_tokens = np.concatenate((txt_masked_tokens, masked_tokens))
                if len(txt_tokens) >= tokenizer.model_max_length:
                    txt_tokens = txt_tokens[0 : tokenizer.model_max_length]
                    txt_masked_tokens = txt_masked_tokens[0 : tokenizer.model_max_length]
                    yield {
                        "tokens": txt_tokens,
                        "masked_tokens": txt_masked_tokens,
                        "language_id": torch.nn.functional.one_hot(
                            torch.tensor(lan_idx), num_classes=len(xtreme_ds.xtreme_lan)
                        ),
                    }
                    break
            lan_idx = (lan_idx + 1) % len(xtreme_ds.xtreme_lan)
