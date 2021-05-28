import xtreme_ds
import oscar_corpus
from transformers import XLMRobertaTokenizer
import collections
from functools import partial, reduce
import multiprocessing as mp
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser(description="token frequency of 40 corpus")
parser.add_argument(
    "--example",
    metavar="N",
    type=int,
    help="number of example from each corpus",
    default=2 ** 16,
)
args = parser.parse_args()
print("with " + str(mp.cpu_count()) + " cpu")
print("example per langauge: " + str(args.example))
tokenizer = XLMRobertaTokenizer.from_pretrained(
    "xlm-roberta-large", cache_dir="/gpfs1/scratch/ckchan666/transformer_model_cache"
)
import collections, time

t = time.time()
for lan in xtreme_ds.xtreme_lan:
    dataset = oscar_corpus.get_corpus(lan)
    example = args.example
    if len(dataset) < example:
        example = len(dataset)

    def trial(i):
        token = tokenizer.encode(text=dataset[int(i)]["text"])
        token = token[1 : len(token) - 1]
        return collections.Counter(token)

    def combine(a, b):
        return a + b

    with mp.Pool(processes=mp.cpu_count()) as pool:
        es_per_process = pool.map(
            partial(map, trial),
            [np.arange(i, example, pool._processes) for i in range(pool._processes)],
        )
        word_frequency = reduce(
            combine, pool.map(partial(reduce, combine), es_per_process)
        )
    oscar_corpus.put_token_frequency(lan, word_frequency)


time_token = time.time() - t

print("seconds needed for " + str(args.example) + " examples for 40 corpus is:")
print(time_token)
