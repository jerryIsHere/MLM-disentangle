from experiment_datasets import oscar_corpus, xtreme_ds
import time

t = time.time()
dataset = oscar_corpus.get_custom_corpus()
print("elapsed time for building corpus")
print(time.time() - t)
for i in range(len(xtreme_ds.xtreme_lan)):
    print(dataset[i])
