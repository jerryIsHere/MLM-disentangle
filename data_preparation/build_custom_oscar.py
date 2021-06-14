from experiment_datasets import oscar_corpus, xtreme_ds
import time
import argparse

parser = argparse.ArgumentParser(description="build custom oscar datasets")
parser.add_argument(
    "--debug",
    action="store_true",
)
args = parser.parse_args()
t = time.time()
print("debug mode: " + str(args.debug))
if args.debug:
    dataset = oscar_corpus.get_custom_corpus_debug()
else:
    dataset = oscar_corpus.get_custom_corpus()
print("elapsed time for building corpus")
print(time.time() - t)
for i in range(len(xtreme_ds.xtreme_lan)):
    print(xtreme_ds.xtreme_lan[i])
    for label in dataset[i]:
        print(label)
        print(dataset[i][label].dtype)
        print(dataset[i][label])
