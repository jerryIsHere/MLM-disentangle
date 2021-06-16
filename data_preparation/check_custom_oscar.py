from experiment_datasets import oscar_corpus, xtreme_ds

print("custom_dataset")
dataset = oscar_corpus.get_custom_corpus_debug()
for i in range(len(xtreme_ds.xtreme_lan) * 2):
    print(xtreme_ds.xtreme_lan[i % len(xtreme_ds.xtreme_lan)])
    for label in dataset[i]:
        print(label)
        print(dataset[i][label])

print("original dataset")
MLMD_ds = oscar_corpus.MLMDisentangleDataset()
MLMD_ds = iter(MLMD_ds)
for i in range(len(xtreme_ds.xtreme_lan) * 2):
    print(xtreme_ds.xtreme_lan[i % len(xtreme_ds.xtreme_lan)])
    example = next(MLMD_ds)
    for label in example:
        print(label)
        print(example[label])
