from experiment_datasets import oscar_corpus, xtreme_ds

dataset = oscar_corpus.get_custom_corpus()
for i in range(len(xtreme_ds.xtreme_lan)):
    print(dataset[i])
