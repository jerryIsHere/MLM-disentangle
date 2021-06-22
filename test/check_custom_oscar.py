from experiment_datasets import oscar_corpus, xtreme_ds

dataset = oscar_corpus.get_custom_corpus_debug()
MLMD_ds = oscar_corpus.MLMDisentangleDataset()
MLMD_ds = iter(MLMD_ds)
for i in range(len(xtreme_ds.xtreme_lan) * 2):
    print("custom dataset")
    print(xtreme_ds.xtreme_lan[i % len(xtreme_ds.xtreme_lan)])
    for label in dataset[i]:
        print(label)
        print(dataset[i][label])
        if label == "masked_tokens" or label == "tokens":
            print(oscar_corpus.tokenizer.convert_ids_to_tokens(dataset[i][label]))
    print()
    print("original dataset")
    print(xtreme_ds.xtreme_lan[i % len(xtreme_ds.xtreme_lan)])
    example = next(MLMD_ds)
    for label in example:
        print(label)
        print(example[label])
        if label == "masked_tokens" or label == "tokens":
            print(oscar_corpus.tokenizer.convert_ids_to_tokens(example[label]))
    print()
