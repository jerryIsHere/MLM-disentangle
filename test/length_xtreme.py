from experiment_datasets import xtreme_ds

sequence_length = {}
ds = xtreme_ds.udposTrainDataset()
sequence_length[ds.__class__.__name__] = 0
for i, each in enumerate(ds):
    l = len(
        xtreme_ds.tokenizer(
            ds.dataset[i]["tokens"],
            is_split_into_words=True,
            truncation=False,
        ).input_ids
    )
    if l > sequence_length[ds.__class__.__name__]:
        sequence_length[ds.__class__.__name__] = l
ds = xtreme_ds.udposValidationDataset()
sequence_length[ds.__class__.__name__] = 0
for i, each in enumerate(ds):
    l = len(
        xtreme_ds.tokenizer(
            ds.dataset[i]["tokens"],
            is_split_into_words=True,
            truncation=False,
        ).input_ids
    )
    if l > sequence_length[ds.__class__.__name__]:
        sequence_length[ds.__class__.__name__] = l
ds = xtreme_ds.udposTestDataset()
sequence_length[ds.__class__.__name__] = 0
for i, each in enumerate(ds):
    id = i
    for lan in ds.dataset:
        length = len(ds.dataset[lan])
        if id < length:
            break
        id -= length
    l = len(
        xtreme_ds.tokenizer(
            ds.dataset[lan][id]["tokens"],
            is_split_into_words=True,
            truncation=False,
        ).input_ids
    )
    if l > sequence_length[ds.__class__.__name__]:
        sequence_length[ds.__class__.__name__] = l


ds = xtreme_ds.panxTrainDataset()
sequence_length[ds.__class__.__name__] = 0
for i, each in enumerate(ds):
    l = len(
        xtreme_ds.tokenizer(
            ds.dataset[i]["tokens"],
            is_split_into_words=True,
            truncation=False,
        ).input_ids
    )
    if l > sequence_length[ds.__class__.__name__]:
        sequence_length[ds.__class__.__name__] = l
ds = xtreme_ds.panxValidationDataset()
sequence_length[ds.__class__.__name__] = 0
for i, each in enumerate(ds):
    l = len(
        xtreme_ds.tokenizer(
            ds.dataset[i]["tokens"],
            is_split_into_words=True,
            truncation=False,
        ).input_ids
    )
    if l > sequence_length[ds.__class__.__name__]:
        sequence_length[ds.__class__.__name__] = l
ds = xtreme_ds.panxTestDataset()
sequence_length[ds.__class__.__name__] = 0
for i, each in enumerate(ds):
    id = i
    for lan in ds.dataset:
        length = len(ds.dataset[lan])
        if id < length:
            break
        id -= length
    l = len(
        xtreme_ds.tokenizer(
            ds.dataset[lan][id]["tokens"],
            is_split_into_words=True,
            truncation=False,
        ).input_ids
    )
    if l > sequence_length[ds.__class__.__name__]:
        sequence_length[ds.__class__.__name__] = l


ds = xtreme_ds.xnliTrainDataset()
sequence_length[ds.__class__.__name__] = 0
for i, each in enumerate(ds):
    l = len(
        xtreme_ds.tokenizer(
            ds.dataset[i]["premise"],
            ds.dataset[i]["hypothesis"],
            truncation=False,
        ).input_ids
    )
    if l > sequence_length[ds.__class__.__name__]:
        sequence_length[ds.__class__.__name__] = l
ds = xtreme_ds.xnliValidationDataset()
sequence_length[ds.__class__.__name__] = 0
for i, each in enumerate(ds):
    id = i
    for split in ds.dataset:
        length = len(ds.dataset[split])
        if id < length:
            break
        id -= length
    l = len(
        xtreme_ds.tokenizer(
            ds.dataset[split][id]["premise"],
            ds.dataset[split][id]["hypothesis"],
            truncation=False,
        ).input_ids
    )
    if l > sequence_length[ds.__class__.__name__]:
        sequence_length[ds.__class__.__name__] = l
ds = xtreme_ds.xnliTestDataset()
sequence_length[ds.__class__.__name__] = 0
for i, each in enumerate(ds):
    l = len(
        xtreme_ds.tokenizer(
            ds.dataset[i]["sentence1"],
            ds.dataset[i]["sentence2"],
            truncation=False,
        ).input_ids
    )
    if l > sequence_length[ds.__class__.__name__]:
        sequence_length[ds.__class__.__name__] = l


ds = xtreme_ds.pawsxTrainDataset()
sequence_length[ds.__class__.__name__] = 0
for i, each in enumerate(ds):
    l = len(
        xtreme_ds.tokenizer(
            ds.dataset[i]["sentence1"],
            ds.dataset[i]["sentence2"],
            truncation=False,
        ).input_ids
    )
    if l > sequence_length[ds.__class__.__name__]:
        sequence_length[ds.__class__.__name__] = l
ds = xtreme_ds.pawsxValidationDataset()
sequence_length[ds.__class__.__name__] = 0
for i, each in enumerate(ds):
    l = len(
        xtreme_ds.tokenizer(
            ds.dataset[i]["sentence1"],
            ds.dataset[i]["sentence2"],
            truncation=False,
        ).input_ids
    )
    if l > sequence_length[ds.__class__.__name__]:
        sequence_length[ds.__class__.__name__] = l
ds = xtreme_ds.pawsxTestDataset()
sequence_length[ds.__class__.__name__] = 0
for i, each in enumerate(ds):
    id = i
    for lan in ds.dataset:
        length = len(ds.dataset[lan])
        if id < length:
            break
        id -= length
    l = len(
        xtreme_ds.tokenizer(
            ds.dataset[lan][id]["sentence1"],
            ds.dataset[lan][id]["sentence2"],
            truncation=False,
        ).input_ids
    )
    if l > sequence_length[ds.__class__.__name__]:
        sequence_length[ds.__class__.__name__] = l


ds = xtreme_ds.xquadTrainDataset()
sequence_length[ds.__class__.__name__] = 0
for i, each in enumerate(ds):
    l = len(
        xtreme_ds.tokenizer(
            ds.dataset[i]["context"],
            ds.dataset[i]["question"],
            truncation=False,
        ).input_ids
    )
    if l > sequence_length[ds.__class__.__name__]:
        sequence_length[ds.__class__.__name__] = l
ds = xtreme_ds.xquadValidationDataset()
sequence_length[ds.__class__.__name__] = 0
for i, each in enumerate(ds):
    l = len(
        xtreme_ds.tokenizer(
            ds.dataset[i]["context"],
            ds.dataset[i]["question"],
            truncation=False,
        ).input_ids
    )
    if l > sequence_length[ds.__class__.__name__]:
        sequence_length[ds.__class__.__name__] = l
ds = xtreme_ds.xquadTestDataset()
sequence_length[ds.__class__.__name__] = 0
for i, each in enumerate(ds):
    id = i
    for lan in ds.dataset:
        length = len(ds.dataset[lan])
        if id < length:
            break
        id -= length
    l = len(
        xtreme_ds.tokenizer(
            ds.dataset[lan][id]["context"],
            ds.dataset[id]["question"],
            truncation=False,
        ).input_ids
    )
    if l > sequence_length[ds.__class__.__name__]:
        sequence_length[ds.__class__.__name__] = l


ds = xtreme_ds.mlqaTestDataset()
sequence_length[ds.__class__.__name__] = 0
for i, each in enumerate(ds):
    id = i
    for lan in ds.dataset:
        length = len(ds.dataset[lan])
        if id < length:
            break
        id -= length
    l = len(
        xtreme_ds.tokenizer(
            ds.dataset[lan][id]["context"],
            ds.dataset[id]["question"],
            truncation=False,
        ).input_ids
    )
    if l > sequence_length[ds.__class__.__name__]:
        sequence_length[ds.__class__.__name__] = l

print(sequence_length)
