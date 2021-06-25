from experiment_datasets import xtreme_ds
import numpy as np

ds = xtreme_ds.udposTrainDataset()
for i, each in enumerate(ds):
    tags = each["tags"].numpy()
    tags = tags[tags != -100]
    assert (tags == np.array(ds.dataset[i]["pos_tags"][0 : len(tags)])).all()
ds = xtreme_ds.udposValidationDataset()
for i, each in enumerate(ds):
    tags = each["tags"].numpy()
    tags = tags[tags != -100]
    assert (tags == np.array(ds.dataset[i]["pos_tags"][0 : len(tags)])).all()
ds = xtreme_ds.udposTestDataset()
for i, each in enumerate(ds):
    for lan in ds.dataset:
        length = len(ds.dataset[lan])
        if i < length:
            break
        i -= length
    tags = each["tags"].numpy()
    tags = tags[tags != -100]
    assert (tags == np.array(ds.dataset[lan][i]["pos_tags"][0 : len(tags)])).all()


ds = xtreme_ds.panxTrainDataset()
for i, each in enumerate(ds):
    tags = each["tags"].numpy()
    tags = tags[tags != -100]
    assert (tags == np.array(ds.dataset[i]["ner_tags"][0 : len(tags)])).all()
ds = xtreme_ds.panxValidationDataset()
for i, each in enumerate(ds):
    tags = each["tags"].numpy()
    tags = tags[tags != -100]
    assert (tags == np.array(ds.dataset[i]["ner_tags"][0 : len(tags)])).all()
ds = xtreme_ds.panxTestDataset()
for i, each in enumerate(ds):
    for lan in ds.dataset:
        length = len(ds.dataset[lan])
        if i < length:
            break
        i -= length
    tags = each["tags"].numpy()
    tags = tags[tags != -100]
    assert (tags == np.array(ds.dataset[lan][i]["ner_tags"][0 : len(tags)])).all()


ds = xtreme_ds.xnliTrainDataset()
for i, each in enumerate(ds):
    assert each["label"] == xtreme_ds.xnliTrainDataset.class_label.index(
        ds.dataset[i]["label"]
    )
ds = xtreme_ds.xnliValidationDataset()
for i, each in enumerate(ds):
    for split in ds.dataset:
        length = len(ds.dataset[split])
        if i < length:
            break
        i -= length
    assert each["label"] == xtreme_ds.xnliTrainDataset.class_label.index(
        ds.dataset[split][i]["label"]
    )
ds = xtreme_ds.xnliTestDataset()
for i, each in enumerate(ds):
    assert each["label"] == xtreme_ds.xnliTrainDataset.class_label.index(
        ds.dataset[i]["label"]
    )


ds = xtreme_ds.pawsxTrainDataset()
for i, each in enumerate(ds):
    assert each["label"] == xtreme_ds.pawsxTrainDataset.class_label.index(
        ds.dataset[i]["label"]
    )
ds = xtreme_ds.pawsxValidationDataset()
for i, each in enumerate(ds):
    assert each["label"] == xtreme_ds.pawsxTrainDataset.class_label.index(
        ds.dataset[i]["label"]
    )
ds = xtreme_ds.pawsxTestDataset()
for i, each in enumerate(ds):
    for lan in ds.dataset:
        length = len(ds.dataset[lan])
        if i < length:
            break
        i -= length
    assert each["label"] == xtreme_ds.pawsxTrainDataset.class_label.index(
        ds.dataset[i]["label"]
    )


ds = xtreme_ds.xquadTrainDataset()
for i, each in enumerate(ds):
    assert (
        xtreme_ds.tokenizer.convert_tokens_to_string(
            xtreme_ds.tokenizer.convert_ids_to_tokens(
                each["tokens"][each["start_positions"] : each["end_positions"]]
            )
        )
        == ds.dataset[i]["answers"]["text"][0]
    )
ds = xtreme_ds.xquadValidationDataset()
for i, each in enumerate(ds):
    assert (
        xtreme_ds.tokenizer.convert_tokens_to_string(
            xtreme_ds.tokenizer.convert_ids_to_tokens(
                each["tokens"][each["start_positions"] : each["end_positions"]]
            )
        )
        == ds.dataset[i]["answers"]["text"][0]
    )
ds = xtreme_ds.xquadTestDataset()
for i, each in enumerate(ds):
    for lan in ds.dataset:
        length = len(ds.dataset[lan])
        if i < length:
            break
        i -= length
    assert (
        xtreme_ds.tokenizer.convert_tokens_to_string(
            xtreme_ds.tokenizer.convert_ids_to_tokens(
                each["tokens"][each["start_positions"] : each["end_positions"]]
            )
        )
        == ds.dataset[lan][i]["answers"]["text"][0]
    )


ds = xtreme_ds.mlqaTestDataset()
for i, each in enumerate(ds):
    for lan in ds.dataset:
        length = len(ds.dataset[lan])
        if i < length:
            break
        i -= length
    assert (
        xtreme_ds.tokenizer.convert_tokens_to_string(
            xtreme_ds.tokenizer.convert_ids_to_tokens(
                each["tokens"][each["start_positions"] : each["end_positions"]]
            )
        )
        == ds.dataset[lan][i]["answers"]["text"][0]
    )


ds = xtreme_ds.tydiqaTrainDataset()
for i, each in enumerate(ds):
    pass
ds = xtreme_ds.tydiqaTestDataset()
for i, each in enumerate(ds):
    assert (
        xtreme_ds.tokenizer.convert_tokens_to_string(
            xtreme_ds.tokenizer.convert_ids_to_tokens(
                each["tokens"][each["start_positions"] : each["end_positions"]]
            )
        )
        == ds.dataset[i]["answers"]["text"][0]
    )


ds = xtreme_ds.bucc2018Dataset()
for i, each in enumerate(ds):
    pass
ds = xtreme_ds.tatoebaDataset()
for i, each in enumerate(ds):
    pass
print("all dataset are error free")
