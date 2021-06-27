from experiment_datasets import xtreme_ds
import numpy as np

ds = xtreme_ds.udposTrainDataset()
for i, each in enumerate(ds):
    tags = each["tags"].numpy()
    tags = tags[tags != -100]
    assert (
        tags
        == np.array(
            each["features"]["pos_tags"][each["offset"] : each["offset"] + len(tags)]
        )
    ).all()
assert i >= len(ds) - 1
ds = xtreme_ds.udposValidationDataset()
for i, each in enumerate(ds):
    tags = each["tags"].numpy()
    tags = tags[tags != -100]
    assert (
        tags
        == np.array(
            each["features"]["pos_tags"][each["offset"] : each["offset"] + len(tags)]
        )
    ).all()
assert i >= len(ds) - 1
ds = xtreme_ds.udposTestDataset()
for i, each in enumerate(ds):
    tags = each["tags"].numpy()
    tags = tags[tags != -100]
    assert (
        tags
        == np.array(
            each["features"]["pos_tags"][each["offset"] : each["offset"] + len(tags)]
        )
    ).all()
assert i >= len(ds) - 1


ds = xtreme_ds.panxTrainDataset()
for i, each in enumerate(ds):
    tags = each["tags"].numpy()
    tags = tags[tags != -100]
    assert (
        tags
        == np.array(
            each["features"]["ner_tags"][each["offset"] : each["offset"] + len(tags)]
        )
    ).all()
assert i >= len(ds) - 1
ds = xtreme_ds.panxValidationDataset()
for i, each in enumerate(ds):
    tags = each["tags"].numpy()
    tags = tags[tags != -100]
    assert (
        tags
        == np.array(
            each["features"]["ner_tags"][each["offset"] : each["offset"] + len(tags)]
        )
    ).all()
assert i >= len(ds) - 1
ds = xtreme_ds.panxTestDataset()
for i, each in enumerate(ds):
    tags = each["tags"].numpy()
    tags = tags[tags != -100]
    assert (
        tags
        == np.array(
            each["features"]["ner_tags"][each["offset"] : each["offset"] + len(tags)]
        )
    ).all()
assert i >= len(ds) - 1


ds = xtreme_ds.xnliTrainDataset()
for i, each in enumerate(ds):
    assert each["label"] == xtreme_ds.xnliTrainDataset.class_label.index(
        ds.dataset[i]["label"]
    )
assert i == len(ds) - 1
ds = xtreme_ds.xnliValidationDataset()
for i, each in enumerate(ds):
    id = i
    for split in ds.dataset:
        length = len(ds.dataset[split])
        if id < length:
            break
        id -= length
    assert each["label"] == xtreme_ds.xnliTrainDataset.class_label.index(
        ds.dataset[split][id]["label"]
    )
assert i == len(ds) - 1
ds = xtreme_ds.xnliTestDataset()
for i, each in enumerate(ds):
    assert each["label"] == xtreme_ds.xnliTestDataset.class_label.index(
        ds.dataset[i]["gold_label"]
    )
assert i == len(ds) - 1


ds = xtreme_ds.pawsxTrainDataset()
for i, each in enumerate(ds):
    assert each["label"] == xtreme_ds.pawsxTrainDataset.class_label.index(
        ds.dataset[i]["label"]
    )
assert i == len(ds) - 1
ds = xtreme_ds.pawsxValidationDataset()
for i, each in enumerate(ds):
    assert each["label"] == xtreme_ds.pawsxTrainDataset.class_label.index(
        ds.dataset[i]["label"]
    )
assert i == len(ds) - 1
ds = xtreme_ds.pawsxTestDataset()
for i, each in enumerate(ds):
    id = i
    for lan in ds.dataset:
        length = len(ds.dataset[lan])
        if id < length:
            break
        id -= length
    assert each["label"] == xtreme_ds.pawsxTrainDataset.class_label.index(
        ds.dataset[lan][id]["label"]
    )
assert i == len(ds) - 1


import datasets

metrics = datasets.load_metric("squad")
ds = xtreme_ds.xquadTrainDataset()
for i, each in enumerate(ds):
    for j, s_p in enumerate(each["start_positions"]):
        reply = xtreme_ds.tokenizer.convert_tokens_to_string(
            xtreme_ds.tokenizer.convert_ids_to_tokens(
                each["tokens"][each["start_positions"][j] : each["end_positions"][j]]
            )
        )
        metrics.add(
            prediction={"id": each["id"], "prediction_text": " ".join(reply)},
            reference={
                "id": each["id"],
                "answers": {
                    "text": [
                        " ".join(xtreme_ds.normalize_string(each["answers"]["text"][j]))
                    ],
                    "answer_start": [each["answers"]["answer_start"][j]],
                },
            },
        )
print(metrics.compute())
assert i == len(ds) - 1
ds = xtreme_ds.xquadValidationDataset()
for i, each in enumerate(ds):
    for j, s_p in enumerate(each["start_positions"]):
        reply = xtreme_ds.tokenizer.convert_tokens_to_string(
            xtreme_ds.tokenizer.convert_ids_to_tokens(
                each["tokens"][each["start_positions"][j] : each["end_positions"][j]]
            )
        )
        answer = xtreme_ds.normalize_string(each["answers"]["text"][j])
        metrics.add(
            prediction={"id": each["id"], "prediction_text": " ".join(reply)},
            reference={
                "id": each["id"],
                "answers": {
                    "text": [
                        " ".join(xtreme_ds.normalize_string(each["answers"]["text"][j]))
                    ],
                    "answer_start": [each["answers"]["answer_start"][j]],
                },
            },
        )
print(metrics.compute())
assert i == len(ds) - 1
ds = xtreme_ds.xquadTestDataset()
for i, each in enumerate(ds):
    id = i
    for lan in ds.dataset:
        length = len(ds.dataset[lan])
        if id < length:
            break
        id -= length
    for j, s_p in enumerate(each["start_positions"]):
        reply = xtreme_ds.tokenizer.convert_tokens_to_string(
            xtreme_ds.tokenizer.convert_ids_to_tokens(
                each["tokens"][each["start_positions"][j] : each["end_positions"][j]]
            )
        )
        answer = xtreme_ds.normalize_string(each["answers"]["text"][j])
        metrics.add(
            prediction={"id": each["id"], "prediction_text": " ".join(reply)},
            reference={
                "id": each["id"],
                "answers": {
                    "text": [
                        " ".join(xtreme_ds.normalize_string(each["answers"]["text"][j]))
                    ],
                    "answer_start": [each["answers"]["answer_start"][j]],
                },
            },
        )
print(metrics.compute())
assert i == len(ds) - 1


ds = xtreme_ds.mlqaTestDataset()
for i, each in enumerate(ds):
    id = i
    for lan in ds.dataset:
        length = len(ds.dataset[lan])
        if id < length:
            break
        id -= length
    for j, s_p in enumerate(each["start_positions"]):
        reply = xtreme_ds.tokenizer.convert_tokens_to_string(
            xtreme_ds.tokenizer.convert_ids_to_tokens(
                each["tokens"][each["start_positions"][j] : each["end_positions"][j]]
            )
        )
        answer = xtreme_ds.normalize_string(each["answers"]["text"][j])
        metrics.add(
            prediction={"id": each["id"], "prediction_text": " ".join(reply)},
            reference={
                "id": each["id"],
                "answers": {
                    "text": [
                        " ".join(xtreme_ds.normalize_string(each["answers"]["text"][j]))
                    ],
                    "answer_start": [each["answers"]["answer_start"][j]],
                },
            },
        )
print(metrics.compute())
assert i == len(ds) - 1


ds = xtreme_ds.tydiqaTrainDataset()
for i, each in enumerate(ds):
    for j, s_p in enumerate(each["start_positions"]):
        reply = xtreme_ds.tokenizer.convert_tokens_to_string(
            xtreme_ds.tokenizer.convert_ids_to_tokens(
                each["tokens"][each["start_positions"][j] : each["end_positions"][j]]
            )
        )
        answer = xtreme_ds.normalize_string(each["answers"]["text"][j])
        metrics.add(
            prediction={"id": each["id"], "prediction_text": " ".join(reply)},
            reference={
                "id": each["id"],
                "answers": {
                    "text": [
                        " ".join(xtreme_ds.normalize_string(each["answers"]["text"][j]))
                    ],
                    "answer_start": [each["answers"]["answer_start"][j]],
                },
            },
        )
print(metrics.compute())
assert i == len(ds) - 1
ds = xtreme_ds.tydiqaTestDataset()
for i, each in enumerate(ds):
    for j, s_p in enumerate(each["start_positions"]):
        reply = xtreme_ds.tokenizer.convert_tokens_to_string(
            xtreme_ds.tokenizer.convert_ids_to_tokens(
                each["tokens"][each["start_positions"][j] : each["end_positions"][j]]
            )
        )
        answer = xtreme_ds.normalize_string(each["answers"]["text"][j])
        metrics.add(
            prediction={"id": each["id"], "prediction_text": " ".join(reply)},
            reference={
                "id": each["id"],
                "answers": {
                    "text": [
                        " ".join(xtreme_ds.normalize_string(each["answers"]["text"][j]))
                    ],
                    "answer_start": [each["answers"]["answer_start"][j]],
                },
            },
        )
print(metrics.compute())
assert i == len(ds) - 1


ds = xtreme_ds.bucc2018Dataset()
for i, each in enumerate(ds):
    pass
assert i == len(ds) - 1
ds = xtreme_ds.tatoebaDataset()
for i, each in enumerate(ds):
    pass
assert i == len(ds) - 1
print("check done")
