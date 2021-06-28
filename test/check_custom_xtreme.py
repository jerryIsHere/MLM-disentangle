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
assert i >= len(ds.dataset) - 1
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
assert i >= len(ds.dataset) - 1
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
assert i >= sum(map(len, ds.dataset.values())) - 1


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
assert i >= len(ds.dataset) - 1
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
assert i >= len(ds.dataset) - 1
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
assert i >= sum(map(len, ds.dataset.values())) - 1


ds = xtreme_ds.xnliTrainDataset()
for i, each in enumerate(ds):
    assert each["label"] == xtreme_ds.xnliTrainDataset.class_label.index(
        ds.dataset[i]["label"]
    )
assert i == len(ds) - 1
ds = xtreme_ds.xnliValidationDataset()
for i, each in enumerate(ds):
    instnace_id = i
    for split in ds.dataset:
        length = len(ds.dataset[split])
        if instnace_id < length:
            break
        instnace_id -= length
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
    instnace_id = i
    for lan in xtreme_ds.TASK2LANGS[ds.task]:
        length = len(ds.dataset[lan])
        if instnace_id < length:
            break
        instnace_id -= length
    assert each["label"] == xtreme_ds.pawsxTrainDataset.class_label.index(
        ds.dataset[lan][id]["label"]
    )
assert i == len(ds) - 1


import datasets
from experiment.xquad.xquad_experiment import (
    normalize_string,
    normalize_ids,
    stringify_ids,
)

squad_metrics_normal = {}
squad_metrics_custom = {}
instnace_ids = {}
ds = xtreme_ds.xquadTrainDataset()
squad_metrics_normal[ds.__class__.__name__] = datasets.load_metric("squad")
squad_metrics_custom[ds.__class__.__name__] = datasets.load_metric("squad")
instnace_ids[ds.__class__.__name__] = set()
for i, each in enumerate(ds):
    instnace_ids[ds.__class__.__name__].add(each["features"]["id"])
    reply_ids = each["tokens"][each["start_positions"] : each["end_positions"]]

    squad_metrics_normal[ds.__class__.__name__].add(
        prediction={
            "id": each["features"]["id"],
            "prediction_text": stringify_ids(reply_ids),
        },
        reference={
            "id": each["features"]["id"],
            "answers": {
                "text": [ans for ans in each["features"]["answers"]["text"]],
                "answer_start": each["features"]["answers"]["answer_start"],
            },
        },
    )
    squad_metrics_custom[ds.__class__.__name__].add(
        prediction={
            "id": each["features"]["id"],
            "prediction_text": normalize_ids(reply_ids),
        },
        reference={
            "id": each["features"]["id"],
            "answers": {
                "text": [
                    normalize_string(ans) for ans in each["features"]["answers"]["text"]
                ],
                "answer_start": each["features"]["answers"]["answer_start"],
            },
        },
    )
assert len(instnace_ids[ds.__class__.__name__]) == len(
    ds.dataset
)  # each question is at least answered once
ds = xtreme_ds.xquadValidationDataset()
squad_metrics_normal[ds.__class__.__name__] = datasets.load_metric("squad")
squad_metrics_custom[ds.__class__.__name__] = datasets.load_metric("squad")
instnace_ids[ds.__class__.__name__] = set()
for i, each in enumerate(ds):
    instnace_ids[ds.__class__.__name__].add(each["features"]["id"])
    reply_ids = each["tokens"][each["start_positions"] : each["end_positions"]]

    squad_metrics_normal[ds.__class__.__name__].add(
        prediction={
            "id": each["features"]["id"],
            "prediction_text": stringify_ids(reply_ids),
        },
        reference={
            "id": each["features"]["id"],
            "answers": {
                "text": [ans for ans in each["features"]["answers"]["text"]],
                "answer_start": each["features"]["answers"]["answer_start"],
            },
        },
    )
    squad_metrics_custom[ds.__class__.__name__].add(
        prediction={
            "id": each["features"]["id"],
            "prediction_text": normalize_ids(reply_ids),
        },
        reference={
            "id": each["features"]["id"],
            "answers": {
                "text": [
                    normalize_string(ans) for ans in each["features"]["answers"]["text"]
                ],
                "answer_start": each["features"]["answers"]["answer_start"],
            },
        },
    )
assert len(instnace_ids[ds.__class__.__name__]) == len(
    ds.dataset
)  # each question is at least answered once
ds = xtreme_ds.xquadTestDataset()
squad_metrics_normal[ds.__class__.__name__] = {}
squad_metrics_custom[ds.__class__.__name__] = {}
instnace_ids[ds.__class__.__name__] = {}
for lan in xtreme_ds.TASK2LANGS[ds.task]:
    squad_metrics_normal[ds.__class__.__name__][lan] = datasets.load_metric("squad")
    squad_metrics_custom[ds.__class__.__name__][lan] = datasets.load_metric("squad")
    instnace_ids[ds.__class__.__name__][lan] = set()
for i, each in enumerate(ds):
    lan = each["lan"]
    instnace_ids[ds.__class__.__name__][lan].add(each["features"]["id"])
    reply_ids = each["tokens"][each["start_positions"] : each["end_positions"]]

    squad_metrics_normal[ds.__class__.__name__][lan].add(
        prediction={
            "id": each["features"]["id"],
            "prediction_text": stringify_ids(reply_ids),
        },
        reference={
            "id": each["features"]["id"],
            "answers": {
                "text": [ans for ans in each["features"]["answers"]["text"]],
                "answer_start": each["features"]["answers"]["answer_start"],
            },
        },
    )
    squad_metrics_custom[ds.__class__.__name__][lan].add(
        prediction={
            "id": each["features"]["id"],
            "prediction_text": normalize_ids(reply_ids),
        },
        reference={
            "id": each["features"]["id"],
            "answers": {
                "text": [
                    normalize_string(ans) for ans in each["features"]["answers"]["text"]
                ],
                "answer_start": each["features"]["answers"]["answer_start"],
            },
        },
    )
for lan in xtreme_ds.TASK2LANGS[ds.task]:
    assert len(instnace_ids[ds.__class__.__name__][lan]) == len(ds.dataset[lan])
    # each question is at least answered once


ds = xtreme_ds.mlqaTestDataset()
squad_metrics_normal[ds.__class__.__name__] = {}
squad_metrics_custom[ds.__class__.__name__] = {}
instnace_ids[ds.__class__.__name__] = {}
for lan in xtreme_ds.TASK2LANGS[ds.task]:
    squad_metrics_normal[ds.__class__.__name__][lan] = datasets.load_metric("squad")
    squad_metrics_custom[ds.__class__.__name__][lan] = datasets.load_metric("squad")
    instnace_ids[ds.__class__.__name__][lan] = set()
for i, each in enumerate(ds):
    lan = each["lan"]
    instnace_ids[ds.__class__.__name__][lan].add(each["features"]["id"])
    reply_ids = each["tokens"][each["start_positions"] : each["end_positions"]]

    squad_metrics_normal[ds.__class__.__name__][lan].add(
        prediction={
            "id": each["features"]["id"],
            "prediction_text": stringify_ids(reply_ids),
        },
        reference={
            "id": each["features"]["id"],
            "answers": {
                "text": [ans for ans in each["features"]["answers"]["text"]],
                "answer_start": each["features"]["answers"]["answer_start"],
            },
        },
    )
    squad_metrics_custom[ds.__class__.__name__][lan].add(
        prediction={
            "id": each["features"]["id"],
            "prediction_text": normalize_ids(reply_ids),
        },
        reference={
            "id": each["features"]["id"],
            "answers": {
                "text": [
                    normalize_string(ans) for ans in each["features"]["answers"]["text"]
                ],
                "answer_start": each["features"]["answers"]["answer_start"],
            },
        },
    )
for lan in xtreme_ds.TASK2LANGS[ds.task]:
    assert len(instnace_ids[ds.__class__.__name__][lan]) == len(ds.dataset[lan])
    # each question is at least answered once


ds = xtreme_ds.tydiqaTrainDataset()
squad_metrics_normal[ds.__class__.__name__] = datasets.load_metric("squad")
squad_metrics_custom[ds.__class__.__name__] = datasets.load_metric("squad")
instnace_ids[ds.__class__.__name__] = set()
for i, each in enumerate(ds):
    instnace_ids[ds.__class__.__name__].add(each["features"]["id"])
    reply_ids = each["tokens"][each["start_positions"] : each["end_positions"]]

    squad_metrics_normal[ds.__class__.__name__].add(
        prediction={
            "id": each["features"]["id"],
            "prediction_text": stringify_ids(reply_ids),
        },
        reference={
            "id": each["features"]["id"],
            "answers": {
                "text": [ans for ans in each["features"]["answers"]["text"]],
                "answer_start": each["features"]["answers"]["answer_start"],
            },
        },
    )
    squad_metrics_custom[ds.__class__.__name__].add(
        prediction={
            "id": each["features"]["id"],
            "prediction_text": normalize_ids(reply_ids),
        },
        reference={
            "id": each["features"]["id"],
            "answers": {
                "text": [
                    normalize_string(ans) for ans in each["features"]["answers"]["text"]
                ],
                "answer_start": each["features"]["answers"]["answer_start"],
            },
        },
    )
assert len(instnace_ids[ds.__class__.__name__]) == len(
    [
        instance
        for instance in ds.dataset
        if xtreme_ds.LANG2ISO[instance["id"].split("-")[0]] == "en"
    ]
)  # each question is at least answered once
ds = xtreme_ds.tydiqaTestDataset()
squad_metrics_normal[ds.__class__.__name__] = {}
squad_metrics_custom[ds.__class__.__name__] = {}
instnace_ids[ds.__class__.__name__] = {}
for lan in xtreme_ds.TASK2LANGS[ds.task]:
    squad_metrics_normal[ds.__class__.__name__][lan] = datasets.load_metric("squad")
    squad_metrics_custom[ds.__class__.__name__][lan] = datasets.load_metric("squad")
    instnace_ids[ds.__class__.__name__][lan] = set()
for i, each in enumerate(ds):
    lan = each["lan"]
    instnace_ids[ds.__class__.__name__][lan].add(each["features"]["id"])
    reply_ids = each["tokens"][each["start_positions"] : each["end_positions"]]

    squad_metrics_normal[ds.__class__.__name__][lan].add(
        prediction={
            "id": each["features"]["id"],
            "prediction_text": stringify_ids(reply_ids),
        },
        reference={
            "id": each["features"]["id"],
            "answers": {
                "text": [ans for ans in each["features"]["answers"]["text"]],
                "answer_start": each["features"]["answers"]["answer_start"],
            },
        },
    )
    squad_metrics_custom[ds.__class__.__name__][lan].add(
        prediction={
            "id": each["features"]["id"],
            "prediction_text": normalize_ids(reply_ids),
        },
        reference={
            "id": each["features"]["id"],
            "answers": {
                "text": [
                    normalize_string(ans) for ans in each["features"]["answers"]["text"]
                ],
                "answer_start": each["features"]["answers"]["answer_start"],
            },
        },
    )
for lan in xtreme_ds.TASK2LANGS[ds.task]:
    assert len(instnace_ids[ds.__class__.__name__][lan]) == len(
        [
            instance
            for instance in ds.dataset
            if xtreme_ds.LANG2ISO[instance["id"].split("-")[0]] == lan
        ]
    )
    # each question is at least answered once


ds = xtreme_ds.bucc2018Dataset()
for i, each in enumerate(ds):
    pass
assert i == len(ds) - 1
ds = xtreme_ds.tatoebaDataset()
for i, each in enumerate(ds):
    pass
assert i == len(ds) - 1
print("check done")
for dataset_name in squad_metrics_normal:
    print(dataset_name)
    if "Test" in dataset_name:
        for lan in squad_metrics_normal[dataset_name]:
            print(lan)
            print(squad_metrics_normal[dataset_name][lan].compute())
    else:
        print(squad_metrics_normal[dataset_name].compute())
    print()

for dataset_name in squad_metrics_custom:
    print(dataset_name)
    if "Test" in dataset_name:
        for lan in squad_metrics_custom[dataset_name]:
            print(lan)
            print(squad_metrics_custom[dataset_name][lan].compute())
    else:
        print(squad_metrics_custom[dataset_name].compute())
    print()
