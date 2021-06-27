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
    for lan in ds.dataset:
        length = len(ds.dataset[lan])
        if instnace_id < length:
            break
        instnace_id -= length
    assert each["label"] == xtreme_ds.pawsxTrainDataset.class_label.index(
        ds.dataset[lan][id]["label"]
    )
assert i == len(ds) - 1


import datasets
from experiment.xquad.xquad_experiment import normalize_string, normalize_ids

squad_metrics = {}
instnace_ids = {}
ds = xtreme_ds.xquadTrainDataset()
squad_metrics[ds.__class__.__name__] = datasets.load_metric("squad")
instnace_ids[ds.__class__.__name__] = set()
for i, each in enumerate(ds):
    instnace_ids[ds.__class__.__name__].add(each["features"]["id"])
    reply = normalize_ids(
        each["tokens"][each["start_positions"] : each["end_positions"]]
    )

    squad_metrics[ds.__class__.__name__].add(
        prediction={
            "id": each["features"]["id"],
            "prediction_text": reply,
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
squad_metrics[ds.__class__.__name__] = datasets.load_metric("squad")
instnace_ids[ds.__class__.__name__] = set()
for i, each in enumerate(ds):
    instnace_ids[ds.__class__.__name__].add(each["features"]["id"])
    reply = normalize_ids(
        each["tokens"][each["start_positions"] : each["end_positions"]]
    )

    squad_metrics[ds.__class__.__name__].add(
        prediction={
            "id": each["features"]["id"],
            "prediction_text": reply,
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
squad_metrics[ds.__class__.__name__] = datasets.load_metric("squad")
instnace_ids[ds.__class__.__name__] = set()
for i, each in enumerate(ds):
    instnace_ids[ds.__class__.__name__].add(each['lan'] + each["features"]["id"])
    reply = normalize_ids(
        each["tokens"][each["start_positions"] : each["end_positions"]]
    )

    squad_metrics[ds.__class__.__name__].add(
        prediction={
            "id": each["features"]["id"],
            "prediction_text": reply,
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
assert len(instnace_ids[ds.__class__.__name__]) == sum(
    map(len, ds.dataset.values())
)  # each question is at least answered once


ds = xtreme_ds.mlqaTestDataset()
squad_metrics[ds.__class__.__name__] = datasets.load_metric("squad")
instnace_ids[ds.__class__.__name__] = set()
for i, each in enumerate(ds):
    instnace_ids[ds.__class__.__name__].add(each["features"]["id"])
    reply = normalize_ids(
        each["tokens"][each["start_positions"] : each["end_positions"]]
    )

    squad_metrics[ds.__class__.__name__].add(
        prediction={
            "id": each["features"]["id"],
            "prediction_text": reply,
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
assert len(instnace_ids[ds.__class__.__name__]) == sum(
    map(len, ds.dataset.values())
)  # each question is at least answered once


ds = xtreme_ds.tydiqaTrainDataset()
squad_metrics[ds.__class__.__name__] = datasets.load_metric("squad")
instnace_ids[ds.__class__.__name__] = set()
for i, each in enumerate(ds):
    instnace_ids[ds.__class__.__name__].add(each["features"]["id"])
    reply = normalize_ids(
        each["tokens"][each["start_positions"] : each["end_positions"]]
    )

    squad_metrics[ds.__class__.__name__].add(
        prediction={
            "id": each["features"]["id"],
            "prediction_text": reply,
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
ds = xtreme_ds.tydiqaTestDataset()
squad_metrics[ds.__class__.__name__] = datasets.load_metric("squad")
instnace_ids[ds.__class__.__name__] = set()
for i, each in enumerate(ds):
    instnace_ids[ds.__class__.__name__].add(each["features"]["id"])
    reply = normalize_ids(
        each["tokens"][each["start_positions"] : each["end_positions"]]
    )

    squad_metrics[ds.__class__.__name__].add(
        prediction={
            "id": each["features"]["id"],
            "prediction_text": reply,
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


ds = xtreme_ds.bucc2018Dataset()
for i, each in enumerate(ds):
    pass
assert i == len(ds) - 1
ds = xtreme_ds.tatoebaDataset()
for i, each in enumerate(ds):
    pass
assert i == len(ds) - 1
print("check done")
for dataset_name in squad_metrics:
    print(dataset_name)
    print(squad_metrics[dataset_name].compute())
