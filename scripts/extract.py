from experiment_datasets import xtreme_ds
import json
from experiment_util.experiment_config import ExperimentConfigSerializer
import pickle

exp_config = json.loads(
    """{
    "discriminators": [
        {
            "dtype": "SingleToken",
            "length": 744,
            "weight": 0.00012,
            "num_labels": 40,
            "label_id": "language_id",
            "hidden_size": 384
        },
        {
            "dtype": "FullSequence",
            "length": 384,
            "weight": 0.00006,
            "num_labels": 40,
            "label_id": "language_id",
            "hidden_size": 96,
            "nhead": 4,
            "num_layers": 3
        },
        {
            "dtype": "FullSequence",
            "length": 288,
            "weight": 0.00006,
            "num_labels": 14,
            "label_id": "family_label",
            "hidden_size": 96,
            "nhead": 4,
            "num_layers": 3
        },
        {
            "dtype": "FullSequence",
            "length": 192,
            "weight": 0.00006,
            "num_labels": 25,
            "label_id": "genus_label",
            "hidden_size": 96,
            "nhead": 4,
            "num_layers": 3
        }
    ],
    "training": {
        "backbone_name": "xlm-roberta-large",
        "model_name": "mlm_disentangle_default",
        "gradient_acc_size": 16,
        "batch_size": 2,
        "max_step": 3375,
        "log_step": 225,
        "num_frozen_layers": 18,
        "mlm_lr": 4e-4,
        "mlm_beta1": 0.9,
        "mlm_beta2": 0.98,
        "mlm_eps": 1e-6
    }
}""",
    cls=ExperimentConfigSerializer,
)
from transformers import XLMRobertaConfig, XLMRobertaForMaskedLM
from experiment_util.experiment_config import ExperimentConfigSerializer
from experiment_models.disentangled_transformer import XLMRobertaForDisentanglement
from experiment_models.multitask_transformer import MultitaskModel
import json

XLMRConfig = XLMRobertaConfig.from_pretrained("xlm-roberta-large")
setattr(XLMRConfig, "discriminators", exp_config["discriminators"])
model = MultitaskModel.create_untrained(
    backbone_name="xlm-roberta-large",
    task_dict={
        "mlm": {
            "type": XLMRobertaForMaskedLM,
            "config": XLMRobertaConfig.from_pretrained("xlm-roberta-large"),
        },
        "disentangle": {
            "type": XLMRobertaForDisentanglement,
            "config": XLMRConfig,
        },
    },
)
import torch

model.load_state_dict(
    torch.load(
        "/gpfs1/home/ckchan666/mlm_disentangle_experiment/model/mlm/default/pytorch_model.bin"
    )
)
disentangled_model = model.backbone
from os import path

have_bucc2018_source = path.exists(
    "/gpfs1/scratch/ckchan666/pickle/bucc2018_source.pickle"
)
have_bucc2018_target = path.exists(
    "/gpfs1/scratch/ckchan666/pickle/bucc2018_target.pickle"
)
if not have_bucc2018_source or not have_bucc2018_target:
    ds = xtreme_ds.bucc2018Dataset()
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=2,
        num_workers=2,
        shuffle=False,
    )
    bucc2018_source = {}
    bucc2018_target = {}
    import gc

    for lan in xtreme_ds.TASK2LANGS[ds.task]:
        bucc2018_source[lan] = {}
        bucc2018_target[lan] = {}
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            if not have_bucc2018_source:
                source_output = disentangled_model(batch["source_tokens"])
                for i, s_id in enumerate(batch["source_id"]):
                    lan = batch["lan"][i]
                    bucc2018_source[lan][s_id] = source_output.last_hidden_state[i][
                        batch["source_tokens"][i] != xtreme_ds.tokenizer.pad_token_id
                    ]
                    bucc2018_source[lan][s_id] = torch.mean(
                        bucc2018_source[lan][s_id], dim=0
                    )
            if not have_bucc2018_target:
                target_output = disentangled_model(batch["target_tokens"])
                for i, s_id in enumerate(batch["target_id"]):
                    lan = batch["lan"][i]
                    bucc2018_target[lan][s_id] = target_output.last_hidden_state[i][
                        batch["target_tokens"][i] != xtreme_ds.tokenizer.pad_token_id
                    ]
                    bucc2018_target[lan][s_id] = torch.mean(
                        bucc2018_target[lan][s_id], dim=0
                    )
    if not have_bucc2018_source:
        filehandler = open(
            "/gpfs1/scratch/ckchan666/pickle/bucc2018_source.pickle", "wb"
        )
        pickle.dump(bucc2018_source, filehandler)
        del bucc2018_source
    if not have_bucc2018_target:
        filehandler = open(
            "/gpfs1/scratch/ckchan666/pickle/bucc2018_target.pickle", "wb"
        )
        pickle.dump(bucc2018_target, filehandler)
        del bucc2018_target
    del ds

have_tatoeba_source = path.exists(
    "/gpfs1/scratch/ckchan666/pickle/tatoeba_source.pickle"
)
have_tatoeba_target = path.exists(
    "/gpfs1/scratch/ckchan666/pickle/tatoeba_target.pickle"
)
if not have_tatoeba_source or not have_tatoeba_target:
    ds = xtreme_ds.tatoebaDataset()
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=2,
        num_workers=2,
        shuffle=False,
    )
    tatoeba_source = {}
    tatoeba_target = {}
    import gc

    for lan in xtreme_ds.TASK2LANGS[ds.task]:
        tatoeba_source[lan] = {}
        tatoeba_target[lan] = {}
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            if not have_tatoeba_source:
                source_output = disentangled_model(batch["source_tokens"])
                for i, s_id in enumerate(batch["source_id"]):
                    lan = batch["lan"][i]
                    tatoeba_source[lan][s_id] = source_output.last_hidden_state[i][
                        batch["source_tokens"][i] != xtreme_ds.tokenizer.pad_token_id
                    ]
                    tatoeba_source[lan][s_id] = torch.mean(
                        tatoeba_source[lan][s_id], dim=0
                    )
            if not have_tatoeba_target:
                target_output = disentangled_model(batch["target_tokens"])
                for i, s_id in enumerate(batch["target_id"]):
                    lan = batch["lan"][i]
                    tatoeba_target[lan][s_id] = target_output.last_hidden_state[i][
                        batch["target_tokens"][i] != xtreme_ds.tokenizer.pad_token_id
                    ]
                    tatoeba_target[lan][s_id] = torch.mean(
                        tatoeba_target[lan][s_id], dim=0
                    )
    if not have_tatoeba_source:
        filehandler = open(
            "/gpfs1/scratch/ckchan666/pickle/tatoeba_source.pickle", "wb"
        )
        pickle.dump(tatoeba_source, filehandler)
        del tatoeba_source
    if not have_tatoeba_target:
        filehandler = open(
            "/gpfs1/scratch/ckchan666/pickle/tatoeba_target.pickle", "wb"
        )
        pickle.dump(tatoeba_target, filehandler)
        del tatoeba_target
    del ds

if not path.exists("/gpfs1/scratch/ckchan666/pickle/udpos_example.pickle"):
    ds = xtreme_ds.udposTestDataset()
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=2,
        num_workers=2,
        shuffle=False,
    )
    udpos_example = {}
    import gc

    for i_batch, batch in enumerate(dataloader):
        with torch.no_grad():
            output = disentangled_model(batch["tokens"])
            for i, s_id in enumerate(batch["tokens"]):
                udpos_example[i_batch] = {
                    "vectors": output.last_hidden_state[i],
                    "tokens": batch["tokens"][i],
                    "tags": batch["tags"][i],
                    "lan": batch["lan"][i],
                    "offset": batch["offset"][i],
                }
    filehandler = open("/gpfs1/scratch/ckchan666/pickle/udpos_example.pickle", "wb")
    pickle.dump(udpos_example, filehandler)
