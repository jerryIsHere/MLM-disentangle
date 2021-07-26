from experiment_datasets import xtreme_ds
import json
from experiment_util.experiment_config import ExperimentConfigSerializer
import pickle
from transformers import XLMRobertaConfig, XLMRobertaForMaskedLM
from experiment_util.experiment_config import ExperimentConfigSerializer
from experiment_models.disentangled_transformer import XLMRobertaForDisentanglement
from experiment_models.multitask_transformer import MultitaskModel
import json
import transformers


model = transformers.XLMRobertaModel.from_pretrained("xlm-roberta-large")
import torch

from os import path

examples_lan = ["vi", "yo"]  # ["eu", "he", "hu", "ja", "kk", "ko", "th", "vi"]

ds = xtreme_ds.udposTestDataset()
dataloader = torch.utils.data.DataLoader(
    ds,
    batch_size=2,
    num_workers=2,
    shuffle=False,
)
udpos_example = []
import gc

last_lan = ""
for i_batch, batch in enumerate(dataloader):
    with torch.no_grad():
        output = model(batch["tokens"])
        for i, s_id in enumerate(batch["tokens"]):
            lan = batch["lan"][i]
            if lan not in examples_lan:
                continue
            if last_lan != lan and last_lan != "":
                filehandler = open(
                    "/gpfs1/scratch/ckchan666/pickle/control_udpos_example_"
                    + last_lan
                    + ".pickle",
                    "wb",
                )
                pickle.dump(udpos_example, filehandler)
                udpos_example = []
                gc.collect()
            vectors = output.last_hidden_state[i]
            tags = batch["tags"][i]
            vectors = vectors[tags != -100, :]
            udpos_example.append(
                {
                    "vectors": vectors,
                    "tags": tags[tags != -100],
                }
            )
            last_lan = lan
    filehandler = open(
        "/gpfs1/scratch/ckchan666/pickle/control_udpos_example_" + last_lan + ".pickle",
        "wb",
    )
    pickle.dump(udpos_example, filehandler)


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
ds = xtreme_ds.udposTestDataset()
dataloader = torch.utils.data.DataLoader(
    ds,
    batch_size=2,
    num_workers=2,
    shuffle=False,
)
udpos_example = []
import gc
for i_batch, batch in enumerate(dataloader):
    with torch.no_grad():
        output = disentangled_model(batch["tokens"])
        for i, s_id in enumerate(batch["tokens"]):
            lan = batch["lan"][i]
            if lan != 'yo':
                continue
            tags = batch["tags"][i]
            vectors = vectors[tags != -100, :]
            udpos_example.append(
                {
                    "vectors": vectors,
                    "tags": tags[tags != -100],
                }
            )
filehandler = open(
                        "/gpfs1/scratch/ckchan666/pickle/udpos_example_"
                        + "yo"
                        + ".pickle",
                        "wb",
                    )
pickle.dump(udpos_example, filehandler)