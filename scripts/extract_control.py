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

examples_lan = ["eu", "he", "hu", "ja", "kk", "ko", "th", "vi"]

if not path.exists("/gpfs1/scratch/ckchan666/pickle/control_udpos_example.pickle"):
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
