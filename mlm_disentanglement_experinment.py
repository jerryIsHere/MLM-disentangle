from disentangled_transformer import (
    MultitaskModel,
    XLMRobertaForDisentanglement,
    DiscriminatorType,
    DiscriminatorConfig,
    ExperinmentConfigSerializer,
)
import oscar_corpus
import transformers
import json
import argparse
import torch
import time

start_time = time.time()
parser = argparse.ArgumentParser(description="token frequency of 40 corpus")
parser.add_argument(
    "--config_json",
    metavar="path",
    type=str,
    help="path to configuration json file",
    default="config/default_experinment.json",
)
parser.add_argument(
    "time",
    metavar="hh:mm:ss",
    type=str,
    help="time limit for this script",
)
parser.add_argument(
    "--log_step",
    metavar="step",
    type=int,
    help="",
    default=1000,
)
parser.add_argument(
    "--max_step",
    metavar="step",
    type=int,
    help="",
    default=10000,
)
args = parser.parse_args()
args.time = sum([a * b for a, b in zip([3600, 60, 1], map(int, args.time.split(":")))])

backbone_name = "xlm-roberta-large"
XLMRobertaConfig = transformers.AutoConfig.from_pretrained(backbone_name)

with open(args.config_json, "r") as outfile:
    experinment_config_dict = json.load(outfile, cls=ExperinmentConfigSerializer)
setattr(XLMRobertaConfig, "discriminators", experinment_config_dict["discriminators"])

multitask_model = MultitaskModel.create(
    backbone_name="xlm-roberta-large",
    task_dict={
        "mlm": {
            "type": transformers.AutoModelForMaskedLM,
            "config": transformers.AutoConfig.from_pretrained(backbone_name),
        },
        "disentangle": {
            "type": XLMRobertaForDisentanglement,
            "config": XLMRobertaConfig,
        },
    },
)

for layers in multitask_model.backbone.encoder.layer[
    : experinment_config_dict["training"].num_frozen_layers
]:  # 0:24
    for param in layers.parameters():
        param.requires_grad = False
optimizermlm = torch.optim.Adam(
    multitask_model.taskmodels_dict["mlm"].parameters(),
    lr=4e-4,
    betas=(0.9, 0.98),
    eps=1e-6,
)
optimizerdisentangle = torch.optim.Adam(
    multitask_model.taskmodels_dict["disentangle"].parameters(),
    lr=4e-4,
    betas=(0.9, 0.98),
    eps=1e-6,
)
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(
    "/gpfs1/home/ckchan666/mlm_disentangle/tensorboard"
    + experinment_config_dict["training"].model_name
)

MLMD_ds = oscar_corpus.MLMDisentangleDataset()
dataloader = torch.utils.data.DataLoader(MLMD_ds, batch_size=4, num_workers=0)
for task in multitask_model.taskmodels_dict:
    multitask_model.taskmodels_dict[task].to(torch.device("cpu"))
mlmLoss = 0
disentangleLoss = 0
log_step = arg.log_step
max_step = arg.max_step
for i, batch in enumerate(dataloader):
    multitask_model.taskmodels_dict["disentangle"].zero_grad()
    multitask_model.taskmodels_dict["disentangle"].to(torch.device("cuda"))
    disentangleOutput = multitask_model.taskmodels_dict["disentangle"](
        input_ids=batch["masked_tokens"],
        labels={
            "language_id": batch["language_id"],
            "genus_label": batch["genus_label"],
            "family_label": batch["family_label"],
        },
    )
    disentangleOutput["loss"].backward()
    optimizerdisentangle.step()
    multitask_model.taskmodels_dict["disentangle"].to(torch.device("cpu"))

    multitask_model.taskmodels_dict["mlm"].zero_grad()
    multitask_model.taskmodels_dict["mlm"].to(torch.device("cuda"))
    mlmOutput = multitask_model.taskmodels_dict["mlm"](
        input_ids=batch["masked_tokens"],
        labels=batch["tokens"],
    )
    mlmOutput["loss"].backward()
    optimizermlm.step()
    multitask_model.taskmodels_dict["mlm"].to(torch.device("cpu"))
    mlmLoss = mlmLoss + mlmOutput["loss"]
    disentangleLoss = disentangleLoss + disentangleOutput["loss"]
    if i % log_step == log_step - 1:
        writer.add_scalar("mlm loss", mlmLoss / log_step, i)
        writer.add_scalar("disentangle loss", disentangleLoss / log_step, i)
        mlmLoss = 0
        disentangleLoss = 0
    if i > max_step or time.time() - start_time > 0.9 * args.time:
        break
