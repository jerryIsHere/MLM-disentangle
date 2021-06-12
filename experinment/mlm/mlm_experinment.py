from experinment_util.experinment_config import ExperinmentConfigSerializer
from experinment_models.multitask_transformer import MultitaskModel
from experinment_models.disentangled_transformer import XLMRobertaForDisentanglement
from experinment_datasets import oscar_corpus
import transformers
import json
import argparse
import torch
import time
import os

start_time = time.time()
parser = argparse.ArgumentParser(description="token frequency of 40 corpus")
parser.add_argument(
    "--config_json",
    metavar="path",
    type=str,
    help="path to configuration json file",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "/default.json"),
)
parser.add_argument(
    "time",
    metavar="hh:mm:ss",
    type=str,
    help="time limit for this script",
)
args = parser.parse_args()
args.time = sum([a * b for a, b in zip([3600, 60, 1], map(int, args.time.split(":")))])
with open(args.config_json, "r") as outfile:
    experinment_config_dict = json.load(outfile, cls=ExperinmentConfigSerializer)
experinment_config_dict["training"].model_name = (
    os.path.abspath(args.config_json).split("/")[-1].split(".")[0]
)

backbone_name = experinment_config_dict["training"].backbone_name
XLMRConfig = transformers.AutoConfig.from_pretrained(backbone_name)

setattr(XLMRConfig, "discriminators", experinment_config_dict["discriminators"])

multitask_model = MultitaskModel.create(
    backbone_name=backbone_name,
    task_dict={
        "mlm": {
            "type": transformers.XLMRobertaForMaskedLM,
            "config": transformers.XLMRobertaConfig.from_pretrained(backbone_name),
        },
        "disentangle": {
            "type": XLMRobertaForDisentanglement,
            "config": XLMRConfig,
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
    lr=experinment_config_dict["training"].mlm_lr,
    betas=(
        experinment_config_dict["training"].mlm_beta1,
        experinment_config_dict["training"].mlm_beta2,
    ),
    eps=experinment_config_dict["training"].mlm_eps,
)
optimizerdisentangle = torch.optim.Adam(
    multitask_model.taskmodels_dict["disentangle"].parameters(),
    lr=experinment_config_dict["training"].disentangle_lr,
    betas=(
        experinment_config_dict["training"].disentangle_beta1,
        experinment_config_dict["training"].disentangle_beta2,
    ),
    eps=experinment_config_dict["training"].disentangle_eps,
)
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(
    "/gpfs1/home/ckchan666/mlm_disentangle_experinment/tensorboard/"
    + os.path.dirname(os.path.abspath(__file__)).split("/")[-1]
    + "/"
    + experinment_config_dict["training"].model_name
)

MLMD_ds = oscar_corpus.MLMDisentangleDataset()
dataloader = torch.utils.data.DataLoader(
    MLMD_ds, batch_size=experinment_config_dict["training"].batch_size, num_workers=0
)

import gc

for task in multitask_model.taskmodels_dict:
    multitask_model.taskmodels_dict[task].cpu()
mlmLoss = 0.0
disentangleLoss = 0.0
gradient_step = 0
print("building time: " + str(time.time() - start_time) + "s")
start_time = time.time()
print(
    "run for "
    + str(experinment_config_dict["training"].max_step)
    + " step with "
    + str(experinment_config_dict["training"].gradient_acc_size)
    + " gradient acc size"
)
for i, batch in enumerate(dataloader):
    # mlm input to gpu
    batch["masked_tokens"] = batch["masked_tokens"].cuda()
    batch["tokens"] = batch["tokens"].cuda()

    # mlm model to gpu
    multitask_model.taskmodels_dict["mlm"].cuda()
    mlmOutput = multitask_model.taskmodels_dict["mlm"](
        input_ids=batch["masked_tokens"],
        labels=batch["tokens"],
    )
    mlmOutput["loss"].backward()

    # mlm model & input to cpu
    mlmLoss = mlmLoss + mlmOutput["loss"].item()
    multitask_model.taskmodels_dict["mlm"].cpu()
    del mlmOutput
    del batch["tokens"]

    # disentangle input to gpu
    batch["language_id"] = batch["language_id"].cuda()
    batch["genus_label"] = batch["genus_label"].cuda()
    batch["family_label"] = batch["family_label"].cuda()

    # disentangle model to gpu
    multitask_model.taskmodels_dict["disentangle"].cuda()
    disentangleOutput = multitask_model.taskmodels_dict["disentangle"](
        input_ids=batch["masked_tokens"],
        labels={
            "language_id": batch["language_id"],
            "genus_label": batch["genus_label"],
            "family_label": batch["family_label"],
        },
    )
    disentangleOutput["loss"].backward()

    # disentangle model & input to cpu
    disentangleLoss = disentangleLoss + disentangleOutput["loss"].item()
    multitask_model.taskmodels_dict["disentangle"].cpu()
    for output in disentangleOutput:
        if disentangleOutput[output] == None:
            continue
        if output == "logits":
            disentangleOutput[output].clear()
    del disentangleOutput
    batch.clear()
    del batch

    if (i + 1) % (
        experinment_config_dict["training"].gradient_acc_size
        / experinment_config_dict["training"].batch_size
    ) == 0:
        optimizermlm.step()
        multitask_model.taskmodels_dict["mlm"].zero_grad()
        optimizerdisentangle.step()
        multitask_model.taskmodels_dict["disentangle"].zero_grad()
        gradient_step += 1
        if gradient_step == 1:
            print(
                "mlm loss ("
                + str(gradient_step)
                + "): "
                + str(mlmLoss / experinment_config_dict["training"].log_step)
            )
            print(
                "disentangle loss ("
                + str(gradient_step)
                + "): "
                + str(disentangleLoss / experinment_config_dict["training"].log_step)
            )
        if gradient_step % experinment_config_dict["training"].log_step == 0:
            # writer.add_scalar("mlm lr", scheduler.get_lr()[0], global_step)
            # writer.add_scalar("disentangle lr", scheduler.get_lr()[0], global_step)
            print(
                "mlm loss ("
                + str(gradient_step)
                + "): "
                + str(mlmLoss / experinment_config_dict["training"].log_step)
            )
            print(
                "disentangle loss ("
                + str(gradient_step)
                + "): "
                + str(disentangleLoss / experinment_config_dict["training"].log_step)
            )
            writer.add_scalar(
                "mlm loss",
                mlmLoss / experinment_config_dict["training"].log_step,
                gradient_step,
            )
            writer.add_scalar(
                "disentangle loss",
                disentangleLoss / experinment_config_dict["training"].log_step,
                gradient_step,
            )
            mlmLoss = 0
            disentangleLoss = 0
            multitask_model.save_pretrained(
                "./" + experinment_config_dict["training"].model_name,
            )
        if gradient_step >= experinment_config_dict["training"].max_step:
            break
        if time.time() - start_time > 0.9 * args.time:
            print(str(time.time() - start_time) + "s exceed 0.9 of the time limit")
            print(str(gradient_step) + "th gradient step")
            break
    gc.collect()


multitask_model.save_pretrained(
    "/gpfs1/home/ckchan666/mlm_disentangle_experinment/model/"
    + os.path.dirname(os.path.abspath(__file__)).split("/")[-1]
    + "/"
    + experinment_config_dict["training"].model_name,
)
print(str(time.time() - start_time) + " seconds elapsed")
from resource import getrusage, RUSAGE_SELF

print(str(getrusage(RUSAGE_SELF).ru_maxrss) + "KB used (peak)")