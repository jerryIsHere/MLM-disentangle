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
args = parser.parse_args()
args.time = sum([a * b for a, b in zip([3600, 60, 1], map(int, args.time.split(":")))])
with open(args.config_json, "r") as outfile:
    experinment_config_dict = json.load(outfile, cls=ExperinmentConfigSerializer)


backbone_name = "xlm-roberta-large"
XLMRobertaConfig = transformers.AutoConfig.from_pretrained(backbone_name)

setattr(XLMRobertaConfig, "discriminators", experinment_config_dict["discriminators"])

multitask_model = MultitaskModel.create(
    backbone_name=backbone_name,
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
    "/gpfs1/home/ckchan666/mlm_disentangle/tensorboard"
    + experinment_config_dict["training"].model_name
)

MLMD_ds = oscar_corpus.MLMDisentangleDataset()
dataloader = torch.utils.data.DataLoader(
    MLMD_ds, batch_size=experinment_config_dict["training"].batch_size, num_workers=0
)

for task in multitask_model.taskmodels_dict:
    multitask_model.taskmodels_dict[task].cpu()
mlmLoss = 0
disentangleLoss = 0
log_step = (
    experinment_config_dict["training"].log_step
    / experinment_config_dict["training"].batch_size
)
max_step = (
    experinment_config_dict["training"].max_step
    / experinment_config_dict["training"].batch_size
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
    multitask_model.taskmodels_dict["mlm"].cpu()
    for output in mlmOutput:
        mlmOutput[output] = mlmOutput[output].cpu()
    batch["tokens"] = batch["tokens"].cpu()

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
    multitask_model.taskmodels_dict["disentangle"].cpu()
    for output in disentangleOutput:
        if disentangleOutput[output] == None:
            continue
        if output == "logits":
            for j in disentangleOutput[output]:
                disentangleOutput[output][j] = disentangleOutput[output][j].cpu()
        else:
            disentangleOutput[output] = disentangleOutput[output].cpu()
    for item_name in batch:
        batch[item_name] = batch[item_name].cpu()

    # count loss
    mlmLoss = mlmLoss + mlmOutput["loss"]
    disentangleLoss = disentangleLoss + disentangleOutput["loss"]

    if (i + 1) % (
        experinment_config_dict["training"].gradient_acc_step
        / experinment_config_dict["training"].batch_size
    ) == 0:
        optimizermlm.step()
        multitask_model.taskmodels_dict["mlm"].zero_grad()
        optimizerdisentangle.step()
        multitask_model.taskmodels_dict["disentangle"].zero_grad()

    if (i + 1) % log_step == 0:
        writer.add_scalar("mlm loss", mlmLoss / log_step, i)
        writer.add_scalar("disentangle loss", disentangleLoss / log_step, i)
        mlmLoss = 0
        disentangleLoss = 0
    if i > max_step or time.time() - start_time > 0.9 * args.time:
        break

multitask_model.save_pretrained("./" + experinment_config_dict["training"].model_name,)