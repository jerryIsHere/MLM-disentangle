from experiment_util.experiment_config import ExperimentConfigSerializer
from experiment_models.multitask_transformer import MultitaskModel
from experiment_models.disentangled_transformer import XLMRobertaForDisentanglement
from experiment_datasets import oscar_corpus
import transformers
import json
import argparse
import torch
import time
import os

start_time = time.time()
parser = argparse.ArgumentParser(description="mlm-disentanglement experinment")
parser.add_argument(
    "--config_json",
    metavar="path",
    type=str,
    help="path to configuration json file",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "default.json"),
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
    experiment_config_dict = json.load(outfile, cls=ExperimentConfigSerializer)
experiment_config_dict["training"].model_name = (
    os.path.abspath(args.config_json).split("/")[-1].split(".")[0]
)

backbone_name = experiment_config_dict["training"].backbone_name
XLMRConfig = transformers.AutoConfig.from_pretrained(backbone_name)

setattr(XLMRConfig, "discriminators", experiment_config_dict["discriminators"])

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
    : experiment_config_dict["training"].num_frozen_layers
]:  # 0:24
    for param in layers.parameters():
        param.requires_grad = False
optimizermlm = torch.optim.Adam(
    multitask_model.parameters(),
    lr=experiment_config_dict["training"].mlm_lr,
    betas=(
        experiment_config_dict["training"].mlm_beta1,
        experiment_config_dict["training"].mlm_beta2,
    ),
    eps=experiment_config_dict["training"].mlm_eps,
)
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(
    "/gpfs1/home/ckchan666/mlm_disentangle_experiment/tensorboard/"
    + os.path.dirname(os.path.abspath(__file__)).split("/")[-1]
    + "/"
    + experiment_config_dict["training"].model_name
)

MLMD_ds = oscar_corpus.get_custom_corpus()
MLMD_ds.set_format(type="torch")
dataloader = torch.utils.data.DataLoader(
    MLMD_ds,
    batch_size=experiment_config_dict["training"].batch_size,
    num_workers=0,
    shuffle=True,
)

import gc

for task in multitask_model.taskmodels_dict:
    multitask_model.taskmodels_dict[task].cpu()
mlmLoss = 0.0
disentangleLoss = 0.0
gradient_step = 0
model_path = (
    "/gpfs1/home/ckchan666/mlm_disentangle_experiment/model/"
    + os.path.dirname(os.path.abspath(__file__)).split("/")[-1]
    + "/"
    + experiment_config_dict["training"].model_name
)
print("building time: " + str(time.time() - start_time) + "s")
start_time = time.time()
print(
    "run for "
    + str(experiment_config_dict["training"].max_step)
    + " step with "
    + str(experiment_config_dict["training"].gradient_acc_size)
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
        experiment_config_dict["training"].gradient_acc_size
        / experiment_config_dict["training"].batch_size
    ) == 0:
        torch.nn.utils.clip_grad_norm_(multitask_model.parameters(), 1.0)
        optimizermlm.step()
        multitask_model.zero_grad()
        gradient_step += 1
        if gradient_step == 1:
            print(
                "mlm loss ("
                + str(gradient_step)
                + "): "
                + str(mlmLoss / experiment_config_dict["training"].gradient_acc_size)
            )
            print(
                "disentangle loss ("
                + str(gradient_step)
                + "): "
                + str(
                    disentangleLoss
                    / experiment_config_dict["training"].gradient_acc_size
                )
            )
            writer.add_scalar(
                "mlm loss",
                mlmLoss / experiment_config_dict["training"].gradient_acc_size,
                gradient_step,
            )
            writer.add_scalar(
                "disentangle loss",
                disentangleLoss / experiment_config_dict["training"].gradient_acc_size,
                gradient_step,
            )
        if gradient_step % experiment_config_dict["training"].log_step == 0:
            # writer.add_scalar("mlm lr", scheduler.get_lr()[0], global_step)
            # writer.add_scalar("disentangle lr", scheduler.get_lr()[0], global_step)
            print(
                "mlm loss ("
                + str(gradient_step)
                + "): "
                + str(
                    mlmLoss
                    / (
                        experiment_config_dict["training"].log_step
                        * experiment_config_dict["training"].gradient_acc_size
                    )
                )
            )
            print(
                "disentangle loss ("
                + str(gradient_step)
                + "): "
                + str(
                    disentangleLoss
                    / (
                        experiment_config_dict["training"].log_step
                        * experiment_config_dict["training"].gradient_acc_size
                    )
                )
            )
            writer.add_scalar(
                "mlm loss",
                mlmLoss
                / (
                    experiment_config_dict["training"].log_step
                    * experiment_config_dict["training"].gradient_acc_size
                ),
                gradient_step,
            )
            writer.add_scalar(
                "disentangle loss",
                disentangleLoss
                / (
                    experiment_config_dict["training"].log_step
                    * experiment_config_dict["training"].gradient_acc_size
                ),
                gradient_step,
            )
            mlmLoss = 0
            disentangleLoss = 0
            multitask_model.save_pretrained(model_path)
        if gradient_step >= experiment_config_dict["training"].max_step:
            break
        if time.time() - start_time > 0.9 * args.time:
            print(str(time.time() - start_time) + "s exceed 0.9 of the time limit")
            print(str(gradient_step) + "th gradient step")
            break
    gc.collect()


multitask_model.save_pretrained(model_path)
print(str(time.time() - start_time) + " seconds elapsed")
