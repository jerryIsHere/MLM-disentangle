from experiment_util.experiment_config import ExperimentConfigSerializer
from experiment_models.multitask_transformer import MultitaskModel
from experiment_models.disentangled_transformer import XLMRobertaForDisentanglement
from experiment_datasets import oscar_corpus
from experiment_datasets import xtreme_ds
import transformers
import json
import argparse
import torch
import time
import os

task = "udpos"
start_time = time.time()
parser = argparse.ArgumentParser(description="udpos disentangle experinment")
parser.add_argument(
    "--config_json",
    metavar="path",
    type=str,
    help="path to configuration json file of the pretrained disentangled model",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "/default.json"),
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

finetune_model = MultitaskModel.create(
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
import torch

finetune_model.load_state_dict(
    torch.load(
        "/gpfs1/home/ckchan666/mlm_disentangle_experiment/model/mlm/"
        + experiment_config_dict["training"].model_name
        + "/pytorch_model.bin"
    )
)
finetune_model.taskmodels_dict.pop("mlm")
finetune_model.add_task(
    task,
    transformers.XLMRobertaForTokenClassification,
    transformers.XLMRobertaConfig.from_pretrained(finetune_model.backbone_name),
)


optimizerudpos = torch.optim.Adam(
    finetune_model.taskmodels_dict[task].parameters(),
    lr=xtreme_ds.TASK[task]["learning rate"],
    eps=xtreme_ds.TASK[task]["weight decay"],
)

optimizerdisentangle = torch.optim.Adam(
    finetune_model.taskmodels_dict["disentangle"].parameters(),
    lr=experiment_config_dict["training"].disentangle_lr,
    betas=(
        experiment_config_dict["training"].disentangle_beta1,
        experiment_config_dict["training"].disentangle_beta2,
    ),
    eps=experiment_config_dict["training"].disentangle_eps,
)
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(
    "/gpfs1/home/ckchan666/mlm_disentangle_experiment/tensorboard/"
    + os.path.dirname(os.path.abspath(__file__)).split("/")[-1]
    + "/"
    + experiment_config_dict["training"].model_name
)

MLMD_ds = oscar_corpus.get_custom_corpus().set_format(type="torch")
disentangle_dataloader = torch.utils.data.DataLoader(
    MLMD_ds,
    batch_size=experiment_config_dict["training"].batch_size,
    num_workers=0,
    shuffle=True,
)
udpos_ds = xtreme_ds.udposTrainDataset()
udpos_ds_dataloader = torch.utils.data.DataLoader(
    udpos_ds, batch_size=2, num_workers=0, shuffle=True
)
gradient_acc_size = 16
batch_size = 2
scheduler = transformers.get_linear_schedule_with_warmup(
    optimizerudpos,
    num_warmup_steps=xtreme_ds.TASK[task]["warmup step"],
    num_training_steps=len(udpos_ds)
    // gradient_acc_size
    * xtreme_ds.TASK[task]["epochs"],
)

import gc

for task in finetune_model.taskmodels_dict:
    finetune_model.taskmodels_dict[task].cpu()
udposLoss = 0.0
disentangleLoss = 0.0
gradient_step = 0
print("building time: " + str(time.time() - start_time) + "s")
start_time = time.time()
print(
    "run for "
    + xtreme_ds.TASK[task]["epochs"]
    + " epoches with "
    + str(gradient_acc_size)
    + " gradient acc size"
)
i = 0
for _ in range(xtreme_ds.TASK[task]["epochs"]):
    for batch in udpos_ds_dataloader:
        #  input to gpu
        batch["tokens"] = batch["tokens"].cuda()
        batch["pos_tags"] = batch["pos_tags"].cuda()

        #  model to gpu
        finetune_model.taskmodels_dict[task].cuda()
        Output = finetune_model.taskmodels_dict[task](
            input_ids=batch["tokens"],
            labels=batch["pos_tags"],
        )
        Output["loss"].backward()

        #  model & input to cpu
        udposLoss = udposLoss + Output["loss"].item()
        finetune_model.taskmodels_dict[task].cpu()
        del Output
        batch.clear()
        del batch

        batch = next(disentangle_dataloader)
        # disentangle input to gpu
        batch["masked_tokens"] = batch["masked_tokens"].cuda()
        batch["language_id"] = batch["language_id"].cuda()
        batch["genus_label"] = batch["genus_label"].cuda()
        batch["family_label"] = batch["family_label"].cuda()

        # disentangle model to gpu
        finetune_model.taskmodels_dict["disentangle"].cuda()
        disentangleOutput = finetune_model.taskmodels_dict["disentangle"](
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
        finetune_model.taskmodels_dict["disentangle"].cpu()
        for output in disentangleOutput:
            if disentangleOutput[output] == None:
                continue
            if output == "logits":
                disentangleOutput[output].clear()
        del disentangleOutput
        batch.clear()
        del batch

        if (i + 1) % (gradient_acc_size / batch_size) == 0:
            optimizerudpos.step()
            finetune_model.taskmodels_dict[task].zero_grad()
            optimizerdisentangle.step()
            finetune_model.taskmodels_dict["disentangle"].zero_grad()
            gradient_step += 1
            if gradient_step == 1:
                print(
                    "loss ("
                    + str(gradient_step)
                    + "): "
                    + str(udposLoss / experiment_config_dict["training"].log_step)
                )
                print(
                    "disentangle loss ("
                    + str(gradient_step)
                    + "): "
                    + str(disentangleLoss / experiment_config_dict["training"].log_step)
                )
            if gradient_step % experiment_config_dict["training"].log_step == 0:
                writer.add_scalar("lr", scheduler.get_lr()[0], i)
                writer.add_scalar("disentangle lr", scheduler.get_lr()[0], i)
                print(
                    " loss ("
                    + str(gradient_step)
                    + "): "
                    + str(udposLoss / experiment_config_dict["training"].log_step)
                )
                print(
                    "disentangle loss ("
                    + str(gradient_step)
                    + "): "
                    + str(disentangleLoss / experiment_config_dict["training"].log_step)
                )
                writer.add_scalar(
                    " loss",
                    udposLoss / experiment_config_dict["training"].log_step,
                    gradient_step,
                )
                writer.add_scalar(
                    "disentangle loss",
                    disentangleLoss / experiment_config_dict["training"].log_step,
                    gradient_step,
                )
                udposLoss = 0
                disentangleLoss = 0
                finetune_model.save_pretrained(
                    "./" + experiment_config_dict["training"].model_name,
                )
        gc.collect()
        i += 1


finetune_model.save_pretrained(
    "/gpfs1/home/ckchan666/mlm_disentangle_experiment/model/"
    + os.path.dirname(os.path.abspath(__file__)).split("/")[-1]
    + "/"
    + experiment_config_dict["training"].model_name,
)
print(str(time.time() - start_time) + " seconds elapsed for training")


test_dataloader = torch.utils.data.DataLoader(
    xtreme_ds.udposTestDataset(), batch_size=2, num_workers=0, shuffle=True
)


for batch in test_dataloader:
    #  input to gpu
    batch["tokens"] = batch["tokens"].cuda()
    batch["pos_tags"] = batch["pos_tags"].cuda()

    #  model to gpu
    finetune_model.taskmodels_dict[task].cuda()
    Output = finetune_model.taskmodels_dict[task](
        input_ids=batch["tokens"],
        labels=batch["pos_tags"],
    )
=
    del Output
    batch.clear()
    del batch


from resource import getrusage, RUSAGE_SELF

print(str(getrusage(RUSAGE_SELF).ru_maxrss) + "KB used (peak)")
