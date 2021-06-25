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
task_config = xtreme_ds.TASK[task]


def build_model(experiment_config_dict, mlm_model_path):
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

    finetune_model.load_state_dict(torch.load(mlm_model_path))
    finetune_model.taskmodels_dict.pop("mlm")
    finetune_model.add_task(
        task,
        transformers.XLMRobertaForTokenClassification,
        transformers.XLMRobertaConfig.from_pretrained(
            finetune_model.backbone_name, num_labels=task_config["num_labels"]
        ),
    )
    return finetune_model


def train(
    finetune_model,
    writer,
    model_path,
    MLMD_ds,
    custom_stop_condition=lambda gradient_step: False,
):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in finetune_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": task_config["weight_decay"],
        },
        {
            "params": [
                p
                for n, p in finetune_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = transformers.AdamW(
        optimizer_grouped_parameters,
        lr=task_config["learning_rate"],
        eps=task_config["adam_epsilon"],
    )

    disentangle_dataloader = torch.utils.data.DataLoader(
        MLMD_ds,
        batch_size=task_config["batch_size"],
        num_workers=0,
        shuffle=True,
    )
    disentangle_iter = iter(disentangle_dataloader)
    udpos_ds = xtreme_ds.udposTrainDataset()
    udpos_ds_dataloader = torch.utils.data.DataLoader(
        udpos_ds, batch_size=2, num_workers=0, shuffle=True
    )
    gradient_acc_size = task_config["gradient_acc_size"]
    batch_size = task_config["batch_size"]
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=task_config["warmup_steps"],
        num_training_steps=len(udpos_ds) // gradient_acc_size * task_config["epochs"],
    )
    log_step_size = len(udpos_ds) // gradient_acc_size * task_config["epochs"] // 20
    import gc

    for task in finetune_model.taskmodels_dict:
        finetune_model.taskmodels_dict[task].cpu()
    udposLoss = 0.0
    disentangleLoss = 0.0
    gradient_step_counter = 0
    print(
        "run for "
        + str(task_config["epochs"])
        + " epoches with "
        + str(gradient_acc_size)
        + " gradient acc size"
    )
    i = 0
    for _ in range(task_config["epochs"]):
        for batch in udpos_ds_dataloader:
            #  input to gpu
            batch["tokens"] = batch["tokens"].cuda()
            batch["tags"] = batch["tags"].cuda()

            #  model to gpu
            finetune_model.taskmodels_dict[task].cuda()
            Output = finetune_model.taskmodels_dict[task](
                input_ids=batch["tokens"],
                labels=batch["tags"],
            )
            Output["loss"].backward()

            #  model & input to cpu
            udposLoss = udposLoss + Output["loss"].item()
            finetune_model.taskmodels_dict[task].cpu()
            del Output
            batch.clear()
            del batch

            batch = next(disentangle_iter)
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
                torch.nn.utils.clip_grad_norm_(finetune_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                finetune_model.zero_grad()
                gradient_step_counter += 1
                if gradient_step_counter % log_step_size == 0:
                    writer.add_scalar("lr", scheduler.get_lr()[0], i)
                    writer.add_scalar("disentangle lr", scheduler.get_lr()[0], i)
                    print(
                        " loss ("
                        + str(gradient_step_counter)
                        + "): "
                        + str(udposLoss / (log_step_size * gradient_acc_size))
                    )
                    print(
                        "disentangle loss ("
                        + str(gradient_step_counter)
                        + "): "
                        + str(disentangleLoss / (log_step_size * gradient_acc_size))
                    )
                    writer.add_scalar(
                        " loss",
                        udposLoss / (log_step_size * gradient_acc_size),
                        gradient_step_counter,
                    )
                    writer.add_scalar(
                        "disentangle loss",
                        disentangleLoss / (log_step_size * gradient_acc_size),
                        gradient_step_counter,
                    )
                    udposLoss = 0
                    disentangleLoss = 0
                    finetune_model.save_pretrained(model_path)
                if custom_stop_condition(gradient_step_counter):
                    break
            gc.collect()
            i += 1

    finetune_model.save_pretrained(model_path)


# testing
def test(finetune_model):
    test_dataloader = torch.utils.data.DataLoader(
        xtreme_ds.udposTestDataset(), batch_size=1, num_workers=0, shuffle=True
    )
    metric = xtreme_ds.METRIC_FUNCTION[task]()
    lan_metric = {}
    for lan in xtreme_ds.TASK2LANGS[task]:
        lan_metric[lan] = xtreme_ds.METRIC_FUNCTION[task]()
    finetune_model.taskmodels_dict[task].cuda()
    for batch in test_dataloader:
        with torch.no_grad():
            #  input to gpu
            batch["tokens"] = batch["tokens"].cuda()
            Output = finetune_model.taskmodels_dict[task](input_ids=batch["tokens"])
            predictions = torch.argmax(Output["logits"], dim=2)
            for i, lan in enumerate(batch["lan"]):
                for j, token_pred in enumerate(predictions[i]):
                    if batch["tags"][i][j] == -100:
                        continue
                    lan_metric[lan].add(
                        prediction=token_pred, reference=batch["tags"][i][j]
                    )
                    metric.add(prediction=token_pred, reference=batch["tags"][i][j])
            del Output
            batch.clear()
            del batch

    for lan in lan_metric:
        print(lan)
        print(lan_metric[lan].compute(average="micro"))
    print("overall f1 score:")
    print(metric.compute(average="micro"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="udpos disentangle experinment")
    parser.add_argument(
        "--config_json",
        metavar="path",
        type=str,
        help="path to configuration json file of the pretrained disentangled model",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "/default.json"
        ),
    )

    args = parser.parse_args()
    with open(args.config_json, "r") as outfile:
        experiment_config_dict = json.load(outfile, cls=ExperimentConfigSerializer)
    experiment_config_dict["training"].model_name = (
        os.path.abspath(args.config_json).split("/")[-1].split(".")[0]
    )
    model = build_model(
        experiment_config_dict=experiment_config_dict,
        mlm_model_path="/gpfs1/home/ckchan666/mlm_disentangle_experiment/model/mlm/"
        + experiment_config_dict["training"].model_name
        + "/pytorch_model.bin",
    )
    start_time = time.time()
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(
        "/gpfs1/home/ckchan666/mlm_disentangle_experiment/tensorboard/"
        + os.path.dirname(os.path.abspath(__file__)).split("/")[-1]
        + "/"
        + experiment_config_dict["training"].model_name
    )
    model_path = (
        "/gpfs1/home/ckchan666/mlm_disentangle_experiment/model/"
        + os.path.dirname(os.path.abspath(__file__)).split("/")[-1]
        + "/"
        + experiment_config_dict["training"].model_name
    )
    MLMD_ds = oscar_corpus.get_custom_corpus()
    MLMD_ds.set_format(type="torch")
    train(finetune_model=model, writer=writer, model_path=model_path, MLMD_ds=MLMD_ds)
    print(str(time.time() - start_time) + " seconds elapsed for training")
    test(model)
