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


def tag_build_model(experiment_config_dict, mlm_model_path, task):
    backbone_name = experiment_config_dict["training"].backbone_name
    XLMRConfig = transformers.AutoConfig.from_pretrained(backbone_name)

    setattr(XLMRConfig, "discriminators", experiment_config_dict["discriminators"])

    finetune_model = MultitaskModel.create_untrained(
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
            finetune_model.backbone_name, num_labels=xtreme_ds.TASK[task]["num_labels"]
        ),
    )
    return finetune_model


def tag_load_finetuned_model(experiment_config_dict, mlm_model_path, task):
    backbone_name = experiment_config_dict["training"].backbone_name
    XLMRConfig = transformers.AutoConfig.from_pretrained(backbone_name)

    setattr(XLMRConfig, "discriminators", experiment_config_dict["discriminators"])

    finetuned_model = MultitaskModel.create_untrained(
        backbone_name=backbone_name,
        task_dict={
            task: {
                "type": transformers.XLMRobertaForTokenClassification,
                "config": transformers.XLMRobertaConfig.from_pretrained(
                    backbone_name,
                    num_labels=xtreme_ds.TASK[task]["num_labels"],
                ),
            },
            "disentangle": {
                "type": XLMRobertaForDisentanglement,
                "config": XLMRConfig,
            },
        },
    )
    import torch

    finetuned_model.load_state_dict(torch.load(mlm_model_path))
    return finetuned_model


def tag_train(
    finetune_model,
    writer,
    model_path,
    MLMD_ds,
    tag_ds,
    custom_stop_condition=lambda gradient_step: False,
):
    task = tag_ds.task
    print("training " + task + " with dataset:" + tag_ds.__class__.__name__)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in finetune_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": xtreme_ds.TASK[task]["weight_decay"],
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
        lr=xtreme_ds.TASK[task]["learning_rate"],
        eps=xtreme_ds.TASK[task]["adam_epsilon"],
    )

    disentangle_dataloader = torch.utils.data.DataLoader(
        MLMD_ds,
        batch_size=xtreme_ds.TASK[task]["batch_size"],
        num_workers=2,
        shuffle=True,
    )
    disentangle_iter = iter(xtreme_ds.loop_iter(disentangle_dataloader))
    tag_ds_dataloader = torch.utils.data.DataLoader(
        tag_ds,
        batch_size=xtreme_ds.TASK[task]["batch_size"],
        num_workers=2,
        shuffle=True,
    )
    gradient_acc_size = xtreme_ds.TASK[task]["gradient_acc_size"]
    batch_size = xtreme_ds.TASK[task]["batch_size"]
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=xtreme_ds.TASK[task]["warmup_steps"],
        num_training_steps=len(tag_ds)
        // gradient_acc_size
        * (xtreme_ds.TASK[task]["epochs"] * 2),
    )
    log_step_size = (
        len(tag_ds) // gradient_acc_size * (xtreme_ds.TASK[task]["epochs"] * 2) // 20
    )
    import gc

    finetune_model.cuda()
    udposLoss = 0.0
    disentangleLoss = 0.0
    gradient_step_counter = 0
    print(
        "run for "
        + str((xtreme_ds.TASK[task]["epochs"] * 2))
        + " epoches with "
        + str(gradient_acc_size)
        + " gradient acc size"
    )
    i = 0
    for _ in range((xtreme_ds.TASK[task]["epochs"] * 2)):
        for batch in tag_ds_dataloader:
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

            #  acc loss
            udposLoss = udposLoss + Output["loss"].item()

            batch = next(disentangle_iter)
            # disentangle input to gpu
            batch["masked_tokens"] = batch["masked_tokens"].cuda()
            batch["language_id"] = batch["language_id"].cuda()
            batch["genus_label"] = batch["genus_label"].cuda()
            batch["family_label"] = batch["family_label"].cuda()

            # disentangle model to gpu
            finetune_model.taskmodels_dict["disentangle"].cuda()
            disentangleOutput = finetune_model.taskmodels_dict["disentangle"](
                input_ids=batch["masked_tokens"][
                    0 : xtreme_ds.TASK[task]["max seq length"]
                ],
                labels={
                    "language_id": batch["language_id"],
                    "genus_label": batch["genus_label"],
                    "family_label": batch["family_label"],
                },
            )
            disentangleOutput["loss"].backward()

            #  acc loss
            disentangleLoss = disentangleLoss + disentangleOutput["loss"].item()

            if (i + 1) % (gradient_acc_size / batch_size) == 0:
                torch.nn.utils.clip_grad_norm_(finetune_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                finetune_model.zero_grad()
                gradient_step_counter += 1
                if gradient_step_counter % log_step_size == 0:
                    writer.add_scalar(
                        writer.filename_suffix + "lr", scheduler.get_lr()[0], i
                    )
                    writer.add_scalar(
                        writer.filename_suffix + "disentangle lr",
                        scheduler.get_lr()[0],
                        i,
                    )
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
                        writer.filename_suffix + " loss",
                        udposLoss / (log_step_size * gradient_acc_size),
                        gradient_step_counter,
                    )
                    writer.add_scalar(
                        writer.filename_suffix + "disentangle loss",
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
def tag_test(finetune_model, tag_ds):
    task = tag_ds.task
    print("evaluating " + task + " with dataset:" + tag_ds.__class__.__name__)
    test_dataloader = torch.utils.data.DataLoader(tag_ds, batch_size=1)
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
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "mlm",
            "default.json",
        ),
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to run training."
    )
    args = parser.parse_args()
    with open(args.config_json, "r") as outfile:
        experiment_config_dict = json.load(outfile, cls=ExperimentConfigSerializer)
    experiment_config_dict["training"].model_name = (
        os.path.abspath(args.config_json).split("/")[-1].split(".")[0]
    )
    print("model configuration name: " + experiment_config_dict["training"].model_name)
    if args.do_train:
        model = tag_build_model(
            experiment_config_dict=experiment_config_dict,
            mlm_model_path="/gpfs1/home/ckchan666/mlm_disentangle_experiment/model/mlm/"
            + experiment_config_dict["training"].model_name
            + "/pytorch_model.bin",
            task="udpos",
        )
        start_time = time.time()
        from torch.utils.tensorboard import SummaryWriter

        ds = xtreme_ds.udposTrainDataset()
        writer = SummaryWriter(
            "/gpfs1/home/ckchan666/mlm_disentangle_experiment/tensorboard/"
            + os.path.dirname(os.path.abspath(__file__)).split("/")[-1]
            + "/"
            + experiment_config_dict["training"].model_name,
            filename_suffix="."
            + ds.task
            + "."
            + experiment_config_dict["training"].model_name,
        )
        model_path = (
            "/gpfs1/home/ckchan666/mlm_disentangle_experiment/model/"
            + os.path.dirname(os.path.abspath(__file__)).split("/")[-1]
            + "/"
            + experiment_config_dict["training"].model_name
        )
        MLMD_ds = oscar_corpus.get_custom_corpus()
        MLMD_ds.set_format(type="torch")
        tag_train(
            finetune_model=model,
            writer=writer,
            model_path=model_path,
            MLMD_ds=MLMD_ds,
            tag_ds=ds,
        )
        print(str(time.time() - start_time) + " seconds elapsed for training")
    if args.do_test:
        ds = xtreme_ds.udposTestDataset()
        model = tag_load_finetuned_model(
            experiment_config_dict=experiment_config_dict,
            mlm_model_path="/gpfs1/home/ckchan666/mlm_disentangle_experiment/model/"
            + ds.task
            + "/"
            + experiment_config_dict["training"].model_name
            + "/pytorch_model.bin",
            task=ds.task,
        )
        tag_test(
            model,
            tag_ds=ds,
        )
