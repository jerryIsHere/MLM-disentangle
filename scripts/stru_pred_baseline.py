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


def tag_train(
    finetune_model,
    writer,
    model_path,
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
        * xtreme_ds.TASK[task]["epochs"],
    )
    log_step_size = (
        len(tag_ds) // gradient_acc_size * xtreme_ds.TASK[task]["epochs"] // 20
    )
    import gc

    finetune_model.cuda()
    udposLoss = 0.0
    gradient_step_counter = 0
    print(
        "run for "
        + str(xtreme_ds.TASK[task]["epochs"])
        + " epoches with "
        + str(gradient_acc_size)
        + " gradient acc size"
    )
    i = 0
    for _ in range(xtreme_ds.TASK[task]["epochs"]):
        for batch in tag_ds_dataloader:
            #  input to gpu
            batch["tokens"] = batch["tokens"].cuda()
            batch["tags"] = batch["tags"].cuda()

            #  model to gpu
            finetune_model.cuda()
            Output = finetune_model(
                input_ids=batch["tokens"],
                labels=batch["tags"],
            )
            Output["loss"].backward()

            #  acc loss
            udposLoss = udposLoss + Output["loss"].item()

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
                    writer.add_scalar(
                        writer.filename_suffix + " loss",
                        udposLoss / (log_step_size * gradient_acc_size),
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
    finetune_model.cuda()
    for batch in test_dataloader:
        with torch.no_grad():
            #  input to gpu
            batch["tokens"] = batch["tokens"].cuda()
            Output = finetune_model(input_ids=batch["tokens"])
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
    with open(
        "/gpfs1/home/ckchan666/MLM-disentangle/experiment/mlm/default.json", "r"
    ) as outfile:
        experiment_config_dict = json.load(outfile, cls=ExperimentConfigSerializer)
    experiment_config_dict["training"].model_name = (
        os.path.abspath(args.config_json).split("/")[-1].split(".")[0]
    )
    print("udpos")
    print("model configuration name: " + experiment_config_dict["training"].model_name)

    
    start_time = time.time()
    from torch.utils.tensorboard import SummaryWriter

    ds = xtreme_ds.udposTrainDataset()
    model = transformers.XLMRobertaForTokenClassification.from_pretrained(
        experiment_config_dict["training"].backbone_name, num_labels=xtreme_ds.TASK[ds.task]["num_labels"]
    )
    writer = SummaryWriter(
        "/gpfs1/home/ckchan666/job/stru_pred_baseline/udpos",
        filename_suffix="."
        + ds.task
        + "."
        + experiment_config_dict["training"].model_name,
    )
    model_path = (
        "/gpfs1/home/ckchan666/job/stru_pred_baseline/udpos"
        + "."
        + experiment_config_dict["training"].model_name
    )
    tag_train(
        finetune_model=model,
        writer=writer,
        model_path=model_path,
        tag_ds=ds,
    )
    print(str(time.time() - start_time) + " seconds elapsed for training")
    ds = xtreme_ds.udposTestDataset()
    tag_test(
        model,
        tag_ds=xtreme_ds.udposTestDataset(),
    )
    print("panx")
    print("model configuration name: " + experiment_config_dict["training"].model_name)

    start_time = time.time()
    from torch.utils.tensorboard import SummaryWriter

    ds = xtreme_ds.panxTrainDataset()
    model = transformers.XLMRobertaForTokenClassification.from_pretrained(
        experiment_config_dict["training"].backbone_name, num_labels=xtreme_ds.TASK[ds.task]["num_labels"]
    )
    writer = SummaryWriter(
        "/gpfs1/home/ckchan666/job/stru_pred_baseline/panx",
        filename_suffix="."
        + ds.task
        + "."
        + experiment_config_dict["training"].model_name,
    )
    model_path = (
        "/gpfs1/home/ckchan666/job/stru_pred_baseline/panx"
        + "."
        + experiment_config_dict["training"].model_name
    )
    tag_train(
        finetune_model=model,
        writer=writer,
        model_path=model_path,
        tag_ds=ds,
    )
    print(str(time.time() - start_time) + " seconds elapsed for training")
    ds = xtreme_ds.panxTestDataset()
    tag_test(
        model,
        tag_ds=ds,
    )
