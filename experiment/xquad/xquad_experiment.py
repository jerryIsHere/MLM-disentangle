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
import re

task = "xquad"


def normalize_string(txt):
    tokenized_detokenized = " ".join(
        xtreme_ds.tokenizer.convert_ids_to_tokens(
            xtreme_ds.tokenizer(txt).input_ids[1:-1]
        )
    )
    return tokenized_detokenized[1:]


def normalize_ids(ids):
    tokenized_detokenized = " ".join(xtreme_ds.tokenizer.convert_ids_to_tokens(ids))
    tokenized_detokenized = tokenized_detokenized[1:]
    if re.search("^\([\d\s]+\)$", tokenized_detokenized):  # is number with brancket
        tokenized_detokenized = tokenized_detokenized[1:-1]
        tokenized_detokenized = "".join(tokenized_detokenized.split())
    if re.search("^[\d\s]+$", tokenized_detokenized):  # is number with brancket
        tokenized_detokenized = "".join(tokenized_detokenized.split())
    return tokenized_detokenized[1:]


def qa_build_model(experiment_config_dict, mlm_model_path, task):
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
        transformers.XLMRobertaForQuestionAnswering,
        transformers.XLMRobertaConfig.from_pretrained(finetune_model.backbone_name),
    )
    return finetune_model


def qa_train(
    finetune_model,
    writer,
    model_path,
    MLMD_ds,
    qa_ds,
    task,
    custom_stop_condition=lambda gradient_step: False,
):
    print("training " + task + " with dataset:" + qa_ds.__class__.__name__)
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
        num_workers=0,
        shuffle=True,
    )
    disentangle_iter = iter(disentangle_dataloader)
    qa_ds_dataloader = torch.utils.data.DataLoader(
        qa_ds, batch_size=2, num_workers=0, shuffle=True
    )
    gradient_acc_size = xtreme_ds.TASK[task]["gradient_acc_size"]
    batch_size = xtreme_ds.TASK[task]["batch_size"]
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=xtreme_ds.TASK[task]["warmup_steps"],
        num_training_steps=len(qa_ds)
        // gradient_acc_size
        * xtreme_ds.TASK[task]["epochs"],
    )
    log_step_size = (
        len(qa_ds) // gradient_acc_size * xtreme_ds.TASK[task]["epochs"] // 20
    )
    import gc

    for task in finetune_model.taskmodels_dict:
        finetune_model.taskmodels_dict[task].cpu()
    xquadLoss = 0.0
    disentangleLoss = 0.0
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
        for batch in qa_ds_dataloader:
            #  input to gpu
            batch["tokens"] = batch["tokens"].cuda()
            batch["start_positions"] = batch["start_positions"].cuda()
            batch["end_positions"] = batch["end_positions"].cuda()
            #  model to gpu
            finetune_model.taskmodels_dict[task].cuda()
            Output = finetune_model.taskmodels_dict[task](
                input_ids=batch["tokens"],
                start_positions=batch["start_positions"][:, 0],
                end_positions=batch["end_positions"][:, 0],
            )
            Output["loss"].backward()

            #  model & input to cpu
            xquadLoss = xquadLoss + Output["loss"].item()
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
                        + str(xquadLoss / (log_step_size * gradient_acc_size))
                    )
                    print(
                        "disentangle loss ("
                        + str(gradient_step_counter)
                        + "): "
                        + str(disentangleLoss / (log_step_size * gradient_acc_size))
                    )
                    writer.add_scalar(
                        " loss",
                        xquadLoss / (log_step_size * gradient_acc_size),
                        gradient_step_counter,
                    )
                    writer.add_scalar(
                        "disentangle loss",
                        disentangleLoss / (log_step_size * gradient_acc_size),
                        gradient_step_counter,
                    )
                    xquadLoss = 0
                    disentangleLoss = 0
                    finetune_model.save_pretrained(model_path)
                if custom_stop_condition(gradient_step_counter):
                    break
            gc.collect()
            i += 1

    finetune_model.save_pretrained(model_path)


# testing
def qa_test(finetune_model, qa_ds, task):
    print("evaluating " + task + " with dataset:" + qa_ds.__class__.__name__)
    test_dataloader = torch.utils.data.DataLoader(
        qa_ds, batch_size=1, num_workers=0, shuffle=True
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
            Output = finetune_model.taskmodels_dict[task](batch["tokens"])
            start_predictions = torch.argmax(Output["start_logits"], dim=1)
            end_predictions = torch.argmax(Output["end_logits"], dim=1)
            for i, lan in enumerate(batch["lan"]):
                predictions = xtreme_ds.tokenizer.convert_tokens_to_string(
                    xtreme_ds.tokenizer.convert_ids_to_tokens(
                        batch["tokens"][i][start_predictions[i] : end_predictions[i]]
                    )
                )
                lan_metric[lan].add(
                    prediction={"id": batch["id"][i], "prediction_text": predictions},
                    reference={
                        "id": batch["id"][i],
                        "answers": {
                            "text": xtreme_ds.normalize_string(
                                batch["answers"]["text"][i]
                            ),
                            "answer_start": batch["answers"]["answer_start"][i],
                        },
                    },
                )
                metric.add(
                    prediction={"id": batch["id"][i], "prediction_text": predictions},
                    reference={
                        "id": batch["id"][i],
                        "answers": {
                            "text": xtreme_ds.normalize_string(
                                batch["answers"]["text"][i]
                            ),
                            "answer_start": batch["answers"]["answer_start"][i],
                        },
                    },
                )
            del Output
            batch.clear()
            del batch

    for lan in lan_metric:
        print(lan)
        print(lan_metric[lan].compute(average="micro"))
    print("overall f1 score:")
    print(metric.compute(average="micro"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="xquad disentangle experinment")
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
    model = qa_build_model(
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
    qa_train(
        finetune_model=model,
        writer=writer,
        model_path=model_path,
        MLMD_ds=MLMD_ds,
        qa_ds=xtreme_ds.xquadTrainDataset(),
    )
    print(str(time.time() - start_time) + " seconds elapsed for training")
    qa_test(model, qa_ds=xtreme_ds.xquadTestDataset())
    qa_test(model, qa_ds=xtreme_ds.mlqaTestDataset())
