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

if __name__ == "__main__":
    from experiment.xnli.xnli_experiment import (
        cls_build_model,
        cls_train,
        cls_test,
        cls_load_finetuned_model,
    )

    parser = argparse.ArgumentParser(description="pawsx disentangle experinment")
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
        model = cls_build_model(
            experiment_config_dict=experiment_config_dict,
            mlm_model_path="/gpfs1/home/ckchan666/mlm_disentangle_experiment/model/mlm/"
            + experiment_config_dict["training"].model_name
            + "/pytorch_model.bin",
            task="pawsx",
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
        cls_train(
            finetune_model=model,
            writer=writer,
            model_path=model_path,
            MLMD_ds=MLMD_ds,
            cls_ds=xtreme_ds.pawsxTrainDataset(),
        )
        print(str(time.time() - start_time) + " seconds elapsed for training")
    if args.do_test:
        ds = xtreme_ds.pawsxTestDataset()
        model = cls_load_finetuned_model(
            experiment_config_dict=experiment_config_dict,
            mlm_model_path="/gpfs1/home/ckchan666/mlm_disentangle_experiment/model/"
            + ds.task
            + "/"
            + experiment_config_dict["training"].model_name
            + "/pytorch_model.bin",
            task=ds.task,
        )
        cls_test(
            model,
            cls_ds=ds,
        )
