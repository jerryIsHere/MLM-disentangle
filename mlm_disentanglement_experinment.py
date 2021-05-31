from disentangled_transformer import (
    MultitaskModel,
    XLMRobertaForDisentanglement,
    DiscriminatorType,
    DiscriminatorConfig,
    DiscriminatorConfigSerializer,
)
import oscar_corpus
import transformers
import json
import argparse

parser = argparse.ArgumentParser(description="token frequency of 40 corpus")
parser.add_argument(
    "--config_json",
    metavar="path",
    type=str,
    help="path to configuration json file",
    default="experinment_configuaration.json",
)
parser.add_argument(
    "--frozen_layers",
    metavar="layers",
    type=int,
    help="number of layer frozen in the disentanglement training",
    default=20,
)
args = parser.parse_args()

backbone_name = "xlm-roberta-large"
XLMRobertaConfig = transformers.AutoConfig.from_pretrained(backbone_name)

with open(args.config_json, "r") as outfile:
    discriminator_config = json.load(outfile, cls=DiscriminatorConfigSerializer)
setattr(XLMRobertaConfig, "discriminators", discriminator_config)

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

for layers in multitask_model.backbone.encoder.layer[: args.frozen_layers]:  # 0:24
    for param in layers.parameters():
        param.requires_grad = False
