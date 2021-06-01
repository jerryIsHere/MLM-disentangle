import numpy as np
import torch
import torch.nn as nn
import transformers

# https://colab.research.google.com/github/zphang/zphang.github.io/blob/master/files/notebooks/Multi_task_Training_with_Transformers_NLP.ipynb#scrollTo=7zSZsp8Cb7gd


class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, backbone, taskmodels_dict):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())

        self.backbone = backbone
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, backbone_name, task_dict):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models.

        We do this by creating each single-task model, and having them share
        the same backbone transformer.
        """
        shared_backbone = None
        taskmodels_dict = {}
        for task in task_dict:
            model = task_dict[task]["type"].from_pretrained(
                backbone_name,
                config=task_dict[task]["config"],
            )
            if shared_backbone is None:
                shared_backbone = getattr(model, cls.get_backbone_attr_name(model))
            else:
                setattr(model, cls.get_backbone_attr_name(model), shared_backbone)
            taskmodels_dict[task] = model
        return cls(backbone=shared_backbone, taskmodels_dict=taskmodels_dict)

    @classmethod
    def get_backbone_attr_name(cls, model):
        """
        The backbone transformer is named differently in each model "architecture".
        This method lets us get the name of the backbone attribute
        """
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("XLMRoberta"):
            return "roberta"
        elif model_class_name.startswith("Bert"):
            return "bert"
        elif model_class_name.startswith("Roberta"):
            return "roberta"
        elif model_class_name.startswith("Albert"):
            return "albert"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")

    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)


from torch.autograd import Function


class RevGradFunction(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


revgrad = RevGradFunction.apply

from torch.nn import Module
from torch import tensor


class RevGrad(Module):
    def __init__(self, alpha=1.0, *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return revgrad(input_, self._alpha)


from transformers.models.roberta.modeling_roberta import *
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import MSELoss


from enum import Enum
import json


class DiscriminatorType(str, Enum):
    SingleToken = "SingleToken"
    FullSequence = "FullSequence"


class DiscriminatorConfig:
    def __init__(
        self,
        dtype: DiscriminatorType,
        length: int,
        weight: float,
        num_labels: int,
        label_id: str,
        hidden_size: int,
        nhead: int = 1,
        num_layers: int = 1,
    ):
        self.dtype = dtype
        self.length = length
        self.weight = weight
        self.num_labels = num_labels
        self.label_id = label_id
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.num_layers = num_layers


class TrainingConfig:
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])


class ExperinmentConfigSerializer(json.JSONEncoder, json.JSONDecoder):
    def default(self, o):
        return o.__dict__

    def __init__(self, *args, **kwargs):
        super(ExperinmentConfigSerializer, self).__init__()
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, config):
        if "dtype" in config:
            discriminator = DiscriminatorConfig(
                dtype=DiscriminatorType(config["dtype"]),
                length=int(config["length"]),
                weight=float(config["weight"]),
                num_labels=int(config["num_labels"]),
                label_id=str(config["label_id"]),
                hidden_size=int(config["hidden_size"]),
            )
            if "nhead" in config:
                discriminator.nhead = int(config["nhead"])
            if "num_layers" in config:
                discriminator.num_layers = int(config["num_layers"])
            return discriminator
        if "model_name" in config:
            return TrainingConfig(config)
        return config


class DisentanglerOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: dict[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class XLMRobertSingleTokenDiscriminator(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, discriminator_config: DiscriminatorConfig):
        super().__init__()
        self.portion_length = discriminator_config.length
        self.revgrad = RevGrad()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(
            discriminator_config.length, discriminator_config.hidden_size
        )
        self.classifier = nn.Linear(
            discriminator_config.hidden_size, discriminator_config.num_labels
        )

    def forward(self, features, **kwargs):
        x = features[:, :, 0 : self.portion_length]
        x = self.revgrad(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class XLMRobertFullSequenceDiscriminator(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, discriminator_config: DiscriminatorConfig):
        super().__init__()
        self.portion_length = discriminator_config.length
        self.revgrad = RevGrad()

        encoder_layers = torch.nn.TransformerEncoderLayer(
            discriminator_config.length,
            discriminator_config.nhead,
            discriminator_config.hidden_size,
            config.hidden_dropout_prob,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layers, discriminator_config.num_layers
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(
            discriminator_config.length, discriminator_config.num_labels
        )

    def forward(self, features, **kwargs):
        x = features[:, :, 0 : self.portion_length]
        x = self.revgrad(x)
        x = self.transformer_encoder(x)
        x = torch.sum(x, 1, keepdim=True)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class XLMRobertaForDisentanglement(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        self.disentangling = {}
        self.discriminators_config = config.discriminators
        for i, discriminator_config in enumerate(self.discriminators_config):
            if discriminator_config.dtype == DiscriminatorType.SingleToken:
                discriminator = XLMRobertSingleTokenDiscriminator(
                    config, discriminator_config
                )
            elif discriminator_config.dtype == DiscriminatorType.FullSequence:
                discriminator = XLMRobertFullSequenceDiscriminator(
                    config, discriminator_config
                )
            self.disentangling[i] = discriminator
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        discriminator_logits = {}
        for i, discriminator_config in enumerate(self.discriminators_config):
            discriminator_logits[i] = self.disentangling[i](
                sequence_output,
                attention_mask=attention_mask,
            )

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = 0
            for i, discriminator_config in enumerate(self.discriminators_config):
                if discriminator_config.dtype == DiscriminatorType.SingleToken:
                    loss = (
                        loss
                        + loss_fct(
                            discriminator_logits[i].view(
                                -1, discriminator_config.num_labels
                            ),
                            labels[discriminator_config.label_id]
                            .view(-1)
                            .repeat_interleave(input_ids.shape[1]),
                        )
                        * discriminator_config.weight
                    )
                elif discriminator_config.dtype == DiscriminatorType.FullSequence:
                    loss = (
                        loss
                        + loss_fct(
                            discriminator_logits[i].view(
                                -1, discriminator_config.num_labels
                            ),
                            labels[discriminator_config.label_id].view(-1),
                        )
                        * discriminator_config.weight
                    )

        if not return_dict:
            output = (discriminator_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return DisentanglerOutput(
            loss=loss,
            logits=discriminator_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
