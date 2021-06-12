import numpy as np
import torch
import torch.nn as nn
import transformers
from typing import Optional, Tuple
from transformers.models.roberta.modeling_roberta import *
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import MSELoss
from experinment_util.discriminator_config import DiscriminatorType, DiscriminatorConfig
from experinment_models.gradient_reverser import RevGrad


class DisentanglerOutput(transformers.file_utils.ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: dict = None
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
        self.discriminators_config = {
            str(i): discriminator_config
            for i, discriminator_config in enumerate(self.discriminators_config)
        }
        for i, discriminator_config in self.discriminators_config.items():
            if discriminator_config.dtype == DiscriminatorType.SingleToken:
                discriminator = XLMRobertSingleTokenDiscriminator(
                    config, discriminator_config
                )
            elif discriminator_config.dtype == DiscriminatorType.FullSequence:
                discriminator = XLMRobertFullSequenceDiscriminator(
                    config, discriminator_config
                )
            self.disentangling[i] = discriminator
        self.disentangling = nn.ModuleDict(self.disentangling)
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
        for i, discriminator_config in self.discriminators_config.items():
            discriminator_logits[i] = self.disentangling[i](
                sequence_output,
                attention_mask=attention_mask,
            )

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = 0
            for i, discriminator_config in self.discriminators_config.items():
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
