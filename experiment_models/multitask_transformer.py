import torch
import torch.nn as nn
import transformers

# https://colab.research.google.com/github/zphang/zphang.github.io/blob/master/files/notebooks/Multi_task_Training_with_Transformers_NLP.ipynb#scrollTo=7zSZsp8Cb7gd


class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, backbone, taskmodels_dict, backbone_name):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())
        self.backbone_name = backbone_name
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
        return cls(
            backbone=shared_backbone,
            taskmodels_dict=taskmodels_dict,
            backbone_name=backbone_name,
        )

    @classmethod
    def create_untrained(cls, backbone_name, task_dict):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models.

        We do this by creating each single-task model, and having them share
        the same backbone transformer.
        """
        shared_backbone = None
        taskmodels_dict = {}
        for task in task_dict:
            model = task_dict[task]["type"](
                config=task_dict[task]["config"],
            )
            if shared_backbone is None:
                shared_backbone = getattr(model, cls.get_backbone_attr_name(model))
            else:
                setattr(model, cls.get_backbone_attr_name(model), shared_backbone)
            taskmodels_dict[task] = model
        return cls(
            backbone=shared_backbone,
            taskmodels_dict=taskmodels_dict,
            backbone_name=backbone_name,
        )

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

    def add_task(self, task_name, task_cls, config):
        model = task_cls.from_pretrained(
            self.backbone_name,
            config=config,
        )
        setattr(model, MultitaskModel.get_backbone_attr_name(model), self.backbone)
        self.taskmodels_dict[task_name] = model
