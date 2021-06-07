from discriminator_config import DiscriminatorConfig, DiscriminatorType
import json


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
