from enum import Enum


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
