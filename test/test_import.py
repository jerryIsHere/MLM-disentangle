from MLM_DISENTANGLE.experinment_util.experinment_config import ExperinmentConfigSerializer
from ..experinment_models.multitask_transformer import MultitaskModel
from ..experinment_models.disentangled_transformer import XLMRobertaForDisentanglement
from ..experinment_datasets import oscar_corpus
from ..experinment_datasets import xtreme_ds
print(len(xtreme_ds.xtreme_lan))
print(len(oscar_corpus.family))
print(len(oscar_corpus.genus))