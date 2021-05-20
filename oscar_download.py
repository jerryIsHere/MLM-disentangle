from datasets import load_dataset
import xtreme_ds
xtreme_lan = set()
for task in xtreme_ds.TASK2LANGS:
    xtreme_lan = xtreme_lan.union([lan for lan in xtreme_ds.TASK2LANGS[task]])
for lan in xtreme_lan:
    dataset = load_dataset('oscar', 'unshuffled_deduplicated_'+lan,cache_dir='/gpfs1/scratch/ckchan666/oscar')
