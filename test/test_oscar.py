from experinment_datasets import oscar_corpus
import torch

MLMD_ds = oscar_corpus.MLMDisentangleDataset()
dataloader = torch.utils.data.DataLoader(MLMD_ds, batch_size=2, num_workers=0)
for i, batch in enumerate(dataloader):
    print(i)
    print(batch)
    if i > 40:
        break
