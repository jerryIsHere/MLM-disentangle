from datasets import load_dataset
import time

t = time.time()
load_dataset(
    "oscar",
    "unshuffled_deduplicated_zh",
    cache_dir="/gpfs1/scratch/ckchan666/oscar",
)
print(time.time() - t)

t = time.time()
load_dataset(
    "oscar",
    "unshuffled_deduplicated_zh",
    cache_dir="/gpfs1/scratch/ckchan666/oscar",
    ignore_verifications=True,
)
print(time.time() - t)
