from datasets import load_dataset
import time
import argparse

parser = argparse.ArgumentParser(description="load time of oscar corpus")
parser.add_argument(
    "lan",
    metavar="lan",
    type=str,
    help="time limit for this script",
)
args = parser.parse_args()

t = time.time()
load_dataset(
    "oscar",
    "unshuffled_deduplicated_" + args.lan,
    cache_dir="/gpfs1/scratch/ckchan666/oscar",
    ignore_verifications=True,
)
print(time.time() - t)

t = time.time()
load_dataset(
    "oscar",
    "unshuffled_deduplicated_" + args.lan,
    cache_dir="/gpfs1/scratch/ckchan666/oscar",
)
print(time.time() - t)