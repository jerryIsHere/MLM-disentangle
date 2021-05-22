def get_corpus(lan):
    from datasets import load_dataset

    return load_dataset(
        "oscar",
        "unshuffled_deduplicated_" + lan,
        cache_dir="/gpfs1/scratch/ckchan666/oscar",
    )


def get_token_frequency(lan):
    import pickle

    filehandler = open(
        "/gpfs1/scratch/ckchan666/oscar_token_frequency/" + lan + ".pickle", "r"
    )
    object = pickle.load(filehandler)


def put_token_frequency(lan, word_frequency):
    import pickle

    filehandler = open(
        "/gpfs1/scratch/ckchan666/oscar_token_frequency/" + lan + ".pickle", "w"
    )
    pickle.dump(word_frequency, filehandler)
