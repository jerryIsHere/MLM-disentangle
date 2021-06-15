import os

custom_xtreme_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "custom_xtreme.py"
)
TASK = {
    "udpos": {},  # POS
    "panx": {},  # NER
    "xnli": {},  # sentence classification
    "pawsx": {},  # sentence classification
    "xquad": {},  # QA
    "mlqa": {},  # QA
    "tydiqa": {},  # QA
    "bucc2018": {},  # retrival
    "tatoeba": {},  # retrival
}
TASK["udpos"]["train"] = (custom_xtreme_path, "udpos.English", "train")
TASK["udpos"]["epochs"] = 10
TASK["udpos"]["max seq length"] = 128
TASK["udpos"]["learning rate"] = 2e-5
TASK["udpos"]["weight decay"] = 0
TASK["udpos"]["warmup step"] = 0
TASK["udpos"]["validation"] = (custom_xtreme_path, "udpos.English", "validation")
TASK["udpos"]["test"] = {}
TASK["udpos"]["test"]["en"] = (custom_xtreme_path, "udpos.English", "test")
TASK["udpos"]["test"]["af"] = (custom_xtreme_path, "udpos.Afrikaans", "test")
TASK["udpos"]["test"]["ar"] = (custom_xtreme_path, "udpos.Arabic", "test")
TASK["udpos"]["test"]["eu"] = (custom_xtreme_path, "udpos.Basque", "test")
TASK["udpos"]["test"]["bg"] = (custom_xtreme_path, "udpos.Bulgarian", "test")
TASK["udpos"]["test"]["nl"] = (custom_xtreme_path, "udpos.Dutch", "test")
TASK["udpos"]["test"]["et"] = (custom_xtreme_path, "udpos.Estonian", "test")
TASK["udpos"]["test"]["fi"] = (custom_xtreme_path, "udpos.Finnish", "test")
TASK["udpos"]["test"]["fr"] = (custom_xtreme_path, "udpos.French", "test")
TASK["udpos"]["test"]["de"] = (custom_xtreme_path, "udpos.German", "test")
TASK["udpos"]["test"]["el"] = (custom_xtreme_path, "udpos.Greek", "test")
TASK["udpos"]["test"]["he"] = (custom_xtreme_path, "udpos.Hebrew", "test")
TASK["udpos"]["test"]["hi"] = (custom_xtreme_path, "udpos.Hindi", "test")
TASK["udpos"]["test"]["hu"] = (custom_xtreme_path, "udpos.Hungarian", "test")
TASK["udpos"]["test"]["id"] = (custom_xtreme_path, "udpos.Indonesian", "test")
TASK["udpos"]["test"]["it"] = (custom_xtreme_path, "udpos.Italian", "test")
TASK["udpos"]["test"]["ja"] = (custom_xtreme_path, "udpos.Japanese", "test")
TASK["udpos"]["test"]["kk"] = (custom_xtreme_path, "udpos.Kazakh", "test")
TASK["udpos"]["test"]["ko"] = (custom_xtreme_path, "udpos.Korean", "test")
TASK["udpos"]["test"]["zh"] = (custom_xtreme_path, "udpos.Chinese", "test")
TASK["udpos"]["test"]["mr"] = (custom_xtreme_path, "udpos.Marathi", "test")
TASK["udpos"]["test"]["fa"] = (custom_xtreme_path, "udpos.Persian", "test")
TASK["udpos"]["test"]["pt"] = (custom_xtreme_path, "udpos.Portuguese", "test")
TASK["udpos"]["test"]["ru"] = (custom_xtreme_path, "udpos.Russian", "test")
TASK["udpos"]["test"]["es"] = (custom_xtreme_path, "udpos.Spanish", "test")
TASK["udpos"]["test"]["tl"] = (custom_xtreme_path, "udpos.Tagalog", "test")
TASK["udpos"]["test"]["ta"] = (custom_xtreme_path, "udpos.Tamil", "test")
TASK["udpos"]["test"]["te"] = (custom_xtreme_path, "udpos.Telugu", "test")
TASK["udpos"]["test"]["th"] = (custom_xtreme_path, "udpos.Thai", "test")
TASK["udpos"]["test"]["tr"] = (custom_xtreme_path, "udpos.Turkish", "test")
TASK["udpos"]["test"]["ur"] = (custom_xtreme_path, "udpos.Urdu", "test")
TASK["udpos"]["test"]["vi"] = (custom_xtreme_path, "udpos.Vietnamese", "test")
TASK["udpos"]["test"]["yo"] = (custom_xtreme_path, "udpos.Yoruba", "test")


TASK["panx"]["train"] = ("xtreme", "PAN-X.en", "train")
TASK["panx"]["epochs"] = 10
TASK["panx"]["max seq length"] = 128
TASK["panx"]["learning rate"] = 2e-5
TASK["panx"]["warmup step"] = 0
TASK["panx"]["weight decay"] = 0
TASK["panx"]["validation"] = ("xtreme", "PAN-X.en", "validation")
TASK["panx"]["test"] = {}
TASK["panx"]["test"]["en"] = ("xtreme", "PAN-X.en", "test")
TASK["panx"]["test"]["af"] = ("xtreme", "PAN-X.af", "test")
TASK["panx"]["test"]["ar"] = ("xtreme", "PAN-X.ar", "test")
TASK["panx"]["test"]["bg"] = ("xtreme", "PAN-X.bg", "test")
TASK["panx"]["test"]["bn"] = ("xtreme", "PAN-X.bn", "test")
TASK["panx"]["test"]["de"] = ("xtreme", "PAN-X.de", "test")
TASK["panx"]["test"]["el"] = ("xtreme", "PAN-X.el", "test")
TASK["panx"]["test"]["en"] = ("xtreme", "PAN-X.en", "test")
TASK["panx"]["test"]["es"] = ("xtreme", "PAN-X.es", "test")
TASK["panx"]["test"]["et"] = ("xtreme", "PAN-X.et", "test")
TASK["panx"]["test"]["eu"] = ("xtreme", "PAN-X.eu", "test")
TASK["panx"]["test"]["fa"] = ("xtreme", "PAN-X.fa", "test")
TASK["panx"]["test"]["fi"] = ("xtreme", "PAN-X.fi", "test")
TASK["panx"]["test"]["fr"] = ("xtreme", "PAN-X.fr", "test")
TASK["panx"]["test"]["he"] = ("xtreme", "PAN-X.he", "test")
TASK["panx"]["test"]["hi"] = ("xtreme", "PAN-X.hi", "test")
TASK["panx"]["test"]["hu"] = ("xtreme", "PAN-X.hu", "test")
TASK["panx"]["test"]["id"] = ("xtreme", "PAN-X.id", "test")
TASK["panx"]["test"]["it"] = ("xtreme", "PAN-X.it", "test")
TASK["panx"]["test"]["ja"] = ("xtreme", "PAN-X.ja", "test")
TASK["panx"]["test"]["jv"] = ("xtreme", "PAN-X.jv", "test")
TASK["panx"]["test"]["ka"] = ("xtreme", "PAN-X.ka", "test")
TASK["panx"]["test"]["kk"] = ("xtreme", "PAN-X.kk", "test")
TASK["panx"]["test"]["ko"] = ("xtreme", "PAN-X.ko", "test")
TASK["panx"]["test"]["ml"] = ("xtreme", "PAN-X.ml", "test")
TASK["panx"]["test"]["mr"] = ("xtreme", "PAN-X.mr", "test")
TASK["panx"]["test"]["ms"] = ("xtreme", "PAN-X.ms", "test")
TASK["panx"]["test"]["my"] = ("xtreme", "PAN-X.my", "test")
TASK["panx"]["test"]["nl"] = ("xtreme", "PAN-X.nl", "test")
TASK["panx"]["test"]["pt"] = ("xtreme", "PAN-X.pt", "test")
TASK["panx"]["test"]["ru"] = ("xtreme", "PAN-X.ru", "test")
TASK["panx"]["test"]["sw"] = ("xtreme", "PAN-X.sw", "test")
TASK["panx"]["test"]["ta"] = ("xtreme", "PAN-X.ta", "test")
TASK["panx"]["test"]["te"] = ("xtreme", "PAN-X.te", "test")
TASK["panx"]["test"]["th"] = ("xtreme", "PAN-X.th", "test")
TASK["panx"]["test"]["tl"] = ("xtreme", "PAN-X.tl", "test")
TASK["panx"]["test"]["tr"] = ("xtreme", "PAN-X.tr", "test")
TASK["panx"]["test"]["ur"] = ("xtreme", "PAN-X.ur", "test")
TASK["panx"]["test"]["vi"] = ("xtreme", "PAN-X.vi", "test")
TASK["panx"]["test"]["yo"] = ("xtreme", "PAN-X.yo", "test")
TASK["panx"]["test"]["zh"] = ("xtreme", "PAN-X.zh", "test")

TASK["xnli"]["train"] = ("multi_nli", None, "train")
TASK["xnli"]["epochs"] = 5
TASK["xnli"]["max seq length"] = 128
TASK["xnli"]["learning rate"] = 2e-5
TASK["xnli"]["warmup step"] = 0
TASK["xnli"]["weight decay"] = 0
TASK["xnli"]["validation"] = {}
TASK["xnli"]["validation"]["matched"] = ("multi_nli", None, "validation_matched")
TASK["xnli"]["validation"]["mismatched"] = ("multi_nli", None, "validation_mismatched")
TASK["xnli"]["test"] = ("xtreme", "XNLI", "test")

TASK["pawsx"]["train"] = ("xtreme", "PAWS-X.en", "train")
TASK["pawsx"]["epochs"] = 5
TASK["pawsx"]["max seq length"] = 128
TASK["pawsx"]["learning rate"] = 2e-5
TASK["pawsx"]["warmup step"] = 0
TASK["pawsx"]["weight decay"] = 0
TASK["pawsx"]["validation"] = ("xtreme", "PAWS-X.en", "validation")
TASK["pawsx"]["test"] = {}
TASK["pawsx"]["test"]["en"] = ("xtreme", "PAWS-X.en", "test")
TASK["pawsx"]["test"]["es"] = ("xtreme", "PAWS-X.es", "test")
TASK["pawsx"]["test"]["de"] = ("xtreme", "PAWS-X.de", "test")
TASK["pawsx"]["test"]["fr"] = ("xtreme", "PAWS-X.fr", "test")
TASK["pawsx"]["test"]["ja"] = ("xtreme", "PAWS-X.ja", "test")
TASK["pawsx"]["test"]["ko"] = ("xtreme", "PAWS-X.ko", "test")
TASK["pawsx"]["test"]["zh"] = ("xtreme", "PAWS-X.zh", "test")

TASK["xquad"]["train"] = ("xtreme", "SQuAD", "train")
TASK["xquad"]["epochs"] = 2
TASK["xquad"]["max seq length"] = 384
TASK["xquad"]["learning rate"] = 3e-5
TASK["xquad"]["warmup step"] = 500
TASK["xquad"]["weight decay"] = 0.0001
TASK["xquad"]["validation"] = ("xtreme", "SQuAD", "validation")
TASK["xquad"]["test"] = {}
TASK["xquad"]["test"]["ar"] = ("xtreme", "XQuAD.ar", "validation")
TASK["xquad"]["test"]["de"] = ("xtreme", "XQuAD.de", "validation")
TASK["xquad"]["test"]["vi"] = ("xtreme", "XQuAD.vi", "validation")
TASK["xquad"]["test"]["zh"] = ("xtreme", "XQuAD.zh", "validation")
TASK["xquad"]["test"]["en"] = ("xtreme", "XQuAD.en", "validation")
TASK["xquad"]["test"]["es"] = ("xtreme", "XQuAD.es", "validation")
TASK["xquad"]["test"]["hi"] = ("xtreme", "XQuAD.hi", "validation")
TASK["xquad"]["test"]["el"] = ("xtreme", "XQuAD.el", "validation")
TASK["xquad"]["test"]["ru"] = ("xtreme", "XQuAD.ru", "validation")
TASK["xquad"]["test"]["th"] = ("xtreme", "XQuAD.th", "validation")
TASK["xquad"]["test"]["tr"] = ("xtreme", "XQuAD.tr", "validation")

"""
Xtreme ignore subset of mlqa which if context & question with different langauge
https://github.com/google-research/xtreme/blob/master/scripts/eval_qa.sh
echo "MLQA"
for lang in en es de ar hi vi zh; do
 echo -n "  $lang "
 TEST_FILE=${MLQA_DIR}/MLQA_V1/test/test-context-$lang-question-$lang.json
 PRED_FILE=${MLQA_PRED_DIR}/predictions_${lang}_.json
 python "${EVAL_MLQA}" "${TEST_FILE}" "${PRED_FILE}" ${lang}
done
"""
TASK["mlqa"]["train"] = ("xtreme", "SQuAD", "train")
TASK["mlqa"]["epochs"] = 2
TASK["mlqa"]["max seq length"] = 384
TASK["mlqa"]["learning rate"] = 3e-5
TASK["mlqa"]["warmup step"] = 500
TASK["mlqa"]["weight decay"] = 0.0001
TASK["mlqa"]["validation"] = ("xtreme", "SQuAD", "validation")
TASK["mlqa"]["test"] = {}
TASK["mlqa"]["test"]["ar"] = ("xtreme", "MLQA.ar.ar", "test")
TASK["mlqa"]["test"]["de"] = ("xtreme", "MLQA.de.de", "test")
TASK["mlqa"]["test"]["vi"] = ("xtreme", "MLQA.vi.vi", "test")
TASK["mlqa"]["test"]["zh"] = ("xtreme", "MLQA.zh.zh", "test")
TASK["mlqa"]["test"]["en"] = ("xtreme", "MLQA.en.en", "test")
TASK["mlqa"]["test"]["es"] = ("xtreme", "MLQA.es.es", "test")
TASK["mlqa"]["test"]["hi"] = ("xtreme", "MLQA.hi.hi", "test")

TASK["tydiqa"]["train"] = ("xtreme", "tydiqa", "train")
TASK["tydiqa"]["epochs"] = 2
TASK["tydiqa"]["max seq length"] = 384
TASK["tydiqa"]["learning rate"] = 3e-5
TASK["tydiqa"]["warmup step"] = 500
TASK["tydiqa"]["weight decay"] = 0.0001
TASK["tydiqa"]["test"] = ("xtreme", "tydiqa", "validation")

TASK["bucc2018"]["src"] = {}
TASK["bucc2018"]["max seq length"] = 512
TASK["bucc2018"]["src"]["de"] = ("xtreme", "bucc18.de", "validation")
TASK["bucc2018"]["src"]["fr"] = ("xtreme", "bucc18.fr", "validation")
TASK["bucc2018"]["src"]["zh"] = ("xtreme", "bucc18.zh", "validation")
TASK["bucc2018"]["src"]["ru"] = ("xtreme", "bucc18.ru", "validation")

TASK["tatoeba"]["src"] = {}
TASK["tatoeba"]["max seq length"] = 512
TASK["tatoeba"]["src"]["afr"] = ("xtreme", "tatoeba.afr", "validation")
TASK["tatoeba"]["src"]["ara"] = ("xtreme", "tatoeba.ara", "validation")
TASK["tatoeba"]["src"]["ben"] = ("xtreme", "tatoeba.ben", "validation")
TASK["tatoeba"]["src"]["bul"] = ("xtreme", "tatoeba.bul", "validation")
TASK["tatoeba"]["src"]["deu"] = ("xtreme", "tatoeba.deu", "validation")
TASK["tatoeba"]["src"]["cmn"] = ("xtreme", "tatoeba.cmn", "validation")
TASK["tatoeba"]["src"]["ell"] = ("xtreme", "tatoeba.ell", "validation")
TASK["tatoeba"]["src"]["est"] = ("xtreme", "tatoeba.est", "validation")
TASK["tatoeba"]["src"]["eus"] = ("xtreme", "tatoeba.eus", "validation")
TASK["tatoeba"]["src"]["fin"] = ("xtreme", "tatoeba.fin", "validation")
TASK["tatoeba"]["src"]["fra"] = ("xtreme", "tatoeba.fra", "validation")
TASK["tatoeba"]["src"]["heb"] = ("xtreme", "tatoeba.heb", "validation")
TASK["tatoeba"]["src"]["hin"] = ("xtreme", "tatoeba.hin", "validation")
TASK["tatoeba"]["src"]["hun"] = ("xtreme", "tatoeba.hun", "validation")
TASK["tatoeba"]["src"]["ind"] = ("xtreme", "tatoeba.ind", "validation")
TASK["tatoeba"]["src"]["ita"] = ("xtreme", "tatoeba.ita", "validation")
TASK["tatoeba"]["src"]["jav"] = ("xtreme", "tatoeba.jav", "validation")
TASK["tatoeba"]["src"]["jpn"] = ("xtreme", "tatoeba.jpn", "validation")
TASK["tatoeba"]["src"]["kat"] = ("xtreme", "tatoeba.kat", "validation")
TASK["tatoeba"]["src"]["kaz"] = ("xtreme", "tatoeba.kaz", "validation")
TASK["tatoeba"]["src"]["kor"] = ("xtreme", "tatoeba.kor", "validation")
TASK["tatoeba"]["src"]["mal"] = ("xtreme", "tatoeba.mal", "validation")
TASK["tatoeba"]["src"]["mar"] = ("xtreme", "tatoeba.mar", "validation")
TASK["tatoeba"]["src"]["nld"] = ("xtreme", "tatoeba.nld", "validation")
TASK["tatoeba"]["src"]["pes"] = ("xtreme", "tatoeba.pes", "validation")
TASK["tatoeba"]["src"]["por"] = ("xtreme", "tatoeba.por", "validation")
TASK["tatoeba"]["src"]["rus"] = ("xtreme", "tatoeba.rus", "validation")
TASK["tatoeba"]["src"]["spa"] = ("xtreme", "tatoeba.spa", "validation")
TASK["tatoeba"]["src"]["swh"] = ("xtreme", "tatoeba.swh", "validation")
TASK["tatoeba"]["src"]["tam"] = ("xtreme", "tatoeba.tam", "validation")
TASK["tatoeba"]["src"]["tgl"] = ("xtreme", "tatoeba.tgl", "validation")
TASK["tatoeba"]["src"]["tha"] = ("xtreme", "tatoeba.tha", "validation")
TASK["tatoeba"]["src"]["tur"] = ("xtreme", "tatoeba.tur", "validation")
TASK["tatoeba"]["src"]["urd"] = ("xtreme", "tatoeba.urd", "validation")
TASK["tatoeba"]["src"]["vie"] = ("xtreme", "tatoeba.vie", "validation")

TASK["tatoeba"]["src"]["tel"] = ("xtreme", "tatoeba.tel", "validation")

# copied from https://github.com/google-research/xtreme/blob/master/third_party/run_retrieval.py
# https://github.com/google-research/xtreme/blob/a0ffd2453e3d8060f20d5804f45a1b75cf7a54fc/utils_preprocess.py
lang3_dict = {
    "afr": "af",
    "ara": "ar",
    "bul": "bg",
    "ben": "bn",
    "deu": "de",
    "ell": "el",
    "spa": "es",
    "est": "et",
    "eus": "eu",
    "pes": "fa",
    "fin": "fi",
    "fra": "fr",
    "heb": "he",
    "hin": "hi",
    "hun": "hu",
    "ind": "id",
    "ita": "it",
    "jpn": "ja",
    "jav": "jv",
    "kat": "ka",
    "kaz": "kk",
    "kor": "ko",
    "mal": "ml",
    "mar": "mr",
    "nld": "nl",
    "por": "pt",
    "rus": "ru",
    "swh": "sw",
    "tam": "ta",
    "tel": "te",
    "tha": "th",
    "tgl": "tl",
    "tur": "tr",
    "urd": "ur",
    "vie": "vi",
    "cmn": "zh",
    # 'eng':'en',
}

TASK["tatoeba"]["src"] = {
    code2: TASK["tatoeba"]["src"].get(code3) for (code3, code2) in lang3_dict.items()
}

TASK2LANGS = {
    "pawsx": "de,en,es,fr,ja,ko,zh".split(","),
    "xnli": "ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh".split(","),
    "panx": "ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu".split(
        ","
    ),
    "udpos": "af,ar,bg,de,el,en,es,et,eu,fa,fi,fr,he,hi,hu,id,it,ja,kk,ko,mr,nl,pt,ru,ta,te,th,tl,tr,ur,vi,yo,zh".split(
        ","
    ),
    "bucc2018": "de,fr,ru,zh".split(","),
    "tatoeba": "ar,he,vi,id,jv,tl,eu,ml,ta,te,af,nl,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,zh,kk,tr,et,fi,hu".split(
        ","
    ),
    "xquad": "en,es,de,el,ru,tr,ar,vi,th,zh,hi".split(","),
    "mlqa": "en,es,de,ar,hi,vi,zh".split(","),
    "tydiqa": "en,ar,bn,fi,id,ko,ru,sw,te".split(","),
}
keywords = ["epochs", "learning rate", "warmup step", "weight decay"]


def sanity_check():
    assert TASK2LANGS.keys() == TASK.keys()
    for task in TASK:
        if "max seq length" not in TASK[task]:
            print(task + ": max seq length not found")
        if task == "bucc2018" or task == "tatoeba":
            for code in TASK2LANGS[task]:
                if code not in TASK[task]["src"]:
                    print(task + ": test split for " + code + " not found")
        else:
            for word in keywords:
                if word not in TASK[task]:
                    print(task + ": " + word + " not found")
            if task == "xnli" or task == "tydiqa":
                continue
            for code in TASK2LANGS[task]:
                if code not in TASK[task]["test"]:
                    print(task + ": test split for " + code + " not found")

    def check_dataset(ds, task):
        pass

    print(
        "sanity check done!\nlanguage code check is not done for xnli or tydiqa\nplease specify data dir for panx"
    )


def get_dataset(set_name, subset_name):
    if subset_name == "tatoeba.tel":
        import os
        import pandas as pd
        from datasets import Dataset

        if os.path.exists("/gpfs1/scratch/ckchan666/xtreme/tatoeba.tel/"):
            df = pd.read_pickle("/gpfs1/scratch/ckchan666/xtreme/tatoeba.tel/cache.pd")
        else:
            os.mkdir("/gpfs1/scratch/ckchan666/xtreme/tatoeba.tel/")
            import requests
            import io

            url = "https://raw.githubusercontent.com/facebookresearch/LASER/master/data/tatoeba/v1/tatoeba.tel-eng.eng"
            s = requests.get(url).content
            eng = pd.read_csv(
                io.StringIO(s.decode("utf-8")),
                sep="\n",
                names=["target_sentence"],
                encoding="utf8",
            )
            url = "https://raw.githubusercontent.com/facebookresearch/LASER/master/data/tatoeba/v1/tatoeba.tel-eng.tel"
            s = requests.get(url).content
            tel = pd.read_csv(
                io.StringIO(s.decode("utf-8")),
                sep="\n",
                names=["source_sentence"],
                encoding="utf8",
            )
            df = pd.concat((eng, tel), axis=1)
            df.to_pickle("/gpfs1/scratch/ckchan666/xtreme/tatoeba.tel/cache.pd")
        return {"validation": Dataset.from_pandas(df, split="validation")}
    from datasets import load_dataset

    data_dir = (
        "/gpfs1/scratch/ckchan666/xtreme"
        if subset_name is not None and "PAN-X" in subset_name
        else None
    )
    return load_dataset(
        set_name,
        subset_name,
        ignore_verifications=True,
        data_dir=data_dir,
        cache_dir="/gpfs1/scratch/ckchan666/xtreme",
    )


def summary():
    for task in TASK:
        if "train" in TASK[task]:
            set_name, subset_name, split = TASK[task]["train"]
            print(
                "training data for "
                + task
                + " : "
                + str(len(get_dataset(set_name, subset_name)[split]))
            )
        if "validation" in TASK[task]:
            if type(TASK[task]["validation"]) is tuple:
                set_name, subset_name, split = TASK[task]["validation"]
                print(
                    "validation data for "
                    + task
                    + " : "
                    + str(len(get_dataset(set_name, subset_name)[split]))
                )
            else:
                for split in TASK[task]["validation"]:
                    set_name, subset_name, split = TASK[task]["validation"][split]
                    print(
                        "validation data ("
                        + split
                        + ") for "
                        + task
                        + " : "
                        + str(len(get_dataset(set_name, subset_name)[split]))
                    )
        if "test" in TASK[task]:
            if type(TASK[task]["test"]) is tuple:
                set_name, subset_name, split = TASK[task]["test"]
                print(
                    "test data for "
                    + task
                    + " : "
                    + str(len(get_dataset(set_name, subset_name)[split]))
                )
            else:
                for lan in TASK[task]["test"]:
                    set_name, subset_name, split = TASK[task]["test"][lan]
                    print(
                        "test data ("
                        + lan
                        + ") for "
                        + task
                        + " : "
                        + str(len(get_dataset(set_name, subset_name)[split]))
                    )
        if "src" in TASK[task]:
            for lan in TASK[task]["src"]:
                set_name, subset_name, split = TASK[task]["src"][lan]
                print(
                    "training data ("
                    + lan
                    + ") for "
                    + task
                    + " : "
                    + str(len(get_dataset(set_name, subset_name)[split]))
                )


xtreme_lan = [
    "af",
    "ar",
    "bg",
    "bn",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fr",
    "he",
    "hi",
    "hu",
    "id",
    "it",
    "ja",
    "jv",
    "ka",
    "kk",
    "ko",
    "ml",
    "mr",
    "ms",
    "my",
    "nl",
    "pt",
    "ru",
    "sw",
    "ta",
    "te",
    "th",
    "tl",
    "tr",
    "ur",
    "vi",
    "yo",
    "zh",
]
from transformers import XLMRobertaTokenizerFast

tokenizer = XLMRobertaTokenizerFast.from_pretrained(
    "xlm-roberta-large",
    cache_dir="/gpfs1/scratch/ckchan666/transformer_model_cache",
)
import numpy as np
import torch


class udposTrainDataset(torch.utils.data.Dataset):
    task = "udpos"

    def __init__(self):
        set_name, subset_name, split = TASK[udposTrainDataset.task]["train"]
        self.dataset = get_dataset(set_name, subset_name)[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        features = self.dataset[id]
        txt = features["tokens"]
        train_encodings = tokenizer(
            txt,
            is_split_into_words=True,
            max_length=TASK[udposTrainDataset.task]["max seq length"],
            truncation=True,
            return_offsets_mapping=True,
        )
        labels = np.ones(len(train_encodings.input_ids), dtype=int) * -100
        ids = np.array(train_encodings.input_ids)
        arr_offset = np.array(train_encodings.offset_mapping)
        labels[
            (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0) & (ids[:] != 6)
        ] = self.dataset[id]["pos_tags"]
        return {
            "tokens": torch.from_numpy(ids).long(),
            "pos_tags": torch.from_numpy(labels).long(),
        }


class udposValidationDataset(torch.utils.data.Dataset):
    task = "udpos"

    def __init__(self):
        set_name, subset_name, split = TASK[udposValidationDataset.task]["validation"]
        self.dataset = get_dataset(set_name, subset_name)[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        features = self.dataset[id]
        txt = features["tokens"]
        train_encodings = tokenizer(
            txt,
            is_split_into_words=True,
            max_length=TASK[udposValidationDataset.task]["max seq length"],
            truncation=True,
            return_offsets_mapping=True,
        )
        labels = np.ones(len(train_encodings.input_ids), dtype=int) * -100
        ids = np.array(train_encodings.input_ids)
        arr_offset = np.array(train_encodings.offset_mapping)
        labels[
            (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0) & (ids[:] != 6)
        ] = self.dataset[id]["pos_tags"]
        return {
            "tokens": torch.from_numpy(ids).long(),
            "pos_tags": torch.from_numpy(labels).long(),
        }


class udposTestDataset(torch.utils.data.Dataset):
    task = "udpos"

    def __init__(self):
        self.dataset = {}
        for lan in TASK[udposTestDataset.task]["test"]:
            set_name, subset_name, split = TASK[udposTestDataset.task]["test"][lan]
            self.dataset[lan] = get_dataset(set_name, subset_name)[split]

    def __len__(self):
        return sum(map(lambda x: len(x), self.dataset.items))

    def __getitem__(self, id_absolute):
        for lan in self.dataset:
            length = len(self.dataset[lan])
            if id_absolute < length:
                id = id_absolute
                break
            id_absolute -= length
        features = self.dataset[lan][id]
        txt = features["tokens"]
        train_encodings = tokenizer(
            txt,
            is_split_into_words=True,
            max_length=TASK[udposTestDataset.task]["max seq length"],
            truncation=True,
            return_offsets_mapping=True,
        )
        labels = np.ones(len(train_encodings.input_ids), dtype=int) * -100
        ids = np.array(train_encodings.input_ids)
        arr_offset = np.array(train_encodings.offset_mapping)
        labels[
            (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0) & (ids[:] != 6)
        ] = self.dataset[lan][id]["pos_tags"]
        return {
            "tokens": torch.from_numpy(ids).long(),
            "pos_tags": torch.from_numpy(labels).long(),
        }


class panxTrainDataset(torch.utils.data.Dataset):
    task = "panx"

    def __init__(self):
        set_name, subset_name, split = TASK[panxTrainDataset.task]["train"]
        self.dataset = get_dataset(set_name, subset_name)[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        features = self.dataset[id]
        txt = features["tokens"]
        train_encodings = tokenizer(
            txt,
            is_split_into_words=True,
            max_length=TASK[panxTrainDataset.task]["max seq length"],
            truncation=True,
            return_offsets_mapping=True,
        )
        labels = np.ones(len(train_encodings.input_ids), dtype=int) * -100
        ids = np.array(train_encodings.input_ids)
        arr_offset = np.array(train_encodings.offset_mapping)
        labels[
            (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0) & (ids[:] != 6)
        ] = self.dataset[id]["ner_tags"]
        return {
            "tokens": torch.from_numpy(ids).long(),
            "ner_tags": torch.from_numpy(labels).long(),
        }


class panxValidationDataset(torch.utils.data.Dataset):
    task = "panx"

    def __init__(self):
        set_name, subset_name, split = TASK[panxValidationDataset.task]["validation"]
        self.dataset = get_dataset(set_name, subset_name)[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        features = self.dataset[id]
        txt = features["tokens"]
        train_encodings = tokenizer(
            txt,
            is_split_into_words=True,
            max_length=TASK[panxValidationDataset.task]["max seq length"],
            truncation=True,
            return_offsets_mapping=True,
        )
        labels = np.ones(len(train_encodings.input_ids), dtype=int) * -100
        ids = np.array(train_encodings.input_ids)
        arr_offset = np.array(train_encodings.offset_mapping)
        labels[
            (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0) & (ids[:] != 6)
        ] = self.dataset[id]["ner_tags"]
        return {
            "tokens": torch.from_numpy(ids).long(),
            "ner_tags": torch.from_numpy(labels).long(),
        }


class panxTestDataset(torch.utils.data.Dataset):
    task = "panx"

    def __init__(self):
        self.dataset = {}
        for lan in TASK[panxTestDataset.task]["test"]:
            set_name, subset_name, split = TASK[panxTestDataset.task]["test"][lan]
            self.dataset[lan] = get_dataset(set_name, subset_name)[split]

    def __len__(self):
        return sum(map(lambda x: len(x), self.dataset.items))

    def __getitem__(self, id_absolute):
        for lan in self.dataset:
            length = len(self.dataset[lan])
            if id_absolute < length:
                id = id_absolute
                break
            id_absolute -= length
        features = self.dataset[lan][id]
        txt = features["tokens"]
        train_encodings = tokenizer(
            txt,
            is_split_into_words=True,
            max_length=TASK[panxTestDataset.task]["max seq length"],
            truncation=True,
            return_offsets_mapping=True,
        )
        labels = np.ones(len(train_encodings.input_ids), dtype=int) * -100
        ids = np.array(train_encodings.input_ids)
        arr_offset = np.array(train_encodings.offset_mapping)
        labels[
            (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0) & (ids[:] != 6)
        ] = self.dataset[lan][id]["ner_tags"]
        return {
            "tokens": torch.from_numpy(ids).long(),
            "ner_tags": torch.from_numpy(labels).long(),
        }


class xquadTrainDataset(torch.utils.data.Dataset):
    task = "xquad"

    def __init__(self):
        set_name, subset_name, split = TASK[xquadTrainDataset.task]["train"]
        self.dataset = get_dataset(set_name, subset_name)[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        features = self.dataset[id]
        train_encodings = tokenizer(
            features["context"],
            features["question"],
            return_tensors="pt",
            max_length=TASK[panxTestDataset.task]["max seq length"],
            truncation=True,
        )
        return {
            "tokens": train_encodings.input_ids.long(),
            "start_positions": torch.Tensor(
                [features["answers"]["answer_start"][0]]
            ).long(),
            "end_positions": torch.Tensor(
                [
                    features["answers"]["answer_start"][0]
                    + len(features["answers"]["text"][0])
                ]
            ).long(),
        }


class xquadValidationDataset(torch.utils.data.Dataset):
    task = "xquad"

    def __init__(self):
        set_name, subset_name, split = TASK[xquadValidationDataset.task]["validation"]
        self.dataset = get_dataset(set_name, subset_name)[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        features = self.dataset[id]
        train_encodings = tokenizer(
            features["context"],
            features["question"],
            return_tensors="pt",
            max_length=TASK[panxTestDataset.task]["max seq length"],
            truncation=True,
        )
        return {
            "tokens": train_encodings.input_ids.long(),
            "start_positions": torch.Tensor(
                [features["answers"]["answer_start"][0]]
            ).long(),
            "end_positions": torch.Tensor(
                [
                    features["answers"]["answer_start"][0]
                    + len(features["answers"]["text"][0])
                ]
            ).long(),
        }


class xquadTestDataset(torch.utils.data.Dataset):
    task = "xquad"

    def __init__(self):
        self.dataset = {}
        for lan in TASK[xquadTestDataset.task]["test"]:
            set_name, subset_name, split = TASK[xquadTestDataset.task]["test"][lan]
            self.dataset[lan] = get_dataset(set_name, subset_name)[split]

    def __len__(self):
        return sum(map(lambda x: len(x), self.dataset.items))

    def __getitem__(self, id_absolute):
        for lan in self.dataset:
            length = len(self.dataset[lan])
            if id_absolute < length:
                id = id_absolute
                break
            id_absolute -= length
        features = self.dataset[lan][id]
        train_encodings = tokenizer(
            features["context"],
            features["question"],
            return_tensors="pt",
            max_length=TASK[panxTestDataset.task]["max seq length"],
            truncation=True,
        )
        return {
            "tokens": train_encodings.input_ids.long(),
            "start_positions": torch.Tensor(
                [features["answers"]["answer_start"][0]]
            ).long(),
            "end_positions": torch.Tensor(
                [
                    features["answers"]["answer_start"][0]
                    + len(features["answers"]["text"][0])
                ]
            ).long(),
        }
