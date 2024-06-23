from torch import cuda
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from os.path import realpath, dirname, join
from pathlib import Path

dir_ = dirname(realpath(__file__))
dir_ = Path(dir_)
# chdir(dir_)

TOKENIZER_PATH = join(dir_, r"saved_model/tokenizer/")
MODEL_PATH = join(dir_, r"saved_model/nllb/")
SRC_LANG = "dyu_Latn"
TGT_LANG = "fra_Latn"
DEVICE = "cuda:0" if cuda.is_available() else "cpu"

class Translater:
    def __init__(self) -> None:
        self.tokenizer: AutoTokenizer = None
        self.model: AutoModelForSeq2SeqLM = None

        self.load()

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH,
            local_files_only=True,
            src_lang = SRC_LANG,
            tgt_lang = TGT_LANG
        )
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        ).to(DEVICE)

    def predict(self, input_string, *args, **kwargs):
        
        inputs = self.tokenizer(
            input_string,
            return_tensors = "pt",
            padding = True,
            truncation = True
        ).to(DEVICE)

        translation = self.model.generate(
            **inputs,
            forced_bos_token_id = self.tokenizer.lang_code_to_id[TGT_LANG]
        )

        translation = self.tokenizer.decode(translation[0], skip_special_tokens=True)

        return translation



if __name__ == "__main__":
    translater = Translater()

    print(translater.predict("Ne ye Ameriken ye"))