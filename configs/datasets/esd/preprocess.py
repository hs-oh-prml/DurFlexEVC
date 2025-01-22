from data_gen.base_preprocess import BasePreprocessor
import os


class ESDPreprocess(BasePreprocessor):
    def meta_data(self):
        for l in open(
            "./filelist.txt"
        ).readlines():
            wav_fn, txt, emo = l.strip().split("|")
            wav_fn = os.path.join(self.raw_data_dir, wav_fn)
            item_name = os.path.basename(wav_fn).replace(".wav", "")
            spk_name = item_name.split("_")[0]
            yield {
                "item_name": item_name,
                "wav_fn": wav_fn,
                "txt": txt,
                "spk_name": spk_name,
                "emo": emo,
            }
