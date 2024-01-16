from data_gen.tts.base_preprocess import BasePreprocessor
import os


class ESDPreprocess(BasePreprocessor):
    def meta_data(self):
        for l in open(
            "/workspace/hs_oh/dataset/Emotional_Speech_Dataset_ESD/eng/esd_all_v2.txt"
        ).readlines():
            wav_fn, txt, emo = l.strip().split("|")
            item_name = os.path.basename(wav_fn).replace(".wav", "")
            spk_name = item_name.split("_")[0]
            yield {
                "item_name": item_name,
                "wav_fn": wav_fn,
                "txt": txt,
                "spk_name": spk_name,
                "emo": emo,
            }
