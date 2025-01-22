import os
import json
import random
import numpy as np
from tqdm import tqdm
from functools import partial
from resemblyzer import VoiceEncoder

from utils.audio import wav2spec
from utils.commons.hparams import hparams
from utils.commons.indexed_datasets import IndexedDatasetBuilder
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from utils.os_utils import remove_file, copy_file

np.seterr(divide="ignore", invalid="ignore")


class BinarizationError(Exception):
    pass


class Binarizer:
    def __init__(self, processed_data_dir=None):
        if processed_data_dir is None:
            processed_data_dir = hparams["processed_data_dir"]
        self.processed_data_dir = processed_data_dir
        self.binarization_args = hparams["binarization_args"]
        self.items = {}
        self.item_names = []

    def load_meta_data(self):
        processed_data_dir = self.processed_data_dir
        items_list = json.load(open(f"{processed_data_dir}/metadata.json"))
        for r in tqdm(items_list, desc="Loading meta data."):
            item_name = r["item_name"]
            self.items[item_name] = r
            self.item_names.append(item_name)

        if self.binarization_args["shuffle"]:
            random.seed(1234)
            random.shuffle(self.item_names)

    @property
    def train_item_names(self):
        range_ = self._convert_range(self.binarization_args["train_range"])
        return self.item_names[range_[0] : range_[1]]

    @property
    def valid_item_names(self):
        range_ = self._convert_range(self.binarization_args["valid_range"])
        return self.item_names[range_[0] : range_[1]]

    @property
    def test_item_names(self):
        range_ = self._convert_range(self.binarization_args["test_range"])
        return self.item_names[range_[0] : range_[1]]

    def _convert_range(self, range_):
        if range_[1] == -1:
            range_[1] = len(self.item_names)
        return range_

    def meta_data(self, prefix):
        if prefix == "valid":
            item_names = self.valid_item_names
        elif prefix == "test":
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names

        for item_name in item_names:
            yield self.items[item_name]

    def process(self):
        self.load_meta_data()
        os.makedirs(hparams["binary_data_dir"], exist_ok=True)
        for fn in ["phone_set.json", "word_set.json", "spk_map.json"]:
            remove_file(f"{hparams['binary_data_dir']}/{fn}")
            copy_file(
                f"{hparams['processed_data_dir']}/{fn}",
                f"{hparams['binary_data_dir']}/{fn}",
            )
        self.process_data("valid")
        self.process_data("test")
        self.process_data("train")

    def process_data(self, prefix):
        data_dir = hparams["binary_data_dir"]
        builder = IndexedDatasetBuilder(f"{data_dir}/{prefix}")
        meta_data = list(self.meta_data(prefix))
        process_item = partial(
            self.process_item, binarization_args=self.binarization_args
        )
        ph_lengths = []
        mel_lengths = []
        total_sec = 0
        items = []
        args = [{"item": item} for item in meta_data]
        for item_id, item in multiprocess_run_tqdm(
            process_item, args, desc="Processing data"
        ):
            if item is not None:
                items.append(item)

        if self.binarization_args["with_spk_embed"]:
            args = [{"wav": item["wav"]} for item in items]
            for item_id, spk_embed in multiprocess_run_tqdm(
                self.get_spk_embed,
                args,
                init_ctx_func=lambda wid: {"voice_encoder": VoiceEncoder().cuda()},
                num_workers=4,
                desc="Extracting spk embed",
            ):
                items[item_id]["spk_embed"] = spk_embed

        for item in items:
            if not self.binarization_args["with_wav"] and "wav" in item:
                del item["wav"]

            builder.add_item(item)
            mel_lengths.append(item["len"])
            assert item["len"] > 0, (item["item_name"], item["txt"], item["mel2ph"])
            if "ph_len" in item:
                ph_lengths.append(item["ph_len"])
            total_sec += item["sec"]
        builder.finalize()
        np.save(f"{data_dir}/{prefix}_lengths.npy", mel_lengths)
        if len(ph_lengths) > 0:
            np.save(f"{data_dir}/{prefix}_ph_lengths.npy", ph_lengths)
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    @classmethod
    def process_item(cls, item, binarization_args):
        item["ph_len"] = len(item["ph_token"])
        wav_fn = item["wav_fn"]
        cls.process_audio(wav_fn, item, binarization_args)
        return item

    @classmethod
    def process_audio(cls, wav_fn, res, binarization_args):
        wav2spec_dict = wav2spec(
            wav_fn,
            fft_size=hparams["fft_size"],
            hop_size=hparams["hop_size"],
            win_length=hparams["win_size"],
            num_mels=hparams["audio_num_mel_bins"],
            fmin=hparams["fmin"],
            fmax=hparams["fmax"],
            sample_rate=hparams["audio_sample_rate"],
            loud_norm=hparams["loud_norm"],
            trim_long_sil=True,
        )
        mel = wav2spec_dict["mel"]
        wav = wav2spec_dict["wav"].astype(np.float16)
        if binarization_args["with_linear"]:
            res["linear"] = wav2spec_dict["linear"]
        res.update(
            {
                "mel": mel,
                "wav": wav,
                "sec": len(wav) / hparams["audio_sample_rate"],
                "len": mel.shape[0],
            }
        )

    @staticmethod
    def get_spk_embed(wav, ctx):
        return ctx["voice_encoder"].embed_utterance(wav.astype(float))

    @property
    def num_workers(self):
        return int(os.getenv("N_PROC", hparams.get("N_PROC", os.cpu_count())))
