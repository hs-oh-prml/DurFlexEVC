import os
import torch.optim
import torch.utils.data
import numpy as np
import torch
import torch.optim
import torch.utils.data
import torch.distributions
from utils.commons.dataset_utils import BaseDataset, collate_1d_or_2d
from utils.commons.indexed_datasets import IndexedDataset


class BaseSpeechDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None, train=False):
        super().__init__(shuffle)
        from utils.commons.hparams import hparams

        self.data_dir = hparams["binary_data_dir"] if data_dir is None else data_dir
        self.prefix = prefix
        self.hparams = hparams
        self.indexed_ds = None
        self.train = train
        if items is not None:
            self.indexed_ds = items
            self.sizes = [1] * len(items)
            self.avail_idxs = list(range(len(self.sizes)))
        else:
            self.sizes = np.load(f"{self.data_dir}/{self.prefix}_lengths.npy")
            if prefix == "test" and len(hparams["test_ids"]) > 0:
                self.avail_idxs = hparams["test_ids"]
            else:
                self.avail_idxs = list(range(len(self.sizes)))
            if prefix == "train" and hparams["min_frames"] > 0:
                self.avail_idxs = [
                    x for x in self.avail_idxs if self.sizes[x] >= hparams["min_frames"]
                ]
            self.sizes = [self.sizes[i] for i in self.avail_idxs]

    def _get_item(self, index):
        if hasattr(self, "avail_idxs") and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f"{self.data_dir}/{self.prefix}")
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        assert len(item["mel"]) == self.sizes[index], (
            len(item["mel"]),
            self.sizes[index],
        )
        wav_fn = item["wav_fn"]
        max_frames = hparams["max_frames"]
        spec = torch.Tensor(item["mel"])[:max_frames]
        max_frames = (
            spec.shape[0] // hparams["frames_multiple"] * hparams["frames_multiple"]
        )
        spec = spec[:max_frames]

        sample = {
            "id": index,
            "wav_fn": wav_fn,
            "item_name": item["item_name"],
            "mel": spec,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
            "spk_id": int(item["spk_id"]),
        }
        return sample

    def collater(self, samples):

        id = torch.LongTensor([s["id"] for s in samples])
        item_names = [s["item_name"] for s in samples]
        wav_fns = [s["wav_fn"] for s in samples]
        mels = collate_1d_or_2d([s["mel"] for s in samples], 0.0)
        mel_lengths = torch.LongTensor([s["mel"].shape[0] for s in samples])
        spk_ids = torch.LongTensor([s["spk_id"] for s in samples])

        batch = {
            "id": id,
            "wav_fn": wav_fns,
            "item_name": item_names,
            "nsamples": len(samples),
            "mels": mels,
            "mel_lengths": mel_lengths,
            "spk_ids": spk_ids,
        }

        return batch


class DurFlexDataset(BaseSpeechDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None, train=False):
        super().__init__(prefix, shuffle, items, data_dir, train)
        self.emo_dict = {"Neutral": 0, "Angry": 1, "Happy": 2, "Sad": 3, "Surprise": 4}

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        item = self._get_item(index)
        hparams = self.hparams

        item_name = item["item_name"]
        spk_name = item_name.split("_")[0]

        unit_item = torch.load(
            os.path.join(
                "{}/units/{}/{}.pt".format(
                    hparams["processed_data_dir"], spk_name, item_name
                )
            )
        )

        sample["unit"] = torch.IntTensor(unit_item["units"])
        sample["unit_frame"] = torch.IntTensor(unit_item["units_frame"])
        sample["hubert_feature"] = torch.FloatTensor(unit_item["features"])
        mel2unit = unit_item["mel2unit"]
        sample["unit_l"] = mel2unit[-1]
        sample["dur_unit"] = torch.IntTensor(unit_item["count"])
        sample["mel2unit"] = mel2unit
        sample["spk_id"] = int(item["spk_id"])
        sample["emotion_id"] = int(self.emo_dict[item["emo"]])
        return sample

    def collater(self, samples):
        batch = super().collater(samples)

        units = collate_1d_or_2d([s["unit"] for s in samples], 0.0)
        unit_frames = collate_1d_or_2d([s["unit_frame"] for s in samples], 0.0)
        dur_unit = collate_1d_or_2d([s["dur_unit"] for s in samples], 0.0)
        unit_l = torch.LongTensor([s["unit_l"] for s in samples])
        mel2unit = collate_1d_or_2d([s["mel2unit"] for s in samples], 0.0)
        hubert_features = collate_1d_or_2d([s["hubert_feature"] for s in samples], 0.0)
        batch["units"] = units
        batch["unit_frames"] = unit_frames
        batch["unit_l"] = unit_l
        batch["dur_unit"] = dur_unit
        batch["mel2unit"] = mel2unit
        batch["hubert_features"] = hubert_features
        hubert_lengths = torch.LongTensor(
            [s["hubert_feature"].shape[0] for s in samples]
        )
        batch["hubert_lengths"] = hubert_lengths

        spk_ids = torch.LongTensor([s["spk_id"] for s in samples])
        batch["spk_ids"] = spk_ids
        emos = torch.LongTensor([s["emotion_id"] for s in samples])
        batch["emotion_ids"] = emos
        return batch
