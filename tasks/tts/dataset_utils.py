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
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(shuffle)
        from utils.commons.hparams import hparams

        self.data_dir = hparams["binary_data_dir"] if data_dir is None else data_dir
        self.prefix = prefix
        self.hparams = hparams
        self.indexed_ds = None
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
        max_frames = hparams["max_frames"]
        spec = torch.Tensor(item["mel"])[:max_frames]
        max_frames = (
            spec.shape[0] // hparams["frames_multiple"] * hparams["frames_multiple"]
        )
        spec = spec[:max_frames]
        ph_token = torch.LongTensor(item["ph_token"][: hparams["max_input_tokens"]])
        sample = {
            "id": index,
            "item_name": item["item_name"],
            "text": item["txt"],
            "txt_token": ph_token,
            "mel": spec,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
        }
        if hparams["use_spk_embed"]:
            sample["spk_embed"] = torch.Tensor(item["spk_embed"])
        if hparams["use_spk_id"]:
            sample["spk_id"] = int(item["spk_id"])
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        id = torch.LongTensor([s["id"] for s in samples])
        item_names = [s["item_name"] for s in samples]
        text = [s["text"] for s in samples]
        txt_tokens = collate_1d_or_2d([s["txt_token"] for s in samples], 0)
        mels = collate_1d_or_2d([s["mel"] for s in samples], 0.0)
        txt_lengths = torch.LongTensor([s["txt_token"].numel() for s in samples])
        mel_lengths = torch.LongTensor([s["mel"].shape[0] for s in samples])

        batch = {
            "id": id,
            "item_name": item_names,
            "nsamples": len(samples),
            "text": text,
            "txt_tokens": txt_tokens,
            "txt_lengths": txt_lengths,
            "mels": mels,
            "mel_lengths": mel_lengths,
        }

        if hparams["use_spk_embed"]:
            spk_embed = torch.stack([s["spk_embed"] for s in samples])
            batch["spk_embed"] = spk_embed
        if hparams["use_spk_id"]:
            spk_ids = torch.LongTensor([s["spk_id"] for s in samples])
            batch["spk_ids"] = spk_ids
        return batch
