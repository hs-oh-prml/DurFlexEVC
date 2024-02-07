import os
import torch
from utils.commons.dataset_utils import collate_1d_or_2d
from tasks.tts.dataset_utils import BaseSpeechDataset


class DurFlexDataset(BaseSpeechDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(prefix, shuffle, items, data_dir)
        self.pitch_type = self.hparams.get("pitch_type")
        self.emo_dict = {"Neutral": 0, "Angry": 1, "Happy": 2, "Sad": 3, "Surprise": 4}

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        item = self._get_item(index)
        hparams = self.hparams
        item_name = item["item_name"]
        spk_name = item_name.split("_")[0]
        unit_item = torch.load(
            os.path.join(
                "{}/unit200/{}/{}.pt".format(
                    hparams["processed_data_dir"], spk_name, item_name
                )
            )
        )
        mel2unit = unit_item["mel2unit"]
        sample["unit"] = torch.IntTensor(unit_item["units"])
        sample["unit_frame"] = torch.IntTensor(unit_item["units_frame"])
        sample["hubert_feature"] = torch.FloatTensor(unit_item["features"])
        sample["dur_unit"] = torch.IntTensor(unit_item["count"])
        sample["mel2unit"] = mel2unit
        sample["unit_l"] = mel2unit[-1]

        if hparams["use_spk_embed"]:
            sample["spk_embed"] = torch.Tensor(item["spk_embed"])
        if hparams["use_spk_id"]:
            sample["spk_id"] = int(item["spk_id"])
        sample["emotion_id"] = int(self.emo_dict[item["emo"]])

        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        hparams = self.hparams
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

        if hparams["use_spk_embed"]:
            spk_embed = torch.stack([s["spk_embed"] for s in samples])
            batch["spk_embed"] = spk_embed
        if hparams["use_spk_id"]:
            spk_ids = torch.LongTensor([s["spk_id"] for s in samples])
            batch["spk_ids"] = spk_ids
        emos = torch.LongTensor([s["emotion_id"] for s in samples])
        batch["emotion_ids"] = emos
        return batch
