import os
import joblib
from itertools import groupby
from tqdm import tqdm
import torch
from torch.nn import functional as F
import torchaudio
from transformers import HubertModel

from models.evc.durflex.utils import LengthRegulator
from tasks.evc.dataset_utils import BaseSpeechDataset
from utils.commons.hparams import set_hparams
from utils.commons.hparams import hparams
from utils.audio.vad import trim_long_silences


def dedup_seq(seq):
    vals, counts = zip(*[(k.item(), sum(1 for _ in g)) for k, g in groupby(seq)])
    return vals, counts


def clustering(feats):
    _, D = feats.shape
    pred = kmeans_model.predict(feats.reshape(-1, D))
    return pred


if __name__ == "__main__":
    set_hparams()
    data_dir = hparams["binary_data_dir"]
    dataset_cls = BaseSpeechDataset
    prefixs = ["train", "valid", "test"]
    unit_dir = os.path.join(hparams["processed_data_dir"], "units")

    kmeans_model = joblib.load(open(hparams["kmeans_model_path"], "rb"))
    kmeans_model.verbose = False
    model = HubertModel.from_pretrained(
        hparams["hubert_model"], output_hidden_states=True
    ).cuda()

    lr = LengthRegulator()
    os.makedirs(unit_dir, exist_ok=True)

    for prefix in prefixs:
        items = []
        dataset = dataset_cls(prefix=prefix, shuffle=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collater,
            batch_size=1,
            num_workers=1,
            pin_memory=False,
        )
        for idx, i in tqdm(enumerate(dataloader)):
            basename = i["item_name"][0]
            spk = basename.split("_")[0]

            wav_fn = i["wav_fn"][0]
            wav, _, _ = trim_long_silences(wav_fn, hparams["audio_sample_rate"])
            wav = torch.from_numpy(wav).unsqueeze(0)

            with torch.no_grad():
                wav = F.pad(wav.cuda(), (40, 40), "reflect")
                output = model(wav)
                hidden = output.hidden_states[-1].cpu().squeeze(0).numpy()

            units = clustering(hidden)
            units = torch.IntTensor(units)
            val, count = dedup_seq(units)
            mel2unit = lr(torch.IntTensor(count).unsqueeze(0))

            data = {
                "features": hidden,
                "units": list(val),
                "units_frame": units,
                "count": list(count),
                "mel2unit": mel2unit.squeeze(0),
            }
            save_dir = os.path.join(
                unit_dir,
                spk,
            )
            os.makedirs(save_dir, exist_ok=True)
            torch.save(data, os.path.join(save_dir, basename + ".pt"))
