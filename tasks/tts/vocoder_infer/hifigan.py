import torch
from modules.vocoder.hifigan.hifigan import SynthesizerTrn
from tasks.tts.vocoder_infer.base_vocoder import register_vocoder, BaseVocoder
from utils.commons.hparams import hparams
from utils.commons.meters import Timer
import json

total_time = 0


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


@register_vocoder("HifiGAN")
class HifiGAN(BaseVocoder):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open("checkpoints/hifigan_16k/config.json", "r") as f:
            data = f.read()
        config = json.loads(data)
        hparams = HParams(**config)
        self.model = SynthesizerTrn(
            hparams.data.n_mel_channels,
            hparams.train.segment_size // hparams.data.hop_length,
            **hparams.model,
            rand=self.device
        )
        checkpoint_dict = torch.load(
            "checkpoints/hifigan_16k/G_2930000.pth", map_location=self.device
        )
        self.model.load_state_dict(checkpoint_dict["model"])
        self.model.to(self.device)
        self.model.eval()

    def spec2wav(self, mel, **kwargs):
        device = self.device
        with torch.no_grad():
            c = torch.FloatTensor(mel).unsqueeze(0).to(device)
            c = c.transpose(2, 1)
            with Timer("hifigan", enable=hparams["profile_infer"]):
                y = self.model.infer(c)
        wav_out = y.squeeze().cpu().numpy()
        wav_out = 0.9 * (wav_out / wav_out.max())
        return wav_out
