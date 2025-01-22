import importlib
import torch
from utils.commons.hparams import hparams, set_hparams


class VocoderInfer:
    def __init__(self, hparams):

        config_path = hparams["vocoder_config"]
        self.config = config = set_hparams(config_path, global_hparams=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pkg = ".".join(hparams["vocoder_cls"].split(".")[:-1])
        cls_name = hparams["vocoder_cls"].split(".")[-1]
        vocoder = getattr(importlib.import_module(pkg), cls_name)
        self.model = vocoder(config)

        checkpoint_dict = torch.load(
            hparams["vocoder_ckpt"], map_location=self.device, weights_only=True
        )

        self.model.load_state_dict(checkpoint_dict["generator"])
        self.model.to(self.device)
        self.model.eval()

    def spec2wav(self, mel, **kwargs):
        device = self.device
        with torch.no_grad():
            c = torch.FloatTensor(mel).unsqueeze(0).to(device)
            c = c.transpose(2, 1)
            y = self.model(c).view(-1)
        wav_out = y.cpu().numpy()
        return wav_out


def parse_dataset_configs():
    max_tokens = hparams["max_tokens"]
    max_sentences = hparams["max_sentences"]
    max_valid_tokens = hparams["max_valid_tokens"]
    if max_valid_tokens == -1:
        hparams["max_valid_tokens"] = max_valid_tokens = max_tokens
    max_valid_sentences = hparams["max_valid_sentences"]
    if max_valid_sentences == -1:
        hparams["max_valid_sentences"] = max_valid_sentences = max_sentences
    return max_tokens, max_sentences, max_valid_tokens, max_valid_sentences
