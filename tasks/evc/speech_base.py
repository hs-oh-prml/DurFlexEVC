import filecmp
import os
import traceback
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from tqdm import tqdm
import utils
from tasks.evc.evc_utils import VocoderInfer, parse_dataset_configs
from utils.audio.align import mel2token_to_dur
from utils.audio.io import save_wav
from utils.audio.pitch_extractors import extract_pitch_simple
from utils.commons.base_task import BaseTask
from utils.commons.dataset_utils import data_loader, BaseConcatDataset
from utils.commons.hparams import set_hparams
from utils.commons.multiprocess_utils import MultiprocessManager
from utils.commons.tensor_utils import tensors_to_scalars
from utils.nn.schedulers import (
    WarmupSchedule
)
from utils.plot.plot import spec_to_figure
from utils.audio.pitch.utils import denorm_f0
import matplotlib.pyplot as plt
import importlib

class SpeechBaseTask(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hparams = self.hparams = set_hparams()
        dataset_cls = hparams["dataset_cls"]
        dataset_pkg = ".".join(dataset_cls.split(".")[:-1])
        dataset_cls_name = dataset_cls.split(".")[-1]
        self.dataset_cls = getattr(
            importlib.import_module(dataset_pkg), dataset_cls_name
        )
        (
            self.max_tokens,
            self.max_sentences,
            self.max_valid_tokens,
            self.max_valid_sentences,
        ) = parse_dataset_configs()
        self.saving_result_pool = None
        self.saving_results_futures = None

    ##########################
    # datasets
    ##########################
    @data_loader
    def train_dataloader(self):
        if self.hparams["train_sets"] != "":
            train_sets = self.hparams["train_sets"].split("|")
            # check if all train_sets have the same spk map and dictionary
            binary_data_dir = self.hparams["binary_data_dir"]
            file_to_cmp = ["phone_set.json"]
            if os.path.exists(f"{binary_data_dir}/word_set.json"):
                file_to_cmp.append("word_set.json")
            if self.hparams["use_spk_id"]:
                file_to_cmp.append("spk_map.json")
            for f in file_to_cmp:
                for ds_name in train_sets:
                    base_file = os.path.join(binary_data_dir, f)
                    ds_file = os.path.join(ds_name, f)
                    assert filecmp.cmp(
                        base_file, ds_file
                    ), f"{f} in {ds_name} is not same with that in {binary_data_dir}."
            train_dataset = BaseConcatDataset(
                [
                    self.dataset_cls(prefix="train", shuffle=True, data_dir=ds_name)
                    for ds_name in train_sets
                ]
            )
        else:
            train_dataset = self.dataset_cls(
                prefix=self.hparams["train_set_name"], shuffle=True, train=True
            )
        return self.build_dataloader(
            train_dataset,
            True,
            self.max_tokens,
            self.max_sentences,
            endless=self.hparams["endless_ds"],
        )

    @data_loader
    def val_dataloader(self):
        valid_dataset = self.dataset_cls(
            prefix=self.hparams["valid_set_name"], shuffle=False, train=False
        )
        return self.build_dataloader(
            valid_dataset,
            False,
            self.max_valid_tokens,
            self.max_valid_sentences,
            batch_by_size=False,
        )

    @data_loader
    def test_dataloader(self):
        test_dataset = self.dataset_cls(
            prefix=self.hparams["test_set_name"], shuffle=False, train=False
        )
        self.test_dl = self.build_dataloader(
            test_dataset,
            False,
            self.max_valid_tokens,
            self.max_valid_sentences,
            batch_by_size=False,
        )
        return self.test_dl

    def build_dataloader(
        self,
        dataset,
        shuffle,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=-1,
        endless=False,
        batch_by_size=True,
    ):
        devices_cnt = torch.cuda.device_count()
        if devices_cnt == 0:
            devices_cnt = 1
        if required_batch_size_multiple == -1:
            required_batch_size_multiple = devices_cnt

        def shuffle_batches(batches):
            np.random.shuffle(batches)
            return batches

        if max_tokens is not None:
            max_tokens *= devices_cnt
        if max_sentences is not None:
            max_sentences *= devices_cnt
        indices = dataset.ordered_indices()
        if batch_by_size:
            batch_sampler = utils.commons.dataset_utils.batch_by_size(
                indices,
                dataset.num_tokens,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
        else:
            batch_sampler = []
            for i in range(0, len(indices), max_sentences):
                batch_sampler.append(indices[i : i + max_sentences])

        if shuffle:
            batches = shuffle_batches(list(batch_sampler))
            if endless:
                batches = [
                    b for _ in range(1000) for b in shuffle_batches(list(batch_sampler))
                ]
        else:
            batches = batch_sampler
            if endless:
                batches = [b for _ in range(1000) for b in batches]
        num_workers = dataset.num_workers
        if self.trainer.use_ddp:
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
            batches = [
                x[rank::num_replicas] for x in batches if len(x) % num_replicas == 0
            ]
        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collater,
            batch_sampler=batches,
            num_workers=num_workers,
            pin_memory=False,
        )

    ##########################
    # scheduler and optimizer
    ##########################
    def build_model(self):
        NotImplemented

    def build_scheduler(self, optimizer):
        return WarmupSchedule(
            optimizer, self.hparams["lr"], self.hparams["warmup_updates"]
        )

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.hparams["lr"],
            betas=(
                self.hparams["optimizer_adam_beta1"],
                self.hparams["optimizer_adam_beta2"],
            ),
            weight_decay=self.hparams["weight_decay"],
        )

        return optimizer

    ##########################
    # training and validation
    ##########################
    def _training_step(self, sample, batch_idx, _):
        loss_output, _ = self(sample)
        total_loss = sum(
            [
                v
                for v in loss_output.values()
                if isinstance(v, torch.Tensor) and v.requires_grad
            ]
        )       
        return total_loss, loss_output

    def forward(self, sample, infer=False):
        """

        :param sample: a batch of data
        :param infer: bool, run in infer mode
        :return:
            if not infer:
                return losses, model_out
            if infer:
                return model_out
        """
        raise NotImplementedError

    def validation_start(self):
        self.vocoder = VocoderInfer(self.hparams)

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs["losses"] = {}
        outputs["losses"], _ = self(sample)
        outputs["total_loss"] = sum(outputs["losses"].values())
        outputs["nsamples"] = sample["nsamples"]
        outputs = tensors_to_scalars(outputs)
        if (
            self.global_step % self.hparams["valid_infer_interval"] == 0
            and batch_idx < self.hparams["num_valid_plots"]
        ):
            model_out = self(sample, infer=True)
            self.save_valid_result(sample, batch_idx, model_out)
        return outputs

    def validation_end(self, outputs):
        self.vocoder = None
        return super(SpeechBaseTask, self).validation_end(outputs)

    ##########################
    # plot
    ##########################

    def plot_mel(self, batch_idx, specs, name=None, title="", f0s=None, dur_info=None):
        vmin = self.hparams["mel_vmin"]
        vmax = self.hparams["mel_vmax"]
        l = []
        for i in specs:
            l.append(i.shape[0])
        max_l = max(l)
        if f0s is not None:
            f0_dict = {}
        else:
            f0_dict = None

        for i in range(len(specs)):
            p = max_l - specs[i].shape[0]
            specs[i] = F.pad(specs[i], (0, 0, 0, p), mode="constant").data
            if f0s is not None:
                f0_dict[i] = F.pad(f0s[i].squeeze(0), (0, p), mode="constant").data

        spec_out = []
        for i in specs:
            spec_out.append(i.detach().cpu().numpy())
        spec_out = np.concatenate(spec_out, -1)

        name = f"mel_val_{batch_idx}" if name is None else name
        h = 3 * len(specs)
        self.logger.add_figure(
            name,
            spec_to_figure(
                spec_out,
                vmin,
                vmax,
                title=title,
                f0s=f0_dict,
                dur_info=dur_info,
                figsize=(12, h),
            ),
            self.global_step,
        )

    ##########################
    # testing
    ##########################
    def save_valid_result(self, sample, batch_idx, model_out):
        sr = self.hparams["audio_sample_rate"]
        f0s = None
        mel_out = model_out["mel_out"]

        if self.hparams["plot_f0"]:
            f0_gt = denorm_f0(sample["f0"][0].cpu(), sample["uv"][0].cpu())
            f0_pred = model_out["f0_denorm"]
            f0s = {"GT": f0_gt, "Pred": f0_pred}
        self.plot_mel(
            batch_idx,
            [sample["mels"][0], mel_out[0]],
            name=f"mel",
            title=f"mel",
            f0s=f0s,
        )

        wav_pred = self.vocoder.spec2wav(mel_out[0].cpu())
        self.logger.add_audio(f"wav_pred_{batch_idx}", wav_pred, self.global_step, sr)

        # gt wav
        if self.global_step <= self.hparams["valid_infer_interval"]:
            mel_gt = sample["mels"][0].cpu()
            wav_gt = self.vocoder.spec2wav(mel_gt)
            self.logger.add_audio(f"wav_gt_{batch_idx}", wav_gt, self.global_step, sr)

    @staticmethod
    def save_result(
        wav_out,
        mel,
        base_fn,
        gen_dir,
        str_phs=None,
        mel2ph=None,
        alignment=None,
        audio_sample_rate=22050,
        out_wav_norm=True,
        mel_vmin=-6,
        mel_vmax=1.5,
        save_mel_npy=False,
    ):
        save_wav(
            wav_out,
            f"{gen_dir}/wavs/{base_fn}.wav",
            audio_sample_rate,
            norm=out_wav_norm,
        )
        fig = plt.figure(figsize=(12, 6))
        plt.imshow(mel.T, origin="lower")
        try:
            f0 = extract_pitch_simple(wav_out)
            f0 = f0 / 10 * (f0 > 0)
            plt.plot(f0, c="white", linewidth=1, alpha=0.6)
            if mel2ph is not None and str_phs is not None:
                decoded_txt = str_phs.split(" ")
                dur = mel2token_to_dur(
                    torch.LongTensor(mel2ph)[None, :], len(decoded_txt)
                )[0].numpy()
                dur = [0] + list(np.cumsum(dur))
                for i in range(len(dur) - 1):
                    shift = (i % 20) + 1
                    plt.text(dur[i], shift, decoded_txt[i])
                    plt.hlines(
                        shift,
                        dur[i],
                        dur[i + 1],
                        colors="b" if decoded_txt[i] != "|" else "black",
                    )
                    plt.vlines(
                        dur[i],
                        0,
                        5,
                        colors="b" if decoded_txt[i] != "|" else "black",
                        alpha=1,
                        linewidth=1,
                    )
            plt.yticks(np.arange(0, mel.shape[1] + 1, 10))
            plt.tight_layout()
            plt.savefig(
                f"{gen_dir}/plot/{base_fn}.png", bbox_inches="tight", format="png"
            )
            plt.close(fig)
            if save_mel_npy:
                np.save(f"{gen_dir}/mel_npy/{base_fn}", mel)
            if alignment is not None:
                # fig, ax = plt.subplots(figsize=(12, 16))
                fig, ax = plt.subplots()
                im = ax.imshow(
                    alignment, aspect="auto", origin="lower", interpolation="none"
                )
                decoded_txt = str_phs.split(" ")
                ax.set_yticks(np.arange(len(decoded_txt)))
                ax.set_yticklabels(list(decoded_txt), fontsize=6)
                fig.colorbar(im, ax=ax)
                fig.savefig(f"{gen_dir}/attn_plot/{base_fn}_attn.png", format="png")
                
                plt.close(fig)
        except Exception:
            traceback.print_exc()
        return None

    def test_start(self):
        self.saving_result_pool = MultiprocessManager(
            int(os.getenv("N_PROC", os.cpu_count()))
        )
        self.saving_results_futures = []
        self.gen_dir = os.path.join(
            self.hparams["work_dir"],
            f'generated_{self.trainer.global_step}_{self.hparams["gen_dir_name"]}',
        )
        self.vocoder = VocoderInfer(self.hparams)
        os.makedirs(self.gen_dir, exist_ok=True)
        os.makedirs(f"{self.gen_dir}/wavs", exist_ok=True)
        os.makedirs(f"{self.gen_dir}/plot", exist_ok=True)
        if self.hparams["save_mel_npy"]:
            os.makedirs(f"{self.gen_dir}/mel_npy", exist_ok=True)

    def test_step(self, sample, batch_idx):
        NotImplemented

    def test_end(self, outputs):
        pd.DataFrame(outputs).to_csv(f"{self.gen_dir}/meta.csv")
        for _1, _2 in tqdm(
            self.saving_result_pool.get_results(), total=len(self.saving_result_pool)
        ):
            pass
        return {}
