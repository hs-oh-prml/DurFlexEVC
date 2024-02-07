import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.durflex_evc.durflex import DurFlex
from tasks.durflex_evc.dataset_utils import DurFlexDataset
from tasks.tts.speech_base import SpeechBaseTask
from utils.audio.align import mel2token_to_dur
from utils.commons.hparams import hparams
from utils.commons.tensor_utils import tensors_to_scalars
from utils.plot.plot import spec_to_figure


class DurFlexTask(SpeechBaseTask):
    def __init__(self):
        super(DurFlexTask, self).__init__()
        self.dataset_cls = DurFlexDataset
        self.cel = nn.CrossEntropyLoss(reduction="none")

    def build_tts_model(self):
        self.model = DurFlex(hparams["n_units"], hparams)

    def run_model(self, sample, infer=False, *args, **kwargs):
        spk_embed = sample.get("spk_embed")
        spk_id = sample.get("spk_ids")
        emotion_id = sample.get("emotion_ids")

        src_feat = sample["hubert_features"]
        y = sample["mels"]  # [B, T_s, 80]
        y_lengths = sample["mel_lengths"]  # [B, T_s, 80]

        if not infer:
            mel2unit = sample["mel2unit"]  # [B, T_s]

            output = self.model(
                features=src_feat,
                mel2unit=mel2unit,
                spk_embed=spk_embed,
                spk_id=spk_id,
                emotion_id=emotion_id,
                tgt_emotion_id=emotion_id,
                y=y,
                y_lengths=y_lengths,
                infer=False,
            )
            losses = {}

            self.add_ce_loss(
                output["unit_pred_frame"],
                output["unit_logits"],
                sample["unit_frames"].float().unsqueeze(1),
                output["unit_nonpadding"],
                losses,
            )
            self.add_dur_sto_loss(output["dur"], losses=losses)
            losses["diff_loss"] = output["diff_loss"]

            return losses, output
        else:
            mel2unit = None
            mel_lengths = sample["mel_lengths"]
            output = self.model(
                features=src_feat,
                spk_embed=spk_embed,
                spk_id=spk_id,
                emotion_id=emotion_id,
                tgt_emotion_id=emotion_id,
                mel_lengths=mel_lengths,
                infer=True,
                y=y,
                y_lengths=y_lengths,
            )
            return output

    def add_ce_loss(self, pred, logits, target, nonpadding, losses=None):
        B, L, N = logits.shape
        if hparams["use_log"]:
            logits = torch.log(logits + 1e-9)
        unit_logits = logits.view(-1, N)  # (B * L_max, N)로 변경
        targets = F.interpolate(
            target,
            size=L,
            mode="nearest",
        ).squeeze(1)
        unit_loss = self.cel(
            unit_logits, targets.view(-1).long()
        )  # (B * L_max,)로 변경
        unit_loss = unit_loss * nonpadding.view(-1)
        unit_loss = unit_loss.sum() / nonpadding.sum()
        unit_accuracy = pred == targets
        unit_accuracy = unit_accuracy.view(-1) * nonpadding.view(-1)
        unit_accuracy = torch.sum(unit_accuracy) / nonpadding.sum()
        losses["unit_loss"] = unit_loss
        losses["unit_accuracy"] = unit_accuracy

    def add_dur_sto_loss(self, l_length, losses=None):
        loss_dur = torch.sum(l_length)
        losses["pdur"] = loss_dur * hparams["lambda_ph_dur"]

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs["losses"] = {}
        outputs["losses"], model_out = self.run_model(sample)
        outputs["nsamples"] = sample["nsamples"]

        sr = hparams["audio_sample_rate"]

        if (
            self.global_step % hparams["valid_infer_interval"] == 0
            and batch_idx < hparams["num_valid_plots"]
            and self.global_step > 0
        ):
            emo_dict = ["Neutral", "Angry", "Happy", "Sad", "Surprise"]
            mel_lengths = sample["mel_lengths"]
            emo_mels = []
            spk_embed = sample.get("spk_embed")
            spk_id = sample.get("spk_ids")

            src_emotion_id = sample.get("emotion_ids")
            src_feat = sample["hubert_features"]

            y = sample["mels"]  # [B, T_s, 80]
            y_lengths = sample["mel_lengths"]  # [B, T_s, 80]

            for idx, _ in enumerate(emo_dict):
                emotion_id = torch.LongTensor([idx]).cuda()
                output = self.model(
                    features=src_feat,
                    spk_embed=spk_embed,
                    spk_id=spk_id,
                    emotion_id=src_emotion_id,
                    tgt_emotion_id=emotion_id,
                    mel_lengths=mel_lengths,
                    infer=True,
                    y=y,
                    y_lengths=y_lengths,
                )
                mel_pred = output["mel_out"]
                emo_mels.append(mel_pred)

                if sample.get("emotion_ids") == emotion_id:
                    dur_info = self.get_plot_dur_info(sample, output)
                    del dur_info["dur_pred"]
                    wav_pred = self.vocoder.spec2wav(output["mel_out"][0].cpu())
                    self.logger.add_audio(
                        f"wav_pdur_{batch_idx}", wav_pred, self.global_step, sr
                    )

            gt_mel = sample["mels"]
            self.save_valid_result(sample, batch_idx, [gt_mel, emo_mels])

        outputs = tensors_to_scalars(outputs)
        return outputs

    def save_valid_result(self, sample, batch_idx, model_out, f0s):
        sr = hparams["audio_sample_rate"]
        gt = model_out[0]
        pred = model_out[1]

        wav_title_gt = "Wav_gt_{}".format(batch_idx)
        wav_gt = self.vocoder.spec2wav(gt[0].cpu())
        self.logger.add_audio(wav_title_gt, wav_gt, self.global_step, sr)

        emo_dict = ["Neutral", "Angry", "Happy", "Sad", "Surprise"]
        for idx, i in enumerate(emo_dict):
            wav_title_pred = "wav_pred_{}/{}".format(batch_idx, i)
            wav_pred = self.vocoder.spec2wav(pred[idx][0].cpu())
            self.logger.add_audio(wav_title_pred, wav_pred, self.global_step, sr)

        mel_title = "mel_{}".format(batch_idx)
        self.plot_mel2(
            batch_idx,
            [
                gt[0],
                pred[0][0],
                pred[1][0],
                pred[2][0],
                pred[3][0],
                pred[4][0],
            ],
            title=mel_title,
            f0s=f0s,
        )

    def plot_mel2(self, batch_idx, specs, name=None, title="", f0s=None, dur_info=None):
        vmin = hparams["mel_vmin"]
        vmax = hparams["mel_vmax"]
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

    def get_plot_dur_info(self, sample, model_out):
        T_txt = sample["units"].shape[1]
        dur_gt = mel2token_to_dur(sample["mel2unit"], T_txt)[0]
        dur_pred = model_out["dur"][0] if "dur" in model_out else dur_gt
        txt = sample["units"][0].cpu().numpy().tolist()
        txt_pred = model_out["units"][0].cpu().numpy().tolist()
        return {
            "dur_gt": dur_gt,
            "dur_pred": dur_pred,
            "txt_gt": txt,
            "txt_pred": txt_pred,
        }

    def test_step(self, sample, batch_idx):
        assert (
            sample["txt_tokens"].shape[0] == 1
        ), "only support batch_size=1 in inference"

        if hparams["gen_dir_name"] == "recon":
            outputs = self.run_model(sample, infer=True)
            text = sample["text"][0]
            item_name = sample["item_name"][0]
            tokens = sample["txt_tokens"][0].cpu().numpy()
            mel_gt = sample["mels"][0].cpu().numpy()
            mel2unit_pred = None
            str_word = sample["units"][0].cpu().numpy().tolist()
            mel_pred = outputs["mel_out"][0].cpu().numpy()

            base_fn = item_name
            gen_dir = self.gen_dir
            wav_pred = self.vocoder.spec2wav(mel_pred)
            self.saving_result_pool.add_job(
                self.save_result,
                args=[wav_pred, mel_pred, base_fn, gen_dir, str_word, mel2unit_pred],
            )
            if hparams.get("save_attn", False):
                attn = outputs["attn"][0].cpu().numpy()
                np.save(f"{gen_dir}/attn/{item_name}.npy", attn)
            print(f"Pred_shape: {mel_pred.shape}, gt_shape: {mel_gt.shape}")
        else:
            emo_dict = ["Neutral", "Angry", "Happy", "Sad", "Surprise"]
            y = sample["mels"]  # [B, T_s, 80]
            y_lengths = sample["mel_lengths"]  # [B, T_s, 80]
            spk_embed = sample.get("spk_embed")
            spk_id = sample.get("spk_ids")
            src_emotion_id = sample.get("emotion_ids")

            src_feat = sample["hubert_features"]

            for i in range(5):
                emotion_id = torch.LongTensor([i]).to(sample["txt_tokens"].device)
                outputs = self.model(
                    features=src_feat,
                    spk_embed=spk_embed,
                    spk_id=spk_id,
                    emotion_id=src_emotion_id,
                    tgt_emotion_id=emotion_id,
                    infer=True,
                    y=y,
                    y_lengths=y_lengths,
                )
                text = sample["text"][0]
                item_name = sample["item_name"][0]
                tokens = sample["txt_tokens"][0].cpu().numpy()
                mel_gt = sample["mels"][0].cpu().numpy()
                mel2unit_pred = None
                str_word = sample["units"][0].cpu().numpy().tolist()
                mel_pred = outputs["mel_out"][0].cpu().numpy()

                base_fn = item_name + "_" + emo_dict[i]
                gen_dir = self.gen_dir
                wav_pred = self.vocoder.spec2wav(mel_pred)
                self.saving_result_pool.add_job(
                    self.save_result,
                    args=[
                        wav_pred,
                        mel_pred,
                        base_fn,
                        gen_dir,
                        str_word,
                        mel2unit_pred,
                    ],
                )
                if hparams.get("save_attn", False):
                    attn = outputs["attn"][0].cpu().numpy()
                    np.save(f"{gen_dir}/attn/{item_name}.npy", attn)
                print(f"Pred_shape: {mel_pred.shape}, gt_shape: {mel_gt.shape}")
        return {
            "item_name": item_name,
            "text": text,
            "ph_tokens": self.token_encoder.decode(tokens.tolist()),
            "wav_fn_pred": base_fn,
        }
