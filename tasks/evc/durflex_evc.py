import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.evc.durflex import DurFlexEVC
from tasks.evc.speech_base import SpeechBaseTask
from utils.commons.hparams import hparams
from utils.commons.tensor_utils import tensors_to_scalars
from utils.nn.model_utils import print_arch, num_params


class DurFlexEVCTask(SpeechBaseTask):
    def __init__(self):
        super(DurFlexEVCTask, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def build_model(self):
        self.model = DurFlexEVC(hparams["n_units"], hparams)
        print_arch(self.model)
        for n, m in self.model.named_children():
            num_params(m, model_name=n)
        return self.model

    def forward(self, sample, infer=False, *args, **kwargs):
        spk_embed = sample.get("spk_embed")
        spk_id = sample.get("spk_ids")

        emotion_id = sample.get("emotion_ids")
        x = sample["hubert_features"]

        y = sample["mels"]  # [B, T_s, 80]
        y_lengths = sample["mel_lengths"]  # [B, T_s, 80]

        if not infer:
            mel2unit = sample["mel2unit"]  # [B, T_s]

            output = self.model(
                x,
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
                output["unit_pred"],
                output["unit_logits"],
                sample["unit_frames"].float().unsqueeze(1),
                output["unit_nonpadding"],
                losses,
            )
            self.add_dur_loss(output["dur"], losses=losses)
            losses["diff_loss"] = output["diff_loss"]

            if hparams["use_spk_encoder"]:
                self.add_emo_loss(output["emo_logits"], emotion_id, losses)
            return losses, output
        else:
            mel2unit = None
            output = self.model(
                x,
                spk_embed=spk_embed,
                spk_id=spk_id,
                emotion_id=emotion_id,
                tgt_emotion_id=emotion_id,
                infer=True,
                y=y,
                y_lengths=y_lengths,
            )
            return output

    def add_ce_loss(self, pred, logits, target, nonpadding, losses=None):
        _, L, N = logits.shape
        logits = torch.log(logits + 1e-9)
        unit_logits = logits.view(-1, N)
        targets = F.interpolate(
            target,
            size=L,
            mode="nearest",
        ).squeeze(1)
        unit_loss = self.ce_loss(unit_logits, targets.view(-1).long())
        unit_loss = unit_loss * nonpadding.view(-1)
        unit_loss = unit_loss.sum() / nonpadding.sum()

        unit_accuracy = pred == targets
        unit_accuracy = unit_accuracy.view(-1) * nonpadding.view(-1)
        unit_accuracy = torch.sum(unit_accuracy) / nonpadding.sum()
        losses["unit_loss"] = unit_loss * 0.1
        losses["unit_accuracy"] = unit_accuracy

    def add_emo_loss(self, logits, target, losses=None):
        emo_loss = self.cel(logits, target.long())  # (B * L_max,)로 변경
        emo_loss = emo_loss.sum() * hparams["lambda_grl"]
        pred = torch.argmax(logits, dim=-1)
        emo_accuracy = pred == target
        emo_accuracy = torch.mean(emo_accuracy.float())
        losses["emo_loss"] = emo_loss
        losses["emo_accuracy"] = emo_accuracy

    def add_dur_loss(self, l_length, losses=None):
        loss_dur = torch.sum(l_length)
        losses["pdur"] = loss_dur * hparams["lambda_ph_dur"]

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs["losses"] = {}
        outputs["nsamples"] = sample["nsamples"]

        if (
            self.global_step % hparams["valid_infer_interval"] == 0
            and batch_idx < hparams["num_valid_plots"]
        ):
            emo_dict = ["Neutral", "Angry", "Happy", "Sad", "Surprise"]
            emo_mels = []
            spk_embed = sample.get("spk_embed")
            spk_id = sample.get("spk_ids")
            src_emotion_id = sample.get("emotion_ids")
            x = sample["hubert_features"]
            y = sample["mels"]
            y_lengths = sample["mel_lengths"]
            f0s = None
            for idx, _ in enumerate(emo_dict):
                emotion_id = torch.LongTensor([idx]).cuda()
                output = self.model(
                    x,
                    spk_embed=spk_embed,
                    spk_id=spk_id,
                    emotion_id=src_emotion_id,
                    tgt_emotion_id=emotion_id,
                    infer=True,
                    y=y,
                    y_lengths=y_lengths,
                )
                mel_pred = output["mel_out"]
                emo_mels.append(mel_pred)
            gt_mel = sample["mels"]
            self.save_valid_result(sample, batch_idx, [gt_mel, emo_mels], f0s=f0s)

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
        self.plot_mel(
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

    def test_step(self, sample, batch_idx):
        sr = hparams["audio_sample_rate"]
        x = sample["hubert_features"]
        y = sample["mels"]  # [B, T_s, 80]
        y_lengths = sample["mel_lengths"]  # [B, T_s, 80]
        spk_id = sample.get("spk_ids")
        emotion_id = sample.get("emotion_ids")

        outputs = self.model(
            x,
            spk_id=spk_id,
            emotion_id=emotion_id,
            tgt_emotion_id=emotion_id,
            infer=True,
            y=y,
            y_lengths=y_lengths,
        )
        item_name = sample["item_name"][0]
        mel_gt = sample["mels"][0].cpu().numpy()
        mel_pred = outputs["mel_out"][0].cpu().numpy()

        base_fn = item_name
        gen_dir = self.gen_dir
        wav_pred = self.vocoder.spec2wav(mel_pred)
        self.saving_result_pool.add_job(
            self.save_result,
            args=[
                wav_pred,
                mel_pred,
                base_fn,
                gen_dir,
                None,
                None,
                None,
                sr,
            ],
        )
        print(f"Pred_shape: {mel_pred.shape}, gt_shape: {mel_gt.shape}")
        return {
            "item_name": item_name,
            "wav_fn_pred": base_fn,
        }
