import torch
import torch.distributions
import torch.nn.functional as F
import torch.optim
import torch.utils.data

from modules.tts.fs import FastSpeech
from tasks.tts.dataset_utils import FastSpeechWordDataset
from tasks.tts.speech_base import SpeechBaseTask
from utils.audio.align import mel2token_to_dur
from utils.audio.pitch.utils import denorm_f0
from utils.commons.hparams import hparams
from modules.tts.discriminator import Discriminator
from utils.nn.model_utils import num_params
from utils.nn.schedulers import WarmupSchedule


class FastSpeechTask(SpeechBaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = FastSpeechWordDataset
        self.sil_ph = self.token_encoder.sil_phonemes()
        if hparams["adv"]:
            disc_win_num = hparams["disc_win_num"]
            h = hparams["mel_disc_hidden_size"]
            self.mel_disc = Discriminator(
                time_lengths=[32, 64, 128][:disc_win_num],
                freq_length=hparams["audio_num_mel_bins"],
                hidden_size=h,
                kernel=(3, 3),
            )
            self.disc_params = list(self.mel_disc.parameters())

    def build_tts_model(self):
        dict_size = len(self.token_encoder)
        self.model = FastSpeech(dict_size, hparams)

    def on_train_start(self):
        super().on_train_start()
        if hparams["adv"]:
            for n, m in self.mel_disc.named_children():
                num_params(m, model_name=f"disc.{n}")

    def run_model(self, sample, infer=False, *args, **kwargs):
        txt_tokens = sample["txt_tokens"]  # [B, T_t]
        spk_embed = sample.get("spk_embed")
        spk_id = sample.get("spk_ids")
        emotion_id = sample.get("emotion_ids")
        if not infer:
            target = sample["mels"]  # [B, T_s, 80]
            mel2ph = sample["mel2ph"]  # [B, T_s]
            f0 = sample.get("f0")
            uv = sample.get("uv")
            output = self.model(
                txt_tokens,
                mel2ph=mel2ph,
                spk_embed=spk_embed,
                spk_id=spk_id,
                emotion_id=emotion_id,
                f0=f0,
                uv=uv,
                infer=False,
            )
            losses = {}
            self.add_mel_loss(output["mel_out"], target, losses)
            self.add_dur_loss(output["dur"], mel2ph, txt_tokens, losses=losses)
            if hparams["use_pitch_embed"]:
                self.add_pitch_loss(output, sample, losses)
            return losses, output
        else:
            use_gt_dur = kwargs.get("infer_use_gt_dur", hparams["use_gt_dur"])
            use_gt_f0 = kwargs.get("infer_use_gt_f0", hparams["use_gt_f0"])
            mel2ph, uv, f0 = None, None, None
            if use_gt_dur:
                mel2ph = sample["mel2ph"]
            if use_gt_f0:
                f0 = sample["f0"]
                uv = sample["uv"]
            output = self.model(
                txt_tokens,
                mel2ph=mel2ph,
                spk_embed=spk_embed,
                spk_id=spk_id,
                emotion_id=emotion_id,
                f0=f0,
                uv=uv,
                infer=True,
            )
            return output

    def add_dur_loss(self, dur_pred, mel2ph, txt_tokens, losses=None):
        """

        :param dur_pred: [B, T], float, log scale
        :param mel2ph: [B, T]
        :param txt_tokens: [B, T]
        :param losses:
        :return:
        """
        B, T = txt_tokens.shape
        nonpadding = (txt_tokens != 0).float()
        dur_gt = mel2token_to_dur(mel2ph, T).float() * nonpadding
        is_sil = torch.zeros_like(txt_tokens).bool()
        for p in self.sil_ph:
            is_sil = is_sil | (txt_tokens == self.token_encoder.encode(p)[0])
        is_sil = is_sil.float()  # [B, T_txt]
        losses["pdur"] = F.mse_loss(
            (dur_pred + 1).log(), (dur_gt + 1).log(), reduction="none"
        )
        losses["pdur"] = (losses["pdur"] * nonpadding).sum() / nonpadding.sum()
        losses["pdur"] = losses["pdur"] * hparams["lambda_ph_dur"]
        # use linear scale for sentence and word duration
        if hparams["lambda_word_dur"] > 0:
            word_id = (is_sil.cumsum(-1) * (1 - is_sil)).long()
            word_dur_p = dur_pred.new_zeros([B, word_id.max() + 1]).scatter_add(
                1, word_id, dur_pred
            )[:, 1:]
            word_dur_g = dur_gt.new_zeros([B, word_id.max() + 1]).scatter_add(
                1, word_id, dur_gt
            )[:, 1:]
            wdur_loss = F.mse_loss(
                (word_dur_p + 1).log(), (word_dur_g + 1).log(), reduction="none"
            )
            word_nonpadding = (word_dur_g > 0).float()
            wdur_loss = (wdur_loss * word_nonpadding).sum() / word_nonpadding.sum()
            losses["wdur"] = wdur_loss * hparams["lambda_word_dur"]
        if hparams["lambda_sent_dur"] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.mse_loss(
                (sent_dur_p + 1).log(), (sent_dur_g + 1).log(), reduction="mean"
            )
            losses["sdur"] = sdur_loss.mean() * hparams["lambda_sent_dur"]

    def add_pitch_loss(self, output, sample, losses):
        mel2ph = sample["mel2ph"]  # [B, T_s]
        f0 = sample["f0"]
        uv = sample["uv"]
        nonpadding = (
            (mel2ph != 0).float()
            if hparams["pitch_type"] == "frame"
            else (sample["txt_tokens"] != 0).float()
        )
        p_pred = output["pitch_pred"]
        assert p_pred[..., 0].shape == f0.shape
        if hparams["use_uv"] and hparams["pitch_type"] == "frame":
            assert p_pred[..., 1].shape == uv.shape, (p_pred.shape, uv.shape)
            losses["uv"] = (
                (
                    F.binary_cross_entropy_with_logits(
                        p_pred[:, :, 1], uv, reduction="none"
                    )
                    * nonpadding
                ).sum()
                / nonpadding.sum()
                * hparams["lambda_uv"]
            )
            nonpadding = nonpadding * (uv == 0).float()
        f0_pred = p_pred[:, :, 0]
        losses["f0"] = (
            (F.l1_loss(f0_pred, f0, reduction="none") * nonpadding).sum()
            / nonpadding.sum()
            * hparams["lambda_f0"]
        )

    def save_valid_result(self, sample, batch_idx, model_out):
        sr = hparams["audio_sample_rate"]
        f0_gt = None
        mel_out = model_out["mel_out"]
        if sample.get("f0") is not None:
            f0_gt = denorm_f0(sample["f0"][0].cpu(), sample["uv"][0].cpu())
        self.plot_mel(batch_idx, sample["mels"], mel_out, f0s=f0_gt)
        if self.global_step > 0:
            wav_pred = self.vocoder.spec2wav(mel_out[0].cpu(), f0=f0_gt)
            self.logger.add_audio(
                f"wav_val_{batch_idx}", wav_pred, self.global_step, sr
            )
            # with gt duration
            model_out = self.run_model(sample, infer=True, infer_use_gt_dur=True)
            dur_info = self.get_plot_dur_info(sample, model_out)
            del dur_info["dur_pred"]
            wav_pred = self.vocoder.spec2wav(model_out["mel_out"][0].cpu(), f0=f0_gt)
            self.logger.add_audio(
                f"wav_gdur_{batch_idx}", wav_pred, self.global_step, sr
            )
            self.plot_mel(
                batch_idx,
                sample["mels"],
                model_out["mel_out"][0],
                f"mel_gdur_{batch_idx}",
                dur_info=dur_info,
                f0s=f0_gt,
            )

            # with pred duration
            if not hparams["use_gt_dur"]:
                model_out = self.run_model(sample, infer=True, infer_use_gt_dur=False)
                dur_info = self.get_plot_dur_info(sample, model_out)
                self.plot_mel(
                    batch_idx,
                    sample["mels"],
                    model_out["mel_out"][0],
                    f"mel_pdur_{batch_idx}",
                    dur_info=dur_info,
                    f0s=f0_gt,
                )
                wav_pred = self.vocoder.spec2wav(
                    model_out["mel_out"][0].cpu(), f0=f0_gt
                )
                self.logger.add_audio(
                    f"wav_pdur_{batch_idx}", wav_pred, self.global_step, sr
                )
        # gt wav
        if self.global_step <= hparams["valid_infer_interval"]:
            mel_gt = sample["mels"][0].cpu()
            wav_gt = self.vocoder.spec2wav(mel_gt, f0=f0_gt)
            self.logger.add_audio(f"wav_gt_{batch_idx}", wav_gt, self.global_step, sr)

    def get_plot_dur_info(self, sample, model_out):
        T_txt = sample["txt_tokens"].shape[1]
        dur_gt = mel2token_to_dur(sample["mel2ph"], T_txt)[0]
        dur_pred = model_out["dur"] if "dur" in model_out else dur_gt
        txt = self.token_encoder.decode(sample["txt_tokens"][0].cpu().numpy())
        txt = txt.split(" ")
        return {"dur_gt": dur_gt, "dur_pred": dur_pred, "txt": txt}

    def _training_step(self, sample, batch_idx, optimizer_idx):
        loss_output = {}
        loss_weights = {}
        disc_start = hparams["adv"]

        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            loss_output, model_out = self.run_model(sample)
            self.model_out_gt = self.model_out = {
                k: v.detach()
                for k, v in model_out.items()
                if isinstance(v, torch.Tensor)
            }
            if disc_start:
                mel_len = sample["mel_lengths"]
                mel_g = sample["mels"]
                mel_p = model_out["mel_out"]
                o = self.mel_disc(mel_g, mel_len)
                o_ = self.mel_disc(mel_p, mel_len)

                p = o["y"]
                p_ = o_["y"]

                if hparams["lambda_mel_adv"] > 0.0:
                    if p_ is not None:
                        loss_output["a"] = F.mse_loss(p_, p_.new_ones(p_.size()))
                        loss_weights["a"] = hparams["lambda_mel_adv"]

            total_loss = sum(
                [
                    loss_weights.get(k, 1) * v
                    for k, v in loss_output.items()
                    if isinstance(v, torch.Tensor) and v.requires_grad
                ]
            )
            loss_output["total_loss"] = total_loss
            loss_output["batch_size"] = sample["txt_tokens"].size()[0]

            return total_loss, loss_output
        else:
            #######################
            #    Discriminator    #
            #######################
            if disc_start and self.global_step % hparams["disc_interval"] == 0:
                model_out = self.model_out_gt

                mel_len = sample["mel_lengths"]
                mel_g = sample["mels"]
                mel_p = model_out["mel_out"].detach()
                o = self.mel_disc(mel_g, mel_len)
                o_ = self.mel_disc(mel_p, mel_len)

                p = o["y"]
                p_ = o_["y"]
                if hparams["lambda_mel_adv"] > 0.0:
                    if p_ is not None:
                        loss_output["r"] = F.mse_loss(p, p.new_ones(p.size()))
                        loss_output["f"] = F.mse_loss(p_, p_.new_zeros(p_.size()))

                total_loss = sum(
                    [
                        loss_weights.get(k, 1) * v
                        for k, v in loss_output.items()
                        if isinstance(v, torch.Tensor) and v.requires_grad
                    ]
                )
                loss_output["total_loss_d"] = total_loss
                loss_output["batch_size"] = sample["txt_tokens"].size()[0]

                return total_loss, loss_output
            total_loss = torch.Tensor([0]).float()
            return total_loss, loss_output

    def build_optimizer(self, model):
        optimizer_gen = torch.optim.AdamW(
            self.model.parameters(),
            lr=hparams["lr"],
            betas=(hparams["optimizer_adam_beta1"], hparams["optimizer_adam_beta2"]),
            weight_decay=hparams["weight_decay"],
        )
        if hparams["adv"]:
            optimizer_disc = (
                torch.optim.AdamW(
                    self.disc_params,
                    lr=hparams["disc_lr"],
                    betas=(
                        hparams["optimizer_adam_beta1"],
                        hparams["optimizer_adam_beta2"],
                    ),
                    **hparams["discriminator_optimizer_params"],
                )
                if len(self.disc_params) > 0
                else None
            )
        else:
            optimizer_disc = None
        return [optimizer_gen, optimizer_disc]

    def build_scheduler(self, optimizer):
        scheduler1 = WarmupSchedule(
            optimizer[0], hparams["lr"], hparams["warmup_updates"]
        )
        if hparams["adv"]:
            scheduler2 = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer[1],  # Discriminator Scheduler
                **hparams["discriminator_scheduler_params"],
            )
        else:
            scheduler2 = None
        return [scheduler1, scheduler2]

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is not None:
            self.scheduler[0].step(
                self.global_step // hparams["accumulate_grad_batches"]
            )
            if hparams["adv"]:
                self.scheduler[1].step(
                    self.global_step // hparams["accumulate_grad_batches"]
                )

    # def test_step(self, sample, batch_idx):
    #     """

    #     :param sample:
    #     :param batch_idx:
    #     :return:
    #     """
    #     assert (
    #         sample["txt_tokens"].shape[0] == 1
    #     ), "only support batch_size=1 in inference"
    #     outputs = self.run_model(sample, infer=True)
    #     text = sample["text"][0]
    #     item_name = sample["item_name"][0]
    #     tokens = sample["txt_tokens"][0].cpu().numpy()
    #     mel_gt = sample["mels"][0].cpu().numpy()
    #     mel_pred = outputs["mel_out"][0].cpu().numpy()
    #     mel2ph = sample["mel2ph"][0].cpu().numpy()
    #     mel2ph_pred = outputs["mel2ph"][0].cpu().numpy()
    #     str_phs = self.token_encoder.decode(tokens, strip_padding=True)
    #     # base_fn = f'[{batch_idx:06d}][{item_name.replace("%", "_")}][%s]'
    #     # if text is not None:
    #     #     base_fn += text.replace(":", "$3A")[:80]
    #     # base_fn = base_fn.replace(" ", "_")
    #     base_fn = item_name

    #     gen_dir = self.gen_dir
    #     wav_pred = self.vocoder.spec2wav(mel_pred)
    #     self.saving_result_pool.add_job(
    #         self.save_result,
    #         args=[wav_pred, mel_pred, base_fn, gen_dir, str_phs, mel2ph_pred],
    #     )
    #     # if hparams["save_gt"]:
    #     #     wav_gt = self.vocoder.spec2wav(mel_gt)
    #     #     self.saving_result_pool.add_job(
    #     #         self.save_result,
    #     #         args=[wav_gt, mel_gt, base_fn % "G", gen_dir, str_phs, mel2ph],
    #     #     )
    #     # os.makedirs(f"{gen_dir}/mel", exist_ok=True)
    #     # torch.save(mel_pred, f"{gen_dir}/mel/{base_fn}.pt")
    #     # torch.save(mel_gt, f"{gen_dir}/mel/{base_fn}_gt.pt")

    #     print(f"Pred_shape: {mel_pred.shape}, gt_shape: {mel_gt.shape}")
    #     return {
    #         "item_name": item_name,
    #         "text": text,
    #         "ph_tokens": self.token_encoder.decode(tokens.tolist()),
    #         "wav_fn_pred": base_fn,
    #         # "wav_fn_gt": base_fn % "G",
    #     }

    def test_step(self, sample, batch_idx):
        assert (
            sample["txt_tokens"].shape[0] == 1
        ), "only support batch_size=1 in inference"
        # outputs = self.run_model(sample, infer=True)
        emo_dict = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
        if hparams["gen_dir_name"] == "recon":
            outputs = self.run_model(sample, infer=True)
            text = sample["text"][0]
            item_name = sample["item_name"][0]
            tokens = sample["txt_tokens"][0].cpu().numpy()
            mel_gt = sample["mels"][0].cpu().numpy()
            mel2ph_pred = None
            str_phs = self.token_encoder.decode(tokens, strip_padding=True)
            mel_pred = outputs["mel_out"][0].cpu().numpy()

            base_fn = item_name
            gen_dir = self.gen_dir
            wav_pred = self.vocoder.spec2wav(mel_pred)
            self.saving_result_pool.add_job(
                self.save_result,
                args=[wav_pred, mel_pred, base_fn, gen_dir, str_phs, mel2ph_pred],
            )
            print(f"Pred_shape: {mel_pred.shape}, gt_shape: {mel_gt.shape}")
        else:
            for i in range(5):
                emotion_id = torch.LongTensor([i]).to(sample["txt_tokens"].device)
                txt_tokens = sample["txt_tokens"]  # [B, T_t]
                spk_embed = sample.get("spk_embed")
                spk_id = sample.get("spk_ids")
                # emotion_id = sample.get("emotion_ids")
                outputs = self.model(
                    txt_tokens,
                    spk_embed=spk_embed,
                    spk_id=spk_id,
                    emotion_id=emotion_id,
                    infer=True,
                )
                text = sample["text"][0]
                item_name = sample["item_name"][0]
                tokens = sample["txt_tokens"][0].cpu().numpy()
                mel_gt = sample["mels"][0].cpu().numpy()
                mel2ph = sample["mel2ph"][0].cpu().numpy()
                mel2ph_pred = None
                str_phs = self.token_encoder.decode(tokens, strip_padding=True)
                mel_pred = outputs["mel_out"][0].cpu().numpy()

                base_fn = item_name + "_" + emo_dict[i]
                gen_dir = self.gen_dir
                wav_pred = self.vocoder.spec2wav(mel_pred)
                self.saving_result_pool.add_job(
                    self.save_result,
                    args=[wav_pred, mel_pred, base_fn, gen_dir, str_phs, mel2ph_pred],
                )
                print(f"Pred_shape: {mel_pred.shape}, gt_shape: {mel_gt.shape}")
        return {
            "item_name": item_name,
            "text": text,
            "ph_tokens": self.token_encoder.decode(tokens.tolist()),
            "wav_fn_pred": base_fn,
            # "wav_fn_gt": base_fn % "G",
        }
