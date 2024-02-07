import random
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from modules.commons.layers import Embedding
from modules.commons.transformer import FFTBlocks, MultiheadAttention
from modules.tts.commons.align_ops import expand_states, dedup_seq
from modules.tts.commons.align_ops import clip_mel2token_to_multiple, expand_states
from modules.durflex_evc.diffusion import Diffusion, Mish
from modules.durflex_evc.duration_predictor import StochasticDurationPredictor
from modules.durflex_evc.style_encoder import StyleEncoder
from modules.commons.nar_tts_modules import LengthRegulator

from utils.audio.align import mel2token_to_dur
from utils.nn.seq_utils import (
    group_hidden_by_segs,
    sequence_mask,
    fix_len_compatibility,
)
from pytorch_revgrad import RevGrad


class DurFlex(nn.Module):
    def __init__(self, dict_size, hparams):
        super().__init__()
        self.hparams = deepcopy(hparams)

        self.n_feats = hparams["audio_num_mel_bins"]
        self.hidden_size = hparams["hidden_size"]
        self.emo_dict = {"Angry": 0, "Happy": 1, "Neutral": 2, "Sad": 3, "Surprise": 4}
        self.emotion_id_proj = Embedding(5, self.hidden_size)
        self.scale = 320 / hparams["hop_size"]
        self.num_downsamplings_in_unet = len(hparams["decoder"]["dim_mults"]) - 1
        self.segment_size = hparams["segment_size"]

        if hparams["use_spk_id"]:
            self.spk_id_proj = Embedding(hparams["num_spk"], self.hidden_size)
        if hparams["use_spk_embed"]:
            self.spk_embed_proj = nn.Linear(256, self.hidden_size, bias=True)
        if hparams["use_spk_encoder"]:
            self.spk_encoder = StyleEncoder()
            self.emo_clf = torch.nn.Sequential(
                RevGrad(),
                nn.Linear(256, 1024),
                Mish(),
                nn.Linear(1024, 256),
                nn.Linear(256, 5),
            )

        self.input_proj = nn.Linear(hparams["feature_dims"], self.hidden_size)

        self.destyle_fft = FFTBlocks(
            self.hidden_size,
            hparams["enc_layers"],
            hparams["enc_kernel_size"],
            num_heads=hparams["num_heads"],
            norm="mixln",
        )
        self.style_fft = FFTBlocks(
            self.hidden_size,
            hparams["enc_layers"],
            hparams["enc_kernel_size"],
            num_heads=hparams["num_heads"],
            norm="cln",
        )

        self.embed = nn.Parameter(torch.FloatTensor(dict_size, self.hidden_size))
        nn.init.normal_(self.embed, mean=0, std=0.5)
        self.unit_attention = MultiheadAttention(
            self.hidden_size,
            hparams["unit_attn_num_heads"],
            dropout=hparams["unit_attn_dropout"],
        )
        self.unit_level_encoder = FFTBlocks(
            self.hidden_size,
            hparams["enc_layers"],
            hparams["enc_kernel_size"],
            num_heads=hparams["num_heads"],
            norm="cln",
        )

        self.frame_level_encoder = FFTBlocks(
            self.hidden_size,
            hparams["enc_layers"],
            hparams["enc_kernel_size"],
            num_heads=hparams["num_heads"],
            norm="cln",
        )

        self.decoder = Diffusion(
            n_feats=hparams["audio_num_mel_bins"],
            dim=hparams["decoder"]["dim"],
            dim_mults=hparams["decoder"]["dim_mults"],
            pe_scale=hparams["decoder"]["pe_scale"],
            beta_min=hparams["decoder"]["beta_min"],
            beta_max=hparams["decoder"]["beta_max"],
            spk_emb_dim=hparams["decoder"]["spk_emb_dim"],
        )

        self.proj_m = nn.Linear(hparams["hidden_size"], hparams["audio_num_mel_bins"])

        self.dur_predictor = StochasticDurationPredictor(
            hparams["hidden_size"],
            hparams["hidden_size"],
            3,
            0.5,
            4,
            gin_channels=hparams["hidden_size"],
        )
        self.length_regulator = LengthRegulator()

    def forward(
        self,
        features=None,
        mel2unit=None,
        spk_embed=None,
        spk_id=None,
        emotion_id=None,
        tgt_emotion_id=None,
        infer=False,
        y=None,
        y_lengths=None,
        diffusion_step=100,
        **kwargs,
    ):
        ret = {}
        src_spk_embed, src_emo_embed, tgt_emo_embed = self.forward_style_embed(
            y, y_lengths, emotion_id, spk_embed, spk_id, tgt_emotion_id=tgt_emotion_id
        )
        if self.hparams["use_spk_encoder"]:
            emo_logits = self.emo_clf(spk_embed)
            ret["emo_logits"] = emo_logits
        src_style_embed = src_spk_embed + src_emo_embed
        tgt_style_embed = src_spk_embed + tgt_emo_embed

        x = self.forward_sae(features, src_style_embed, tgt_style_embed)
        x, x_nonpadding, units, mel2unit = self.forward_ua(x, ret)

        x = self.unit_level_encoder(
            x, style_vector=tgt_style_embed.unsqueeze(1).transpose(0, 1)
        )

        dur_inp = x * x_nonpadding
        if infer:
            mel2unit = None
        mel2unit = self.forward_dur(
            dur_inp, mel2unit, units, tgt_style_embed, ret, infer
        )

        tgt_nonpadding = (mel2unit > 0).float()[:, :, None]
        x = expand_states(x, mel2unit)

        ret["unit_nonpadding"] = tgt_nonpadding.squeeze(-1)
        if self.scale != 1:
            if not infer:
                b, l, h = y.shape
            else:
                l = round(x.shape[1] * self.scale)
            x = F.interpolate(
                x.transpose(1, 2),
                size=l,
                mode="linear",
            ).transpose(1, 2)
            tgt_nonpadding = F.interpolate(
                tgt_nonpadding.float().transpose(1, 2),
                size=l,
                mode="linear",
            ).transpose(1, 2)
            mel2unit = (
                F.interpolate(
                    mel2unit.float().unsqueeze(1),
                    size=l,
                    mode="linear",
                )
                .long()
                .squeeze(1)
            )

        style_embed = tgt_style_embed.unsqueeze(1)
        if style_embed.shape[1] == 1:
            style_embed = style_embed.expand(x.shape[0], x.shape[1], -1)

        x = self.frame_level_encoder(
            x,
            style_vector=style_embed.transpose(0, 1),
        )

        x = self.proj_m(x)
        x = x * tgt_nonpadding

        x = x.transpose(1, 2)  #
        y_max_length = x.shape[-1]
        y_mask = tgt_nonpadding  # .transpose(1, 2)
        style_embed = style_embed.transpose(1, 2)
        if not infer:
            if y_max_length < self.segment_size:
                pad_size = self.segment_size - y_max_length
                y = torch.cat([y, torch.zeros_like(y)[:, :, :pad_size]], dim=-1)
                y_mask = torch.cat(
                    [y_mask, torch.zeros_like(y_mask)[:, :, :pad_size]], dim=-1
                )
                x = torch.cat([x, torch.zeros_like(x)[:, :, :pad_size]], dim=-1)

            max_offset = (y_lengths - self.segment_size).clamp(0)
            offset_ranges = list(
                zip([0] * max_offset.shape[0], max_offset.cpu().numpy())
            )
            out_offset = torch.LongTensor(
                [
                    torch.tensor(random.choice(range(start, end)) if end > start else 0)
                    for start, end in offset_ranges
                ]
            ).to(y_lengths)
            x_cut = torch.zeros(
                x.shape[0],
                x.shape[1],
                self.segment_size,
                dtype=x.dtype,
                device=x.device,
            )
            y_cut = torch.zeros(
                y.shape[0],
                self.n_feats,
                self.segment_size,
                dtype=y.dtype,
                device=y.device,
            )
            style_embed_cut = torch.zeros(
                style_embed.shape[0],
                style_embed.shape[1],
                self.segment_size,
                dtype=x.dtype,
                device=x.device,
            )

            y_cut_lengths = []
            y = y.transpose(1, 2)
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = self.segment_size + (
                    y_lengths[i] - self.segment_size
                ).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                x_cut[i, :, :y_cut_length] = x[i, :, cut_lower:cut_upper]
                style_embed_cut[i, :, :y_cut_length] = style_embed[
                    i, :, cut_lower:cut_upper
                ]

            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)
            if y_cut_mask.shape[-1] < self.segment_size:
                y_cut_mask = torch.nn.functional.pad(
                    y_cut_mask, (0, self.segment_size - y_cut_mask.shape[-1])
                )
            x_cut = x_cut * y_cut_mask

            diff_loss, xt = self.decoder.compute_loss(
                y_cut,
                y_cut_mask,
                x_cut,
                spk_emb=style_embed_cut,  # style_embed,
            )
            ret["diff_loss"] = diff_loss
        else:
            y_max_length_ = fix_len_compatibility(
                y_max_length, self.num_downsamplings_in_unet
            )
            y_mask = (
                sequence_mask(
                    torch.LongTensor([y_max_length]).to(x.device),
                    y_max_length_,
                )
                .unsqueeze(1)
                .to(y_mask.dtype)
            )
            x = torch.cat(
                [
                    x,
                    torch.zeros_like(x)[:, :, : y_max_length_ - y_max_length],
                ],
                dim=-1,
            )
            style_embed = torch.cat(
                [
                    style_embed,
                    torch.zeros_like(style_embed)[:, :, : y_max_length_ - y_max_length],
                ],
                dim=-1,
            )
            z = torch.randn_like(x, device=x.device)
            decoder_outputs = self.decoder(
                z,
                y_mask,
                x,
                spk_emb=style_embed,  # style_embed,
                n_timesteps=diffusion_step,
            )
            decoder_outputs = decoder_outputs[:, :, :y_max_length]
            ret["mel_out"] = decoder_outputs.transpose(1, 2)
        return ret

    def forward_style_embed(
        self,
        y,
        y_length=None,
        emotion_id=None,
        spk_embed=None,
        spk_id=None,
        tgt_emotion_id=None,
    ):
        if self.hparams["use_spk_id"]:
            spk_embed = self.spk_id_proj(spk_id)
        if self.hparams["use_spk_embed"]:
            spk_embed = self.spk_embed_proj(spk_embed)
        if self.hparams["use_spk_encoder"]:
            y_mask = sequence_mask(y_length).unsqueeze(1)
            spk_embed = self.spk_encoder(y.transpose(1, 2), y_mask)
        if tgt_emotion_id is not None:
            tgt_emo_embed = self.emotion_id_proj(emotion_id)
        else:
            tgt_emo_embed = None
        emo_embed = self.emotion_id_proj(emotion_id)

        return spk_embed, emo_embed, tgt_emo_embed

    def forward_sae(self, x, srs_style, tgt_style):
        x = self.input_proj(x)
        x = self.destyle_fft(x, style_vector=srs_style.unsqueeze(1).transpose(0, 1))
        x = self.style_fft(
            x,
            style_vector=tgt_style.unsqueeze(1).transpose(0, 1),
        )
        return x

    def forward_ua(self, x, ret):
        query = x  # [N, 1, E//2]
        N, L, _ = x.shape
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)
        attn, attn_weights = self.unit_attention(
            query.transpose(0, 1),
            keys.transpose(0, 1),
            keys.transpose(0, 1),
            need_weights=True,
            before_softmax=True,
        )
        ret["unit_logits"] = torch.log(attn_weights + 1e-9)
        unit_pred = torch.argmax(attn_weights, dim=-1)
        ret["unit_pred_frame"] = unit_pred
        _, count = dedup_seq(unit_pred)
        count = count.cuda()
        mel2unit = self.length_regulator(count)
        unit_len = mel2unit.max()
        units = (
            group_hidden_by_segs(unit_pred.unsqueeze(-1), mel2unit, unit_len)[0]
            .squeeze(-1)
            .long()
        )
        ret["mel2unit"] = mel2unit
        ret["units"] = units
        x = group_hidden_by_segs(attn.transpose(0, 1), mel2unit, unit_len)[0]
        x_nonpadding = (units > 0).float()[:, :, None]
        return x, x_nonpadding, units, mel2unit

    def forward_dur(
        self,
        dur_input,
        mel2ph,
        txt_tokens,
        style_embed,
        ret,
        infer,
        length_scale=1,
        noise_scale_w=1.0,
    ):
        """

        :param dur_input: [B, T_txt, H]
        :param mel2ph: [B, T_mel]
        :param txt_tokens: [B, T_txt]
        :param ret:
        :return:
        """
        src_padding = txt_tokens == 0
        B, T = txt_tokens.shape
        nonpadding = (txt_tokens != 0).float()
        dur_input = dur_input.detach()
        if infer:
            logw = self.dur_predictor(
                dur_input.transpose(1, 2),
                nonpadding.unsqueeze(1),
                g=style_embed.unsqueeze(-1),
                reverse=True,
                noise_scale=noise_scale_w,
            )
            dur = torch.exp(logw) * nonpadding * length_scale
            dur = torch.ceil(dur).squeeze(1)
            mel2ph = self.length_regulator(dur, src_padding).detach()
        else:
            dur_gt = mel2token_to_dur(mel2ph, T).float() * nonpadding
            dur = self.dur_predictor(
                dur_input.transpose(1, 2),
                nonpadding.unsqueeze(1),
                dur_gt.unsqueeze(1),
                g=style_embed.unsqueeze(-1),
            )
            dur = dur / torch.sum(nonpadding)

        ret["dur"] = dur
        mel2ph = clip_mel2token_to_multiple(mel2ph, self.hparams["frames_multiple"])
        return mel2ph
