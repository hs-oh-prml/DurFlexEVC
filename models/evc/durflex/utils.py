import torch
import torch.nn.functional as F
from utils.commons.dataset_utils import collate_1d_or_2d
from itertools import groupby


def dedup_seq(seq):
    B, L = seq.shape
    vals, counts = [], []
    for i in range(B):
        val, count = zip(*[(k.item(), sum(1 for _ in g)) for k, g in groupby(seq[i])])
        vals.append(torch.LongTensor(val))
        counts.append(torch.LongTensor(count))
    vals = collate_1d_or_2d(vals, 0)
    counts = collate_1d_or_2d(counts, 0)
    return vals, counts


def fix_len_compatibility(length, num_downsamplings_in_unet=3):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return int(length)
        length += 1


def expand_states(h, mel2token):
    h = F.pad(h, [0, 0, 1, 0])
    mel2token_ = mel2token[..., None].repeat([1, 1, h.shape[-1]])
    h = torch.gather(h, 1, mel2token_)  # [B, T, H]
    return h


def clip_mel2token_to_multiple(mel2token, frames_multiple):
    max_frames = mel2token.shape[1] // frames_multiple * frames_multiple
    mel2token = mel2token[:, :max_frames]
    return mel2token


class LengthRegulator(torch.nn.Module):
    def __init__(self, pad_value=0.0):
        super(LengthRegulator, self).__init__()
        self.pad_value = pad_value

    def forward(self, dur, dur_padding=None, alpha=1.0):
        """
        Example (no batch dim version):
            1. dur = [2,2,3]
            2. token_idx = [[1],[2],[3]], dur_cumsum = [2,4,7], dur_cumsum_prev = [0,2,4]
            3. token_mask = [[1,1,0,0,0,0,0],
                             [0,0,1,1,0,0,0],
                             [0,0,0,0,1,1,1]]
            4. token_idx * token_mask = [[1,1,0,0,0,0,0],
                                         [0,0,2,2,0,0,0],
                                         [0,0,0,0,3,3,3]]
            5. (token_idx * token_mask).sum(0) = [1,1,2,2,3,3,3]

        :param dur: Batch of durations of each frame (B, T_txt)
        :param dur_padding: Batch of padding of each frame (B, T_txt)
        :param alpha: duration rescale coefficient
        :return:
            mel2ph (B, T_speech)
        assert alpha > 0
        """
        dur = torch.round(dur.float() * alpha).long()
        if dur_padding is not None:
            dur = dur * (1 - dur_padding.long())
        token_idx = torch.arange(1, dur.shape[1] + 1)[None, :, None].to(dur.device)
        dur_cumsum = torch.cumsum(dur, 1)
        dur_cumsum_prev = F.pad(dur_cumsum, [1, -1], mode="constant", value=0)

        pos_idx = torch.arange(dur.sum(-1).max())[None, None].to(dur.device)
        token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (
            pos_idx < dur_cumsum[:, :, None]
        )
        mel2token = (token_idx * token_mask.long()).sum(1)
        return mel2token