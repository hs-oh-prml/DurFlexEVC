import torch
import torch.nn.functional as F
from utils.commons.dataset_utils import collate_1d_or_2d
from itertools import groupby


def build_word_mask(x2word, y2word):
    return (x2word[:, :, None] == y2word[:, None, :]).long()


def mel2ph_to_mel2word(mel2ph, ph2word):
    mel2word = (ph2word - 1).gather(1, (mel2ph - 1).clamp(min=0)) + 1
    mel2word = mel2word * (mel2ph > 0).long()
    return mel2word


def clip_mel2token_to_multiple(mel2token, frames_multiple):
    max_frames = mel2token.shape[1] // frames_multiple * frames_multiple
    mel2token = mel2token[:, :max_frames]
    return mel2token


def expand_states(h, mel2token):
    h = F.pad(h, [0, 0, 1, 0])
    mel2token_ = mel2token[..., None].repeat([1, 1, h.shape[-1]])
    h = torch.gather(h, 1, mel2token_)  # [B, T, H]
    return h


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
