import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.commons.transformer import MultiheadAttention


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, dropout):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = attn * mask
        attn = self.softmax(attn)
        p_attn = self.dropout(attn)

        output = torch.bmm(p_attn, v)
        return output, attn


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class AffineLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AffineLinear, self).__init__()
        affine = nn.Linear(in_dim, out_dim)
        self.affine = affine

    def forward(self, input):
        return self.affine(input)


class LinearNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
        spectral_norm=False,
    ):
        super(LinearNorm, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias)

        if spectral_norm:
            self.fc = nn.utils.spectral_norm(self.fc)

    def forward(self, input):
        out = self.fc(input)
        return out


class ConvNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        spectral_norm=False,
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, input):
        out = self.conv(input)
        return out


class StyleAdaptiveEncoder(nn.Module):
    def __init__(
        self,
        n_layers,
        d_model,
        n_heads,
        style_dim,
        dec_dim,
        fft_conv1d_filter_size,
        fft_conv1d_kernel_size,
        enc_dropout,
    ):
        super(StyleAdaptiveEncoder, self).__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_head = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.d_inner = fft_conv1d_filter_size
        self.fft_conv1d_kernel_size = fft_conv1d_kernel_size
        self.d_out = dec_dim
        self.style_dim = style_dim
        self.dropout = enc_dropout

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    self.d_model,
                    self.d_inner,
                    self.n_head,
                    self.fft_conv1d_kernel_size,
                    self.style_dim,
                    self.dropout,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.fc_out = nn.Linear(self.d_model, self.d_out)

    def forward(self, x, style_vector, mask=None):
        mask = x.abs().sum(-1).eq(0).data if mask is None else mask
        # max_len = x.shape[1]
        # slf_attn_mask = mask.expand(-1, max_len, -1)
        slf_attn_mask = 1 - mask.transpose(0, 1).float()[:, :, None]  # [T, B, 1]
        print(mask.shape, slf_attn_mask.shape)
        # fft blocks
        for enc_layer in self.layer_stack:
            x, _ = enc_layer(x, style_vector, mask=mask, slf_attn_mask=slf_attn_mask)

        # last fc
        x = self.fc_out(x)
        return x


class FFTBlock(nn.Module):
    """FFT Block"""

    def __init__(
        self,
        d_model,
        d_inner,
        n_head,
        fft_conv1d_kernel_size,
        style_dim,
        dropout,
    ):
        super(FFTBlock, self).__init__()

        self.slf_attn = MultiheadAttention(
            d_model,
            n_head,
            encoder_decoder_attention=True,
            dropout=dropout,
            bias=False,
        )
        self.saln_0 = StyleAdaptiveLayerNorm(d_model, style_dim)

        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, fft_conv1d_kernel_size, dropout=dropout
        )
        self.saln_1 = StyleAdaptiveLayerNorm(d_model, style_dim)

    def forward(self, x, style_vector, mask=None, slf_attn_mask=None):
        # multi-head self attn
        residual = x
        x, slf_attn = self.slf_attn(
            query=x, key=x, value=x, key_padding_mask=slf_attn_mask
        )
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = x * (1 - slf_attn_mask.float()).transpose(0, 1)[..., None]
        x = self.saln_0(x, style_vector)
        x = x * mask

        # position wise FF
        output = self.pos_ffn(x)

        output = self.saln_1(output, style_vector)
        output = output * mask

        return output, slf_attn


class AffineLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AffineLinear, self).__init__()
        affine = nn.Linear(in_dim, out_dim)
        self.affine = affine

    def forward(self, input):
        return self.affine(input)


class StyleAdaptiveLayerNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super(StyleAdaptiveLayerNorm, self).__init__()
        self.in_channel = in_channel
        self.norm = nn.LayerNorm(in_channel, elementwise_affine=False)

        self.style = AffineLinear(style_dim, in_channel * 2)
        self.style.affine.bias.data[:in_channel] = 1
        self.style.affine.bias.data[in_channel:] = 0

    def forward(self, input, style_code):
        # style
        style = self.style(style_code).unsqueeze(1)
        gamma, beta = style.chunk(2, dim=-1)

        out = self.norm(input)
        out = gamma * out + beta
        return out


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, fft_conv1d_kernel_size, dropout=0.1):
        super().__init__()
        self.w_1 = ConvNorm(d_in, d_hid, kernel_size=fft_conv1d_kernel_size[0])
        self.w_2 = ConvNorm(d_hid, d_in, kernel_size=fft_conv1d_kernel_size[1])

        self.mish = Mish()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        residual = input

        output = input.transpose(1, 2)
        output = self.w_2(self.dropout(self.mish(self.w_1(output))))
        output = output.transpose(1, 2)

        output = self.dropout(output) + residual
        return output
