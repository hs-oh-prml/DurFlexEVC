import torch
from torch import nn
import random


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.linear_layer(x)


class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1, eps=1e-5):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=eps)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Permute(nn.Module):
    def __init__(self, *args):
        super(Permute, self).__init__()
        self.args = args

    def forward(self, x):
        return x.permute(self.args)


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


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
        style = self.style(style_code)
        gamma, beta = style.chunk(2, dim=-1)
        out = self.norm(input)
        out = gamma * out + beta
        return out



class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, hidden_size=256):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self._activated = True
        self.hidden_size = hidden_size
        self.affine_layer = LinearNorm(
            hidden_size,
            2 * hidden_size,  # For both b (bias) g (gain)
        )

    def __repr__(self):
        return f"MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps})"

    def set_activation_status(self, status=True):
        self._activated = status

    def forward(self, x, spk_embed):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu, sig = torch.mean(x, dim=-1, keepdim=True), torch.std(
            x, dim=-1, keepdim=True
        )
        x_normed = (x - mu) / (sig + 1e-6)  # [B, T, H_m]

        lmda = self.beta.sample((B, 1, 1))
        lmda = lmda.to(x.device)

        # Get Bias and Gain
        mu1, sig1 = torch.split(
            self.affine_layer(spk_embed), self.hidden_size, dim=-1
        )  # [B, 1, 2 * H_m] --> 2 * [B, 1, H_m]

        # MixStyle
        perm = torch.randperm(B)
        mu2, sig2 = mu1[perm], sig1[perm]

        mu_mix = mu1 * lmda + mu2 * (1 - lmda)
        sig_mix = sig1 * lmda + sig2 * (1 - lmda)

        # Perform Scailing and Shifting
        return sig_mix * x_normed + mu_mix  # [B, T, H_m]
