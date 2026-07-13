import torch
import numpy as np

from .transformer import Transformer

class FourierFeatures(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer("freqs", 2 * torch.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * torch.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.flatten().outer(self.freqs)
        y = y + self.phases
        y = y.cos() * np.sqrt(2)
        return y.reshape((x.shape[0], x.shape[1], x.shape[2]*y.shape[-1]))

class PointFeatureExtractor(Transformer):
    """Self-attention feature extractor (l_self layers, h heads), Section 4.2."""
    def __init__(self, in_dim, embed_dim=256, num_heads=8, num_layers=4, dropout=0.0):
        super(PointFeatureExtractor, self).__init__(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout, residual=True)
        self.linear_in = torch.nn.Linear(in_dim, embed_dim)

    def forward(self, x:torch.Tensor, return_hiddens=False):
        x = self.linear_in(x)
        x, hiddens = super(PointFeatureExtractor, self).forward(x, x, return_hiddens=True)
        if return_hiddens:
            return x, hiddens
        else:
            return x

class PointMatcher(Transformer):
    """Final single-head cross-attention layer whose pre-softmax logits are the correspondence matrix rho."""
    def __init__(self, embed_dim=256, dropout=0.0):
        super(PointMatcher, self).__init__(embed_dim=embed_dim, num_heads=1, num_layers=1, dropout=dropout, residual=True)

class AttentionMatcher(torch.nn.Module):
    def __init__(self, in_dim, embed_dim=256, num_heads=8, num_layers=[4, 4], dropout=0.0, fourier=0, self_only=False, matcher_only=False, symmetric=True):
        num_self_layers = num_layers[0]
        num_cross_layers = num_layers[1]
        in_dim = in_dim + (in_dim*fourier)
        super(AttentionMatcher, self).__init__()

        # CROSS ATTENTION MATCHER (Section 4.3): l_cross-1 multi-head layers + 1 single-head final layer.
        # Weight-shared between the A->B and B->A passes (Figure 1 caption: "weight sharing").
        self.single_cross_layer = num_cross_layers <= 1
        if not self.single_cross_layer:
            self.multi_head_cross_layers = Transformer(embed_dim, num_heads, num_cross_layers-1, dropout=dropout, residual=True)
        self.point_matcher = PointMatcher(embed_dim=embed_dim, dropout=dropout)

        # SELF ATTENTION FEATURE EXTRACTOR (Section 4.2), also weight-shared between A and B.
        # matcher_only=True reproduces the Table 2/3 ablation rows that skip the feature extractor
        # entirely (x, y are then whatever was passed in, e.g. raw coordinates or precomputed
        # DiffusionNet/Diff3f features computed externally).
        self.matcher_only = matcher_only
        if matcher_only:
            self.feature_extractor = torch.nn.Identity()
        else:
            self.feature_extractor = PointFeatureExtractor(in_dim, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_self_layers, dropout=dropout)

        self.embed_dim = embed_dim
        self.self_only = self_only
        self.symmetric = symmetric
        self.fourier = FourierFeatures(fourier) if fourier > 0 else None

    def forward(self, x, y, return_hiddens=False):
        if self.fourier:
            x = torch.cat((x, self.fourier(x)), dim=-1)
            y = torch.cat((y, self.fourier(y)), dim=-1)

        if self.matcher_only:
            x_attn_hiddens = None
            y_attn_hiddens = None
        else:
            x, x_attn_hiddens = self.feature_extractor(x, return_hiddens=True)
            y, y_attn_hiddens = self.feature_extractor(y, return_hiddens=True)

        if self.self_only:
            res = x@y.transpose(-1,-2)
            matcher_attn_hiddens = None
            matcher_attn_hiddens_sym = None
        else:
            x_cross = x if self.single_cross_layer else self.multi_head_cross_layers(x, y, return_hiddens=False)
            out, matcher_attn_hiddens = self.point_matcher.forward(x_cross, y, return_hiddens=True)

            if self.symmetric:
                y_cross = y if self.single_cross_layer else self.multi_head_cross_layers(y, x, return_hiddens=False)
                out_sym, matcher_attn_hiddens_sym = self.point_matcher.forward(y_cross, x, return_hiddens=True)
                res = matcher_attn_hiddens[-1]["pre_softmax_attn"].squeeze(1)/2 + matcher_attn_hiddens_sym[-1]["pre_softmax_attn"].transpose(-1,-2).squeeze(1)/2
            else:
                matcher_attn_hiddens_sym = None
                res = matcher_attn_hiddens[-1]["pre_softmax_attn"].squeeze(1)

        hiddens = {
            "x_intermediate" : x,
            "y_intermediate" : y,
            "x_attn_hiddens" : x_attn_hiddens,
            "y_attn_hiddens" : y_attn_hiddens,
            "matcher_attn_hiddens" : matcher_attn_hiddens,
            "matcher_attn_hiddens_sym" : matcher_attn_hiddens_sym,
        }
        if return_hiddens:
            return res, hiddens
        else:
            return res

    def self_layers(self):
        return torch.nn.ModuleList([self.feature_extractor])

    def self_parameters(self):
        params = []
        for layer in self.self_layers():
            params = params + list(layer.parameters())
        return params

if __name__ == "__main__":
    model = AttentionMatcher(3)
    x = torch.tensor([
        [
            [0.2, 1.0, 0.3],
            [3.5, 1.0, 0.3],
            [3.5, 1.0, 0.2]
        ],
        [
            [0.1, 2.0, 0.3],
            [3.5, 1.0, 0.6],
            [3.4, 5.0, 0.2]
        ]
    ])
    y = torch.rand_like(x)
    out, hiddens = model(x, y, return_hiddens=True)
    print(out)
