import torch
import torch.nn as nn
import torch.nn.functional as F


class HRRBinding(nn.Module):
    def forward(self, x, y):
        # x, y: (..., D)

        x_fft = torch.fft.rfft(x, dim=-1)
        y_fft = torch.fft.rfft(y, dim=-1)

        bound = torch.fft.irfft(
            x_fft * y_fft,
            n=x.shape[-1],
            dim=-1
        )

        return bound


class HRRUnbinding(nn.Module):
    def forward(self, memory, query):
        # Approximate inverse via circular correlation

        m_fft = torch.fft.rfft(memory, dim=-1)
        q_fft = torch.fft.rfft(query, dim=-1)

        out = torch.fft.irfft(
            m_fft * torch.conj(q_fft),
            n=query.shape[-1],
            dim=-1
        )

        return out

class HRRFormerMixer(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        layer_idx: int = None,
        **kwargs
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.layer_idx = layer_idx

        self.query = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.key = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.value = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.output = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.binding = HRRBinding()
        self.unbinding = HRRUnbinding()

    def split_heads(self, x):

        B, T, D = x.shape

        x = x.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        )

        return x.transpose(1, 2)

    def merge_heads(self, x):

        B, H, T, Dh = x.shape

        x = x.transpose(
            1,
            2
        ).contiguous()

        return x.view(
            B,
            T,
            H * Dh
        )

    def forward(self, x):

        # -------------------------
        # projections
        # -------------------------

        q = self.split_heads(self.query(x))
        k = self.split_heads(self.key(x))
        v = self.split_heads(self.value(x))
        # q,k,v:
        # [B, H, T, Dh]

        # -------------------------
        # HRR key-value binding
        # -------------------------

        bound = self.binding(k,v)
        # [B, H, T, Dh]

        # -------------------------
        # bundle associations
        # -------------------------

        memory = bound.cumsum(dim=-2,keepdim=True)

        # [B, H, 1, Dh]

        # -------------------------
        # query memory
        # -------------------------

        vp = self.unbinding(memory,q)

        # [B, H, T, Dh]

        # -------------------------
        # similarity
        # -------------------------

        scale = F.cosine_similarity(v, vp, dim=-1).unsqueeze(-1)

        # [B, H, T, 1]

        # -------------------------
        # normalize
        # -------------------------

        weight = torch.softmax(scale,dim=-2)

        # -------------------------
        # weight values
        # -------------------------

        weighted_value = weight * v

        # [B, H, T, Dh]

        # -------------------------
        # merge heads
        # -------------------------

        weighted_value = self.merge_heads(weighted_value)

        # [B, T, D]

        return self.output(
            weighted_value
        )


