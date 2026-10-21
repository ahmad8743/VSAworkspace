import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

s = 1 / np.sqrt(10)

P1 = s * np.array([
    [1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,1]
])

P2 = s * np.array([
    [ 0,-1, 0, 0, 0, 0, 0, 0],
    [ 1, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 1, 0, 0, 0, 0],
    [ 0, 0,-1, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 1, 0, 0],
    [ 0, 0, 0, 0,-1, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0,-1],
    [ 0, 0, 0, 0, 0, 0, 1, 0]
])

P3 = s * np.array([
    [ 0, 0,-1, 0, 0, 0, 0, 0],
    [ 0, 0, 0,-1, 0, 0, 0, 0],
    [ 1, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 1, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 1, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 1],
    [ 0, 0, 0, 0,-1, 0, 0, 0],
    [ 0, 0, 0, 0, 0,-1, 0, 0]
])

P4 = s * np.array([
    [ 0, 0, 0,-1, 0, 0, 0, 0],
    [ 0, 0, 1, 0, 0, 0, 0, 0],
    [ 0,-1, 0, 0, 0, 0, 0, 0],
    [ 1, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 1],
    [ 0, 0, 0, 0, 0, 0,-1, 0],
    [ 0, 0, 0, 0, 0, 1, 0, 0],
    [ 0, 0, 0, 0,-1, 0, 0, 0]
])

P5 = s * np.array([
    [ 0, 0, 0, 0,-1, 0, 0, 0],
    [ 0, 0, 0, 0, 0,-1, 0, 0],
    [ 0, 0, 0, 0, 0, 0,-1, 0],
    [ 0, 0, 0, 0, 0, 0, 0,-1],
    [ 1, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 1, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 1, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 1, 0, 0, 0, 0]
])

P6 = s * np.array([
    [ 0, 0, 0, 0, 0,-1, 0, 0],
    [ 0, 0, 0, 0, 1, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0,-1],
    [ 0, 0, 0, 0, 0, 0, 1, 0],
    [ 0,-1, 0, 0, 0, 0, 0, 0],
    [ 1, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0,-1, 0, 0, 0, 0],
    [ 0, 0, 1, 0, 0, 0, 0, 0]
])

P7 = s * np.array([
    [ 0, 0, 0, 0, 0, 0,-1, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 1],
    [ 0, 0, 0, 0, 1, 0, 0, 0],
    [ 0, 0, 0, 0, 0,-1, 0, 0],
    [ 0, 0,-1, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 1, 0, 0, 0, 0],
    [ 1, 0, 0, 0, 0, 0, 0, 0],
    [ 0,-1, 0, 0, 0, 0, 0, 0]
])

P8 = s * np.array([
    [ 0, 0, 0, 0, 0, 0, 0,-1],
    [ 0, 0, 0, 0, 0, 0,-1, 0],
    [ 0, 0, 0, 0, 0, 1, 0, 0],
    [ 0, 0, 0, 0, 1, 0, 0, 0],
    [ 0, 0, 0,-1, 0, 0, 0, 0],
    [ 0, 0,-1, 0, 0, 0, 0, 0],
    [ 0, 1, 0, 0, 0, 0, 0, 0],
    [ 1, 0, 0, 0, 0, 0, 0, 0]
])

P = np.array([P1, P2, P3, P4, P5, P6, P7, P8])
P_TENSOR = torch.tensor(P, dtype=torch.float32)


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

class HRROctonionBinding(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("P", P_TENSOR)

    def forward(self, x, y):
        # x, y: (..., D) with D a multiple of 8
        *batch_shape, D = x.shape
        a_blocks = x.reshape(*batch_shape, D // 8, 8)
        b_blocks = y.reshape(*batch_shape, D // 8, 8)

        c_blocks = torch.einsum(
            '...ni,ijk,...nj->...nk',
            a_blocks, self.P.to(dtype=x.dtype), b_blocks
        )

        return c_blocks.reshape(*batch_shape, D)

class HRROctonionUnbinding(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("P", P_TENSOR)

    def forward(self, memory, query):
        D = query.shape[-1]

        q_blocks = query.reshape(
            *query.shape[:-1],
            D // 8,
            8
        )

        m_blocks = memory.reshape(
            *memory.shape[:-1],
            D // 8,
            8
        )

        m_blocks = m_blocks.expand(
            *query.shape[:-1],
            D // 8,
            8
        )

        a_hat_blocks = torch.einsum(
            'ijk,...nj,...nk->...ni',
            self.P.to(dtype=query.dtype),
            q_blocks,
            m_blocks
        )

        return a_hat_blocks.reshape(
            *query.shape[:-1],
            D
        )
    
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

        memory = bound.sum(dim=-2, keepdim=True)

        # [B, H, 1, Dh]

        # -------------------------
        # query memory
        # -------------------------

        vp = self.unbinding(memory,q)
        # vp = self.merge_heads(vp)
        # return self.output(vp)

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

class HRRFormerOctonionMixer(nn.Module):

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

        self.binding = HRROctonionBinding()
        self.unbinding = HRROctonionUnbinding()

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

        memory = bound.sum(dim=-2, keepdim=True)

        # [B, H, 1, Dh]

        # -------------------------
        # query memory
        # -------------------------

        vp = self.unbinding(memory,q)
        vp = self.merge_heads(vp)
        return self.output(vp)

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


class MyHRRFormerMixer(nn.Module):

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

        memory = bound.sum(dim=-2, keepdim=True)

        # [B, H, 1, Dh]

        # -------------------------
        # query memory
        # -------------------------

        vp = self.unbinding(memory,q)
        vp = self.merge_heads(vp)
        return self.output(vp)

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
