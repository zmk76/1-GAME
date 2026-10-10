import torch
import torch.nn.functional as F
from functools import partial
import numpy as np
import math
from torch import nn

class GraphNormalizer:
    def __init__(self, mode='mean_std', mean=None, std=None):
        assert mode in ['minmax', 'mean_std'], "mode must be 'minmax' or 'mean_std'"
        self.mode = mode
        if mode == 'mean_std':
            if mean is None or std is None:
                raise ValueError("mean and std must be provided")
            self.mean = torch.tensor(mean).view(1, -1)
            self.std = torch.tensor(std).view(1, -1)

    @classmethod
    def from_list(cls, config):
        return cls(*config)

    def norm(self, x):
        if self.mode == 'minmax':
            return x * 2 - 1
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnorm(self, x):
        if self.mode == 'minmax':
            return (x.clip(-1, 1) + 1) * 0.5
        return x * self.std.to(x.device) + self.mean.to(x.device)

def stopgrad(x):
    return x.detach()

def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    delta_sq = torch.mean(error ** 2, dim=1)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq
    return stopgrad(w) * loss

class GraphMeanFlow(torch.nn.Module):
    def __init__(
            self,
            embed_dim=64,
            normalizer=['mean_std', 0.0, 1.0],
            flow_ratio=0.5,
            time_dist=['lognorm', -0.4, 1.0],
            semantic_loss_weight=1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.normer = GraphNormalizer.from_list(normalizer)
        self.flow_ratio = flow_ratio
        self.semantic_loss_weight = semantic_loss_weight
        self.jvp_fn = torch.autograd.functional.jvp
        self.create_graph = True
        self.time_dist = time_dist

    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)
        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])
        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]
        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    # 1-GM training loss
    def training_losses2(self, model, x_start, targetEmbeds=None, ancs=None, noise=None):
        n = x_start.shape[0]
        device = x_start.device
        x_norm = self.normer.norm(x_start)

        t, r = self.sample_t_r(n, device)
        t_ = t.view(-1, 1)
        r_ = r.view(-1, 1)

        if noise is None:
            eps = torch.randn_like(x_norm)
        else:
            eps = noise

        z_t = (1 - t_) * x_norm + t_ * eps
        v_t = eps - x_norm

        u, dudt = self.jvp_fn(
            lambda z, t, r: model(z, t, False, r),
            (z_t, t, r),
            (v_t, torch.ones_like(t), torch.zeros_like(r)),
            create_graph=self.create_graph
        )
        dudt = dudt.view(n, -1)
        u_tgt = v_t - (t_ - r_) * dudt
        error = u - stopgrad(u_tgt)

        recon_loss = adaptive_l2_loss(error)
        total_loss = recon_loss
        graph_meanflows_loss = total_loss[ancs] if ancs is not None else total_loss
        rec_Embeds = z_t - (t_ - r_) * u
        rec_Embeds = self.normer.unnorm(rec_Embeds)
        return graph_meanflows_loss, rec_Embeds

    def mean_flat(self, tensor):
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    # Single-Step Reverse Generation: From Anisotropic Noise to Cross-Modal Contrastive Views in One Step
    @torch.no_grad()
    def p_sample(self, model, x_start, sampling_steps=None, noise=None):
        model.eval()
        num_nodes = x_start.shape[0]
        device = x_start.device

        z1 = self.normer.norm(x_start)
        t = torch.ones(num_nodes, device=device)
        r = torch.zeros_like(t)
        if noise is None:
            eps = torch.randn_like(z1)
        else:
            eps = noise
        u = model(eps, t, mess_dropout=False, r=r)
        z0 = eps - u
        return self.normer.unnorm(z0)

class Denoise(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.time_emb_dim = emb_size
        self.norm = norm

        self.time_emb = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )
        in_dims_temp = [self.in_dims[0] + 2 * self.time_emb_dim] + self.in_dims[1:]
        self.in_layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(self.out_dims[:-1], self.out_dims[1:])])
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers + self.out_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def timestep_embedding(self, t, dim):
        half_dim = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=t.device) / half_dim)
        emb = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb if dim % 2 == 0 else torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

    def forward(self, z, t, mess_dropout=True, r=None):
        if r is None:
            r = torch.zeros_like(t)
        t_emb = self.time_emb(self.timestep_embedding(t, self.time_emb_dim))
        r_emb = self.time_emb(self.timestep_embedding(r, self.time_emb_dim))
        time_emb = torch.cat([t_emb, r_emb], dim=-1)

        if self.norm:
            z = F.normalize(z)
        if mess_dropout:
            z = self.drop(z)

        h = torch.cat([z, time_emb], dim=-1)
        for layer in self.in_layers:
            h = torch.tanh(layer(h))
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        return h