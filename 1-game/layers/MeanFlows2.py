import torch
import torch.nn.functional as F
from functools import partial
import numpy as np
import math
from torch import nn



class GraphNormalizer:
    def __init__(self, mode='minmax', mean=None, std=None):
        assert mode in ['minmax', 'mean_std'], "mode must be 'minmax' or 'mean_std'"
        self.mode = mode

        if mode == 'mean_std':
            if mean is None or std is None:
                raise ValueError("mean and std must be provided for 'mean_std' mode")
            self.mean = torch.tensor(mean).view(1, -1)
            self.std = torch.tensor(std).view(1, -1)

    @classmethod
    def from_list(cls, config):
        mode, mean, std = config
        return cls(mode, mean, std)

    def norm(self, x):
        if self.mode == 'minmax':
            return x * 2 - 1
        elif self.mode == 'mean_std':
            return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnorm(self, x):
        if self.mode == 'minmax':
            x = x.clip(-1, 1)
            return (x + 1) * 0.5
        elif self.mode == 'mean_std':
            return x * self.std.to(x.device) + self.mean.to(x.device)


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss_graph(error, gamma=0.5, c=1e-3):
    delta_sq = torch.mean(error ** 2, dim=1, keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean()


class GraphMeanFlow:
    def __init__(
        self,
        embed_dim=64,
        num_nodes=None,
        normalizer=['minmax', None, None],
        flow_ratio=0.50,
        time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=0.10,
        cfg_scale=2.0,
        cfg_uncond='v',
        jvp_api='autograd',
        semantic_loss_weight=1.0,  # 语义转换损失权重
    ):
        self.embed_dim = embed_dim
        self.num_nodes = num_nodes
        self.normer = GraphNormalizer.from_list(normalizer)

        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.cfg_ratio = cfg_ratio
        self.w = cfg_scale
        self.cfg_uncond = cfg_uncond
        self.semantic_loss_weight = semantic_loss_weight


        assert jvp_api in ['funtorch', 'autograd'], "jvp_api must be 'funtorch' or 'autograd'"
        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif jvp_api == 'autograd':
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True

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

    def training_losses2(self, usr_denoiser, usrEmbeds, h_usrEmbeds, ancs):

        batch_size = h_usrEmbeds.shape[0]
        device = h_usrEmbeds.device


        t, r = self.sample_t_r(batch_size, device)
        t_ = t.view(-1, 1).detach()
        r_ = r.view(-1, 1).detach()


        e = torch.randn_like(h_usrEmbeds)
        x_norm = self.normer.norm(h_usrEmbeds)
        z = (1 - t_) * x_norm + t_ * e
        v = e - x_norm


        if self.w is not None:

            with torch.no_grad():
                u_t = usr_denoiser(z, t, mess_dropout=False)  # 无条件预测
            v_hat = self.w * v + (1 - self.w) * u_t


            if self.cfg_uncond == 'v':
                cfg_mask = (torch.rand_like(t.float()) < self.cfg_ratio).view(-1, 1)
                v_hat = torch.where(cfg_mask, v, v_hat)
        else:
            v_hat = v


        model_partial = partial(usr_denoiser, mess_dropout=False)
        jvp_args = (
            lambda z, t, r: model_partial(z, (t + r).float()),  # 图去噪器输入：z + 时间嵌入(t+r)
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )


        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)


        dudt = dudt[0]
        u_tgt = v_hat - (t_ - r_) * dudt


        error = u - stopgrad(u_tgt)
        recon_loss = adaptive_l2_loss_graph(error)


        diff_usrEmbeds = self.normer.unnorm(u)
        semantic_loss = F.mse_loss(diff_usrEmbeds[ancs], usrEmbeds[ancs])


        u_diff_loss = recon_loss + self.semantic_loss_weight * semantic_loss


        u_diff_loss = u_diff_loss[ancs].mean()

        return u_diff_loss, diff_usrEmbeds



if __name__ == "__main__":

    class Denoise(nn.Module):
        def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
            super(Denoise, self).__init__()
            self.in_dims = in_dims
            self.out_dims = out_dims
            self.time_emb_dim = emb_size
            self.norm = norm
            self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
            out_dims_temp = self.out_dims
            self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
            self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
            self.drop = nn.Dropout(dropout)
            self.init_weights()

        def init_weights(self):
            for layer in self.in_layers:
                size = layer.weight.size()
                std = np.sqrt(2.0 / (size[0] + size[1]))
                layer.weight.data.normal_(0.0, std)
                layer.bias.data.normal_(0.0, 0.001)
            for layer in self.out_layers:
                size = layer.weight.size()
                std = np.sqrt(2.0 / (size[0] + size[1]))
                layer.weight.data.normal_(0.0, std)
                layer.bias.data.normal_(0.0, 0.001)
            size = self.emb_layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            self.emb_layer.weight.data.normal_(0.0, std)
            self.emb_layer.bias.data.normal_(0.0, 0.001)

        def forward(self, x, timesteps, mess_dropout=True):
            freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).to(x.device)
            temp = timesteps[:, None].float() * freqs[None]
            time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
            if self.time_emb_dim % 2:
                time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
            emb = self.emb_layer(time_emb)
            if self.norm:
                x = F.normalize(x)
            if mess_dropout:
                x = self.drop(x)
            h = torch.cat([x, emb], dim=-1)
            for i, layer in enumerate(self.in_layers):
                h = layer(h)
                h = torch.tanh(h)
            for i, layer in enumerate(self.out_layers):
                h = layer(h)
                if i != len(self.out_layers) - 1:
                    h = torch.tanh(h)
            return h


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_meanflow = GraphMeanFlow(
        embed_dim=64,
        num_nodes=1000,
        flow_ratio=0.5,
        time_dist=['lognorm', -0.4, 1.0],
        cfg_scale=2.0,
        jvp_api='autograd'
    ).to(device)


    N = 1000
    D = 64
    usr_denoiser = Denoise(in_dims=[D + 64, 128, 64], out_dims=[64, 128, D], emb_size=64).to(device)
    usrEmbeds = torch.randn(N, D).to(device)
    h_usrEmbeds = torch.randn(N, D).to(device)
    ancs = torch.randint(0, N, (200,)).to(device)


    u_diff_loss, diff_usrEmbeds = graph_meanflow.training_losses2(
        usr_denoiser, usrEmbeds, h_usrEmbeds, ancs
    )

    print(f"重构损失: {u_diff_loss.item():.4f}")
    print(f"重构用户嵌入形状: {diff_usrEmbeds.shape}")  # (1000, 64)