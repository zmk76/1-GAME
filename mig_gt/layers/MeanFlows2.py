import torch
import torch.nn.functional as F
from functools import partial
import numpy as np
import math
from torch import nn


# 适配图数据的归一化器（二维数据：节点数×嵌入维度）
class GraphNormalizer:
    def __init__(self, mode='minmax', mean=None, std=None):
        assert mode in ['minmax', 'mean_std'], "mode must be 'minmax' or 'mean_std'"
        self.mode = mode

        if mode == 'mean_std':
            if mean is None or std is None:
                raise ValueError("mean and std must be provided for 'mean_std' mode")
            self.mean = torch.tensor(mean).view(1, -1)  # 适配(N, D)维度
            self.std = torch.tensor(std).view(1, -1)

    @classmethod
    def from_list(cls, config):
        mode, mean, std = config
        return cls(mode, mean, std)

    def norm(self, x):
        """归一化图嵌入 (N, D)"""
        if self.mode == 'minmax':
            return x * 2 - 1
        elif self.mode == 'mean_std':
            return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnorm(self, x):
        """逆归一化图嵌入 (N, D)"""
        if self.mode == 'minmax':
            x = x.clip(-1, 1)
            return (x + 1) * 0.5
        elif self.mode == 'mean_std':
            return x * self.std.to(x.device) + self.mean.to(x.device)


def stopgrad(x):
    """停止梯度传播"""
    return x.detach()


def adaptive_l2_loss_graph(error, gamma=0.5, c=1e-3):
    """
    适配图数据的自适应L2损失（MeanFlow核心损失）
    Args:
        error: 图嵌入误差 (N, D)
        gamma: 原论文的γ参数
        c: 稳定性常数
    Returns:
        标量损失
    """
    delta_sq = torch.mean(error ** 2, dim=1, keepdim=False)  # (N,) 按节点计算均方误差
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean()


class GraphMeanFlow:
    def __init__(
        self,
        embed_dim=64,  # 图嵌入维度
        num_nodes=None,  # 节点数（推荐系统中为用户数）
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

        # MeanFlow核心参数
        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.cfg_ratio = cfg_ratio
        self.w = cfg_scale
        self.cfg_uncond = cfg_uncond
        self.semantic_loss_weight = semantic_loss_weight

        # JVP接口适配（自动微分计算梯度，体现平均速度）
        assert jvp_api in ['funtorch', 'autograd'], "jvp_api must be 'funtorch' or 'autograd'"
        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif jvp_api == 'autograd':
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True

    def sample_t_r(self, batch_size, device):
        """
        采样t和r（MeanFlow核心：r ≤ t，部分样本r=t）
        适配图数据的batch_size（节点数）
        """
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # Sigmoid映射到[0,1]

        # 保证t ≥ r
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        # 部分样本设置r=t（flow_ratio控制比例）
        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    def training_losses2(self, usr_denoiser, usrEmbeds, h_usrEmbeds, ancs):
        """
        兼容图扩散的调用方式：
        Args:
            usr_denoiser: 图去噪器（MeanFlow的反向过程模型）
            usrEmbeds: 语义嵌入 (N, D)，用于计算语义转换损失
            h_usrEmbeds: 原始图嵌入 (N, D)，MeanFlow处理的核心数据
            ancs: 图锚点（推荐系统用户锚点），用于筛选损失计算样本
        Returns:
            u_diff_loss: 重构损失（含语义损失）
            diff_usrEmbeds: 重构的用户嵌入 (N, D)
        """
        batch_size = h_usrEmbeds.shape[0]  # 节点数
        device = h_usrEmbeds.device

        # 1. 采样t/r（MeanFlow核心：时间步，替代图扩散的ts）
        t, r = self.sample_t_r(batch_size, device)
        t_ = t.view(-1, 1).detach()  # (N, 1) 适配图嵌入维度
        r_ = r.view(-1, 1).detach()

        # 2. 正向加噪（MeanFlow的z = (1-t)x + tε，适配图数据）
        e = torch.randn_like(h_usrEmbeds)  # 噪声 (N, D)
        x_norm = self.normer.norm(h_usrEmbeds)  # 归一化图嵌入
        z = (1 - t_) * x_norm + t_ * e  # 加噪后的图嵌入
        v = e - x_norm  # 速度项

        # 3. CFG（Classifier-Free Guidance）适配（可选，兼容原MeanFlow）
        if self.w is not None:
            # 无条件指导（适配图去噪器）
            with torch.no_grad():
                u_t = usr_denoiser(z, t, mess_dropout=False)  # 无条件预测
            v_hat = self.w * v + (1 - self.w) * u_t

            # 按掩码替换无条件样本的v_hat
            if self.cfg_uncond == 'v':
                cfg_mask = (torch.rand_like(t.float()) < self.cfg_ratio).view(-1, 1)
                v_hat = torch.where(cfg_mask, v, v_hat)
        else:
            v_hat = v

        # 4. JVP计算dudt（核心：平均速度替代瞬时速度）
        # 构造图去噪器的partial函数（适配JVP输入）
        model_partial = partial(usr_denoiser, mess_dropout=False)
        jvp_args = (
            lambda z, t, r: model_partial(z, (t + r).float()),  # 图去噪器输入：z + 时间嵌入(t+r)
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )

        # 执行JVP计算梯度（dudt：模型输出对时间的导数，体现速度）
        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)

        # 5. 计算目标输出u_tgt（MeanFlow核心：平均速度）
        dudt = dudt[0]  # 提取z对应的梯度项 (N, D)
        u_tgt = v_hat - (t_ - r_) * dudt  # 平均速度替代瞬时速度

        # 6. 计算重构损失（自适应L2损失）
        error = u - stopgrad(u_tgt)
        recon_loss = adaptive_l2_loss_graph(error)

        # 7. 计算语义转换损失（usrEmbeds与重构嵌入的MSE）
        diff_usrEmbeds = self.normer.unnorm(u)  # 逆归一化得到重构嵌入
        semantic_loss = F.mse_loss(diff_usrEmbeds[ancs], usrEmbeds[ancs])

        # 8. 总损失（重构损失 + 语义损失）
        u_diff_loss = recon_loss + self.semantic_loss_weight * semantic_loss

        # 仅返回锚点筛选后的损失（兼容原扩散逻辑）
        u_diff_loss = u_diff_loss[ancs].mean()

        return u_diff_loss, diff_usrEmbeds


# ------------------------------ 测试用例（适配你的图扩散调用）------------------------------
if __name__ == "__main__":
    # 1. 初始化图去噪器（与你的Denoise类一致）
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

    # 2. 初始化GraphMeanFlow
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_meanflow = GraphMeanFlow(
        embed_dim=64,
        num_nodes=1000,
        flow_ratio=0.5,
        time_dist=['lognorm', -0.4, 1.0],
        cfg_scale=2.0,
        jvp_api='autograd'
    ).to(device)

    # 3. 构造测试数据（模拟推荐系统图数据）
    N = 1000  # 用户数（节点数）
    D = 64   # 嵌入维度
    usr_denoiser = Denoise(in_dims=[D + 64, 128, 64], out_dims=[64, 128, D], emb_size=64).to(device)
    usrEmbeds = torch.randn(N, D).to(device)  # 语义嵌入
    h_usrEmbeds = torch.randn(N, D).to(device)  # 图嵌入
    ancs = torch.randint(0, N, (200,)).to(device)  # 锚点（筛选200个用户）

    # 4. 调用（完全兼容你的图扩散调用方式）
    u_diff_loss, diff_usrEmbeds = graph_meanflow.training_losses2(
        usr_denoiser, usrEmbeds, h_usrEmbeds, ancs
    )

    print(f"重构损失: {u_diff_loss.item():.4f}")
    print(f"重构用户嵌入形状: {diff_usrEmbeds.shape}")  # (1000, 64)