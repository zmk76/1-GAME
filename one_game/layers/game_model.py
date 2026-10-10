import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
import numpy as np
import dgl
import dgl.sampling
import torch.distributions as D

from Params import p_args
from one_game.layers.OneStepGM import GraphMeanFlow, Denoise
from one_game.layers.common import MyLinear, MyMLP, get_activation
from one_game.layers.GNNBackbone import MGDCF

device = torch.device(f'cuda:{p_args.gpu}' if torch.cuda.is_available() else 'cpu')


# Long-path Random Walk
class RandomWalkSampler:
    def __init__(self, g, num_steps=2):
        self.g = g
        self.num_steps = num_steps

    def sample(self, start_ids, start_type='user'):
        traces = dgl.sampling.random_walk(self.g, start_ids, length=self.num_steps)[0]
        return traces[:, -1]

# Adversarial Reinforcement Learning
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.shared = nn.Sequential(*layers)
        self.fc_mean = nn.Linear(dims[-1], output_dim)
        self.fc_logstd = nn.Linear(dims[-1], output_dim)
        nn.init.zeros_(self.fc_logstd.weight)
        nn.init.constant_(self.fc_logstd.bias, 0.0)

    def forward(self, z):
        h = self.shared(z)
        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h).clamp(-5, 2)
        return mean, log_std


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        dims = [input_dim] + hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dims):
        super().__init__()
        dims = [state_dim] + hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state).squeeze(-1)


def ppo_update(policy_net, value_net, opt_p, opt_v,
               states, actions, old_log_probs, rewards, dones, advantages,
               clip_eps=0.2, epochs=5, batch_size=64):
    N = states.size(0)
    for _ in range(epochs):
        idx = torch.randperm(N)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            b_idx = idx[start:end]
            s = states[b_idx]
            a = actions[b_idx]
            old_lp = old_log_probs[b_idx]
            adv = advantages[b_idx]

            mean, log_std = policy_net(s)
            std = torch.exp(log_std)
            dist = D.Normal(mean, std)
            new_log_prob = dist.log_prob(a).sum(dim=-1)
            ratio = torch.exp(new_log_prob - old_lp)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            values = value_net(s)
            value_loss = F.mse_loss(values, rewards[b_idx])

            opt_p.zero_grad()
            policy_loss.backward()
            opt_p.step()

            opt_v.zero_grad()
            value_loss.backward()
            opt_v.step()



class ONEGAME(nn.Module):
    def __init__(self,
                 k_e, k_t, k_v,
                 alpha, beta,
                 input_feat_drop_rate,
                 feat_drop_rate,
                 user_x_drop_rate,
                 item_x_drop_rate,
                 edge_drop_rate, z_drop_rate,
                 user_in_channels=None,
                 item_v_in_channels=None,
                 item_v_hidden_channels_list=None,
                 item_t_in_channels=None,
                 item_t_hidden_channels_list=None,
                 bn=True,
                 num_clusters=0,
                 num_samples=0,
                 use_dual=False,
                 flow_ratio=0.5,
                 use_gen=False,
                 walk_steps=7,
                 *args, **kwargs):
        super().__init__()

        self.use_gen = use_gen
        self.walk_steps = walk_steps

        self.emb_mgdcf = MGDCF(k_e, alpha, beta, 0.0, edge_drop_rate, 0.0) if k_e >= 0 else None
        self.t_mgdcf = MGDCF(k_t, alpha, beta, 0.0, edge_drop_rate, 0.0) if k_t >= 0 else None
        self.v_mgdcf = MGDCF(k_v, alpha, beta, 0.0, edge_drop_rate, 0.0) if k_v >= 0 else None

        self.z_dropout = nn.Dropout(z_drop_rate)
        self.user_gnn_input_dropout = nn.Dropout(user_x_drop_rate)
        self.item_gnn_input_dropout = nn.Dropout(item_x_drop_rate)

        self.use_dual = use_dual
        self.num_samples = num_samples

        self.input_feat_dropout = nn.Dropout(input_feat_drop_rate)

        self.t_ff = nn.Sequential(
            MyMLP(
                item_t_in_channels,
                item_t_hidden_channels_list,
                activation="prelu",
                drop_rate=feat_drop_rate,
                bn=True,
                output_activation="prelu",
                output_drop_rate=0.0,
                output_bn=True,
            )
        )
        self.v_ff = nn.Sequential(
            MyMLP(
                item_v_in_channels,
                item_v_hidden_channels_list,
                activation="prelu",
                drop_rate=feat_drop_rate,
                bn=True,
                output_activation="prelu",
                output_drop_rate=0.0,
                output_bn=True
            )
        )

        self.graph_meanflow = GraphMeanFlow(
            embed_dim=user_in_channels,
            normalizer=['mean_std', 0.0, 1.0],
            flow_ratio=flow_ratio,
            time_dist=['lognorm', -0.4, 1.0],
            semantic_loss_weight=1.0,
        ).to(device)

        self.meanflows_denoiser_visual = Denoise(
            in_dims=[user_in_channels, 128, 256],
            out_dims=[256, 128, user_in_channels],
            emb_size=user_in_channels
        ).to(device)

        self.meanflows_denoiser_textual = Denoise(
            in_dims=[user_in_channels, 128, 256],
            out_dims=[256, 128, user_in_channels],
            emb_size=user_in_channels
        ).to(device)

        self.user_in_channels = user_in_channels

        if self.use_gen:
            self.generator = Generator(self.user_in_channels, [128, 256], self.user_in_channels).to(device)
            self.discriminator = Discriminator(self.user_in_channels, [128, 64]).to(device)
            self.value_net = ValueNetwork(self.user_in_channels, [128, 64]).to(device)
            self.gen_opt = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
            self.disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
            self.val_opt = torch.optim.Adam(self.value_net.parameters(), lr=1e-4)
            self.fusion_weights = nn.Parameter(torch.ones(2, device=device) / 2)

            self.walk_encoder_user = nn.Linear(self.user_in_channels, self.user_in_channels).to(device)
            self.walk_encoder_item = nn.Linear(self.user_in_channels, self.user_in_channels).to(device)
            nn.init.xavier_uniform_(self.walk_encoder_user.weight)
            nn.init.xavier_uniform_(self.walk_encoder_item.weight)

            self.walk_sampler = None
            self.v_feat = None
            self.t_feat = None
            self.num_users = None
            self.num_items = None
            self.all_interaction_set = None


            self.neg_item_ids = None
            self.neg_user_ids = None


    def _sample_negative_ids(self, u, i):
        import random
        max_tries = 10
        graph_device = self.g.device
        neg_item = None
        for _ in range(max_tries):
            neg_node = self.walk_sampler.sample(torch.tensor([u], device=graph_device), start_type='user')[0]
            if neg_node < 0:
                neg_node = u + self.num_users
            neg_item_candidate = neg_node - self.num_users
            if neg_item_candidate < 0 or neg_item_candidate >= self.num_items:
                neg_item_candidate = random.randint(0, self.num_items - 1)
            if (int(u), int(neg_item_candidate)) not in self.all_interaction_set:
                neg_item = neg_item_candidate
                break
        if neg_item is None:
            while True:
                cand = random.randint(0, self.num_items - 1)
                if (int(u), cand) not in self.all_interaction_set:
                    neg_item = cand
                    break

        neg_user = None
        for _ in range(max_tries):
            neg_node = self.walk_sampler.sample(torch.tensor([i + self.num_users], device=graph_device), start_type='item')[0]
            if neg_node < 0:
                neg_node = i
            neg_user_candidate = neg_node
            if neg_user_candidate < 0 or neg_user_candidate >= self.num_users:
                neg_user_candidate = random.randint(0, self.num_users - 1)
            if (int(neg_user_candidate), int(i)) not in self.all_interaction_set:
                neg_user = neg_user_candidate
                break
        if neg_user is None:
            while True:
                cand = random.randint(0, self.num_users - 1)
                if (cand, int(i)) not in self.all_interaction_set:
                    neg_user = cand
                    break

        return neg_item, neg_user

    def precompute_negatives(self, train_edges):
        if not self.use_gen:
            return
        if torch.is_tensor(train_edges):
            train_edges = train_edges.cpu().numpy()
        num_edges = len(train_edges)
        neg_items = []
        neg_users = []
        print(f"Precomputing negative samples for {num_edges} training edges...")
        for idx, (u, i) in enumerate(train_edges):
            neg_i, neg_u = self._sample_negative_ids(int(u), int(i))
            neg_items.append(neg_i)
            neg_users.append(neg_u)
            if (idx + 1) % 5000 == 0:
                print(f"  processed {idx+1}/{num_edges}")
        self.neg_item_ids = torch.tensor(neg_items, dtype=torch.long, device=device)
        self.neg_user_ids = torch.tensor(neg_users, dtype=torch.long, device=device)
        print("Precomputation done.")

    def set_graph(self, g, v_feat, t_feat, num_users, num_items, all_edges=None, train_edges=None):
        if self.use_gen:
            self.g = g
            self.v_feat = v_feat
            self.t_feat = t_feat
            self.num_users = num_users
            self.num_items = num_items
            self.walk_sampler = RandomWalkSampler(g, num_steps=self.walk_steps)

            self.all_interaction_set = set()
            if all_edges is not None:
                edges = all_edges.cpu().numpy() if torch.is_tensor(all_edges) else all_edges
                for u, i in edges:
                    self.all_interaction_set.add((int(u), int(i)))

            if train_edges is not None:
                self.precompute_negatives(train_edges)

    def sample_negative_bidirectional(self, user_ids, item_ids, user_embeddings):
        max_tries = 10
        device = user_ids.device
        batch_size = len(user_ids)

        neg_item_ids = None
        for _ in range(max_tries):
            neg_item_node_ids = self.walk_sampler.sample(user_ids, start_type='user')
            neg_item_node_ids = torch.where(neg_item_node_ids < 0, user_ids + self.num_users, neg_item_node_ids)
            neg_item_ids_temp = neg_item_node_ids - self.num_users
            neg_item_ids_temp = torch.clamp(neg_item_ids_temp, 0, self.num_items - 1)
            bad = False
            for j, u in enumerate(user_ids):
                if (u.item(), neg_item_ids_temp[j].item()) in self.all_interaction_set:
                    bad = True
                    break
            if not bad:
                neg_item_ids = neg_item_ids_temp
                break
        if neg_item_ids is None:
            neg_item_ids = torch.randint(0, self.num_items, (batch_size,), device=device)
            for j, u in enumerate(user_ids):
                while (u.item(), neg_item_ids[j].item()) in self.all_interaction_set:
                    neg_item_ids[j] = torch.randint(0, self.num_items, (1,), device=device).item()

        neg_user_ids = None
        item_node_ids = item_ids + self.num_users
        for _ in range(max_tries):
            neg_user_node_ids = self.walk_sampler.sample(item_node_ids, start_type='item')
            neg_user_node_ids = torch.where(neg_user_node_ids < 0, item_ids, neg_user_node_ids)
            neg_user_ids_temp = torch.clamp(neg_user_node_ids, 0, self.num_users - 1)
            bad = False
            for j, i in enumerate(item_ids):
                if (neg_user_ids_temp[j].item(), i.item()) in self.all_interaction_set:
                    bad = True
                    break
            if not bad:
                neg_user_ids = neg_user_ids_temp
                break
        if neg_user_ids is None:
            neg_user_ids = torch.randint(0, self.num_users, (batch_size,), device=device)
            for j, i in enumerate(item_ids):
                while (neg_user_ids[j].item(), i.item()) in self.all_interaction_set:
                    neg_user_ids[j] = torch.randint(0, self.num_users, (1,), device=device).item()

        w_v, w_t = torch.softmax(self.fusion_weights, dim=0)
        neg_item_v_proj = self.v_ff(self.v_feat[neg_item_ids])
        neg_item_t_proj = self.t_ff(self.t_feat[neg_item_ids])
        neg_item_feat = w_v * neg_item_v_proj + w_t * neg_item_t_proj
        neg_item_encoded = self.walk_encoder_item(neg_item_feat)

        neg_user_emb = user_embeddings[neg_user_ids]
        neg_user_encoded = self.walk_encoder_user(neg_user_emb)

        return neg_item_encoded, neg_user_encoded

    def forward(self, g, user_embeddings, item_v_feat, item_t_feat,
                item_embeddings=None, batch_edges=None, batch_indices=None, return_all=False):
        item_v_feat = self.input_feat_dropout(item_v_feat)
        item_t_feat = self.input_feat_dropout(item_t_feat)

        encoded_t = self.t_ff(item_t_feat)
        encoded_v = self.v_ff(item_v_feat)

        num_users = user_embeddings.size(0)
        user_h = user_embeddings

        if self.emb_mgdcf is not None:
            emb_h = self.emb_mgdcf(
                g,
                torch.cat([
                    self.user_gnn_input_dropout(user_h),
                    torch.zeros_like(encoded_t) if item_embeddings is None else self.item_gnn_input_dropout(
                        item_embeddings)
                ], dim=0)
            )
        else:
            emb_h = None

        if self.t_mgdcf is not None:
            t_h = self.t_mgdcf(
                g,
                torch.cat([
                    torch.zeros_like(user_h),
                    self.item_gnn_input_dropout(encoded_t),
                ], dim=0)
            )
        else:
            t_h = None

        if self.v_mgdcf is not None:
            v_h = self.v_mgdcf(
                g,
                torch.cat([
                    torch.zeros_like(user_h),
                    self.item_gnn_input_dropout(encoded_v),
                ], dim=0)
            )
        else:
            v_h = None

        combined_h = None
        for h in [emb_h, t_h, v_h]:
            if h is not None:
                if combined_h is None:
                    combined_h = h
                else:
                    combined_h = combined_h + h

        combined_h = self.z_dropout(combined_h)

        num_items = item_v_feat.size(0)
        item_h = combined_h[num_users:]
        memory_index = torch.randint(0, num_items, [combined_h.size(0), self.num_samples])
        memory = item_h[memory_index]
        z_memory_h = torch.concat([combined_h.unsqueeze(1), memory], dim=1)

        if not return_all:
            return combined_h

        if self.use_gen:
            state_all = torch.randn_like(v_h)
            mean_all, log_std_all = self.generator(state_all)
            std_all = torch.exp(log_std_all)
            action_all = mean_all + std_all * torch.randn_like(mean_all)

            state_user = state_all[:num_users]
            action_user = action_all[:num_users]
            state_item = state_all[num_users:]
            action_item = action_all[num_users:]

            combined_noise = action_all

            if batch_indices is not None and self.neg_item_ids is not None and self.neg_user_ids is not None:
                neg_item_ids = self.neg_item_ids[batch_indices]
                neg_user_ids = self.neg_user_ids[batch_indices]
                w_v, w_t = torch.softmax(self.fusion_weights, dim=0)
                neg_item_v_proj = self.v_ff(self.v_feat[neg_item_ids])
                neg_item_t_proj = self.t_ff(self.t_feat[neg_item_ids])
                neg_item_feat = w_v * neg_item_v_proj + w_t * neg_item_t_proj
                neg_item_encoded = self.walk_encoder_item(neg_item_feat)
                neg_user_emb = user_embeddings[neg_user_ids]
                neg_user_encoded = self.walk_encoder_user(neg_user_emb)
            else:
                neg_item_encoded, neg_user_encoded = self.sample_negative_bidirectional(
                    batch_edges[:, 0], batch_edges[:, 1], user_embeddings
                )

            loss_vis, _ = self.graph_meanflow.training_losses2(
                self.meanflows_denoiser_visual, v_h, noise=combined_noise
            )
            loss_txt, _ = self.graph_meanflow.training_losses2(
                self.meanflows_denoiser_textual, t_h, noise=combined_noise
            )
            meanflows_loss = loss_vis.mean() + loss_txt.mean()

            meanflows_Embeds_visual = self.graph_meanflow.p_sample(
                self.meanflows_denoiser_visual, v_h, noise=combined_noise
            )
            meanflows_Embeds_textual = self.graph_meanflow.p_sample(
                self.meanflows_denoiser_textual, t_h, noise=combined_noise
            )
            user_emb = combined_h[:num_users]
            item_emb = combined_h[num_users:]
            user_emb_vis = meanflows_Embeds_visual[:num_users]
            item_emb_vis = meanflows_Embeds_visual[num_users:]
            user_emb_txt = meanflows_Embeds_textual[:num_users]
            item_emb_txt = meanflows_Embeds_textual[num_users:]
            a_idx, b_idx = batch_edges[:, 0], batch_edges[:, 1]
            cl_loss = (self.InfoNCELoss(user_emb_vis[a_idx], user_emb[a_idx], p_args.temp) +
                       self.InfoNCELoss(item_emb_vis[b_idx], item_emb[b_idx], p_args.temp) +
                       self.InfoNCELoss(user_emb_txt[a_idx], user_emb[a_idx], p_args.temp) +
                       self.InfoNCELoss(item_emb_txt[b_idx], item_emb[b_idx], p_args.temp))
            meanflows_loss += cl_loss * p_args.ssl_reg


            real_score_item = self.discriminator(neg_item_encoded.detach())
            real_score_user = self.discriminator(neg_user_encoded.detach())
            fake_score_item = self.discriminator(action_item.detach())
            fake_score_user = self.discriminator(action_user.detach())


            disc_loss = (
                    F.binary_cross_entropy_with_logits(real_score_item, torch.ones_like(real_score_item)) +
                    F.binary_cross_entropy_with_logits(real_score_user, torch.ones_like(real_score_user)) +
                    F.binary_cross_entropy_with_logits(fake_score_item, torch.zeros_like(fake_score_item)) +
                    F.binary_cross_entropy_with_logits(fake_score_user, torch.zeros_like(fake_score_user))
            )

            with torch.no_grad():
                reward_user = torch.sigmoid(self.discriminator(action_user))
                value_user = self.value_net(state_user)
                advantage_user = reward_user - value_user
                reward_item = torch.sigmoid(self.discriminator(action_item))
                value_item = self.value_net(state_item)
                advantage_item = reward_item - value_item

                reward = torch.cat([reward_user, reward_item], dim=0)
                advantage = torch.cat([advantage_user, advantage_item], dim=0)
                state = torch.cat([state_user, state_item], dim=0)
                action = torch.cat([action_user, action_item], dim=0)

            dist_all = D.Normal(mean_all, std_all)
            log_prob_all = dist_all.log_prob(action_all).sum(dim=-1)
            log_prob_user = log_prob_all[:num_users]
            log_prob_item = log_prob_all[num_users:]
            log_prob = torch.cat([log_prob_user, log_prob_item], dim=0)



            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()


            total_loss = meanflows_loss

            return (combined_h, emb_h, t_h, v_h, encoded_t, encoded_v, z_memory_h,
                    total_loss, reward, advantage, state, action, log_prob)


        else:
            meanflows_loss_visual, _ = self.graph_meanflow.training_losses2(self.meanflows_denoiser_visual, v_h)
            meanflows_loss_textual, _ = self.graph_meanflow.training_losses2(self.meanflows_denoiser_textual, t_h)
            meanflows_loss = meanflows_loss_visual.mean() + meanflows_loss_textual.mean()
            meanflows_Embeds_combined_h_visual = self.graph_meanflow.p_sample(self.meanflows_denoiser_visual, v_h)
            meanflows_Embeds_combined_h_textual = self.graph_meanflow.p_sample(self.meanflows_denoiser_textual, t_h)

            user_emb = combined_h[:num_users]
            item_emb = combined_h[num_users:]
            user_emb_visual = meanflows_Embeds_combined_h_visual[:num_users]
            item_emb_visual = meanflows_Embeds_combined_h_visual[num_users:]
            user_emb_textual = meanflows_Embeds_combined_h_textual[:num_users]
            item_emb_textual = meanflows_Embeds_combined_h_textual[num_users:]

            a_indices = batch_edges[:, 0]
            b_indices = batch_edges[:, 1]
            cl_loss = (self.InfoNCELoss(user_emb_visual[a_indices], user_emb[a_indices], p_args.temp)
                       + self.InfoNCELoss(item_emb_visual[b_indices], item_emb[b_indices], p_args.temp))
            cl_loss += (self.InfoNCELoss(user_emb_textual[a_indices], user_emb[a_indices], p_args.temp)
                        + self.InfoNCELoss(item_emb_textual[b_indices], item_emb[b_indices], p_args.temp))
            meanflows_loss += (cl_loss * p_args.ssl_reg)

            return combined_h, emb_h, t_h, v_h, encoded_t, encoded_v, z_memory_h, meanflows_loss

    def InfoNCELoss(self, z1, z2, temp):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim_matrix = z1 @ z2.T / temp
        n = z1.size(0)
        labels = torch.arange(n, device=z1.device)
        return F.cross_entropy(sim_matrix, labels)

    # Update the anisotropic agent and value network.
    def update_generator(self, states, actions, old_log_probs, rewards, dones, advantages):
        if self.use_gen:
            ppo_update(self.generator, self.value_net,
                       self.gen_opt, self.val_opt,
                       states, actions, old_log_probs, rewards, dones, advantages,
                       clip_eps=p_args.clip_eps, epochs=5, batch_size=64)