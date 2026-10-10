# coding: utf-8

import os
import json
from Params import p_args
from one_game.configs.default_config import add_arguments_by_config_class, combine_args_into_config
from one_game.configs.mm_mgdcf_default_config import MMMGDCFConfig
from one_game.configs.masked_mm_mgdcf_default_config import load_masked_mm_mgdcf_default_config
import sys
import time
import torch.nn.functional as F

from one_game.layers.game_model import ONEGAME
from one_game.vector_search.vector_search import VectorSearchEngine

config = load_masked_mm_mgdcf_default_config(p_args.dataset)
config = combine_args_into_config(config, p_args)

print(config)

os.environ["CUDA_VISIBLE_DEVICES"] = p_args.gpu

from one_game.utils.random_utils import reset_seed

reset_seed(p_args.seed)

from one_game.layers.MultimodalEncoder import MMMGDCF
import shortuuid
from one_game.layers.sign import random_project, sign_pre_compute
from one_game.losses import compute_info_bpr_loss, compute_l2_loss
from one_game.utils.data_loader_utils import create_tensors_dataloader
from one_game.evaluation.ranking import evaluate_mean_global_metrics
from one_game.layers.GNNBackbone import MGDCF
from one_game.load_data import load_data
import torch
import numpy as np
import dgl
import dgl.function as fn
import torch.nn as nn
from dataclasses import asdict

embedding_size = config.embedding_size
device = torch.device(f'cuda:{p_args.gpu}' if torch.cuda.is_available() else 'cpu')

train_user_item_edges, valid_user_item_edges, test_user_item_edges, train_user_items_dict, train_mask_user_items_dict, valid_user_items_dict, valid_mask_user_items_dict, test_user_items_dict, test_mask_user_items_dict, num_users, num_items, v_feat, t_feat = load_data(
    p_args.dataset)

all_edges = torch.cat([
    torch.tensor(train_user_item_edges, device='cpu'),
    torch.tensor(valid_user_item_edges, device='cpu'),
    torch.tensor(test_user_item_edges, device='cpu')
], dim=0)

start_time = time.time()
run_id = shortuuid.uuid()
result_dir = p_args.result_dir
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
result_path = os.path.join(result_dir, "{}.json".format(run_id))
tmp_result_path = os.path.join(result_dir, "{}.json.tmp".format(run_id))

if config.use_rp:
    v_feat = random_project(v_feat, t_feat.size(-1))

num_train_user_item_edges = len(train_user_item_edges)
g = MGDCF.build_sorted_homo_graph(train_user_item_edges, num_users=num_users, num_items=num_items).to(device)
assert g.num_edges() == num_train_user_item_edges * 2 + num_users + num_items

num_nodes = g.num_nodes()
degs = g.in_degrees().to(device)
item_item_g = None

v_feat = v_feat.to(device)
t_feat = t_feat.to(device)

user_embeddings = np.random.randn(num_users, embedding_size) / np.sqrt(embedding_size)
user_embeddings = torch.tensor(user_embeddings, dtype=torch.float32, requires_grad=True, device=device)

item_embeddings = np.random.randn(num_items, embedding_size) / np.sqrt(embedding_size)
item_embeddings = torch.tensor(item_embeddings, dtype=torch.float32, requires_grad=True, device=device)

method = p_args.method

if method == "one_game_base1":
    model = MMMGDCF(
        k_e=config.k_e, k_t=config.k_t, k_v=config.k_v,
        alpha=config.alpha, beta=config.beta,
        input_feat_drop_rate=config.input_feat_drop_rate,
        feat_drop_rate=config.feat_drop_rate,
        user_x_drop_rate=config.user_x_drop_rate,
        item_x_drop_rate=config.item_x_drop_rate,
        edge_drop_rate=config.edge_drop_rate,
        z_drop_rate=config.z_drop_rate,
        item_v_in_channels=v_feat.size(-1),
        item_v_hidden_channels_list=[config.feat_hidden_units, embedding_size],
        item_t_in_channels=t_feat.size(-1),
        item_t_hidden_channels_list=[config.feat_hidden_units, embedding_size],
        bn=config.bn,
    ).to(device)

elif method == "one_game_base2":
    model = ONEGAME(
        k_e=config.k_e, k_t=config.k_t, k_v=config.k_v,
        alpha=config.alpha, beta=config.beta,
        input_feat_drop_rate=config.input_feat_drop_rate,
        feat_drop_rate=config.feat_drop_rate,
        user_x_drop_rate=config.user_x_drop_rate,
        item_x_drop_rate=config.item_x_drop_rate,
        edge_drop_rate=config.edge_drop_rate,
        z_drop_rate=config.z_drop_rate,
        user_in_channels=config.embedding_size,
        item_v_in_channels=v_feat.size(-1),
        item_v_hidden_channels_list=[config.feat_hidden_units, embedding_size],
        item_t_in_channels=t_feat.size(-1),
        item_t_hidden_channels_list=[config.feat_hidden_units, embedding_size],
        bn=config.bn,
        num_clusters=config.num_clusters,
        num_samples=config.num_samples,
        flow_ratio=p_args.flow_ratio,
        walk_steps=p_args.walk_steps,
        use_gen=False,
    ).to(device)

elif method == "one_game":
    model = ONEGAME(
        k_e=config.k_e, k_t=config.k_t, k_v=config.k_v,
        alpha=config.alpha, beta=config.beta,
        input_feat_drop_rate=config.input_feat_drop_rate,
        feat_drop_rate=config.feat_drop_rate,
        user_x_drop_rate=config.user_x_drop_rate,
        item_x_drop_rate=config.item_x_drop_rate,
        edge_drop_rate=config.edge_drop_rate,
        z_drop_rate=config.z_drop_rate,
        user_in_channels=config.embedding_size,
        item_v_in_channels=v_feat.size(-1),
        item_v_hidden_channels_list=[config.feat_hidden_units, embedding_size],
        item_t_in_channels=t_feat.size(-1),
        item_t_hidden_channels_list=[config.feat_hidden_units, embedding_size],
        bn=config.bn,
        num_clusters=config.num_clusters,
        num_samples=config.num_samples,
        flow_ratio=p_args.flow_ratio,
        walk_steps=p_args.walk_steps,
        use_gen=True,
    ).to(device)
    model.set_graph(g, v_feat, t_feat, num_users, num_items,
                    all_edges=all_edges,
                    train_edges=train_user_item_edges)

use_clip_loss = False
use_mm_mf_loss = False

def forward(g, batch_edges=None, batch_indices=None, return_all=False):
    if return_all:
        if method == "one_game":
            outputs = model(g, user_embeddings, v_feat, t_feat,
                            item_embeddings=item_embeddings if config.use_item_emb else None,
                            batch_edges=batch_edges,
                            batch_indices=batch_indices,
                            return_all=True)

            (virtual_h, emb_h, t_h, v_h, encoded_t, encoded_v, z_memory_h,
             meanflows_loss, reward, advantage, state, action, log_prob) = outputs
            user_h = virtual_h[:num_users]
            item_h = virtual_h[num_users:]
            user_emb_h = emb_h[:num_users]
            item_emb_h = emb_h[num_users:]
            user_t_h = t_h[:num_users]
            item_t_h = t_h[num_users:]
            if v_h is not None:
                user_v_h = v_h[:num_users]
                item_v_h = v_h[num_users:]
            else:
                user_v_h = None
                item_v_h = None
            return (user_h, item_h, user_emb_h, item_emb_h, user_t_h, item_t_h,
                    user_v_h, item_v_h, encoded_t, encoded_v, z_memory_h,
                    meanflows_loss, reward, advantage, state, action, log_prob)
        else:
            outputs = model(g, user_embeddings, v_feat, t_feat,
                            item_embeddings=item_embeddings if config.use_item_emb else None,
                            batch_edges=batch_edges,
                            batch_indices=batch_indices,
                            return_all=True)
            (virtual_h, emb_h, t_h, v_h, encoded_t, encoded_v, z_memory_h, meanflows_loss) = outputs
            user_h = virtual_h[:num_users]
            item_h = virtual_h[num_users:]
            user_emb_h = emb_h[:num_users]
            item_emb_h = emb_h[num_users:]
            user_t_h = t_h[:num_users]
            item_t_h = t_h[num_users:]
            if v_h is not None:
                user_v_h = v_h[:num_users]
                item_v_h = v_h[num_users:]
            else:
                user_v_h = None
                item_v_h = None
            return (user_h, item_h, user_emb_h, item_emb_h, user_t_h, item_t_h,
                    user_v_h, item_v_h, encoded_t, encoded_v, z_memory_h, meanflows_loss)
    else:
        virtual_h = model(g, user_embeddings, v_feat, t_feat,
                          item_embeddings=item_embeddings if config.use_item_emb else None,
                          batch_edges=batch_edges,
                          batch_indices=batch_indices,
                          return_all=False)
        user_h = virtual_h[:num_users]
        item_h = virtual_h[num_users:]
        return user_h, item_h


def evaluate(user_items_dict, mask_user_items_dict):
    model.eval()
    user_h, item_h = forward(g)
    user_h = user_h.detach().cpu().numpy()
    item_h = item_h.detach().cpu().numpy()
    mean_results_dict = evaluate_mean_global_metrics(user_items_dict, mask_user_items_dict,
                                                     user_h, item_h, k_list=[10, 20],
                                                     metrics=["precision", "recall", "ndcg"])
    return mean_results_dict


def update_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        new_lr = param_group['lr'] * config.lr_decay
        if new_lr >= config.lr_decay_min:
            param_group['lr'] = new_lr


train_edges_data_loader = create_tensors_dataloader(
    torch.arange(len(train_user_item_edges)),
    torch.tensor(train_user_item_edges), batch_size=config.batch_size, shuffle=True
)

optimizer = torch.optim.Adam([user_embeddings, item_embeddings] + list(model.parameters()), lr=config.lr)

early_stop_metric = "recall@20"
best_valid_score = 0.0
early_stop_valid_results_dict = None
early_stop_test_results_dict = None
best_epoch = None

combined_config_dict = vars(p_args)
for k, v in asdict(config).items():
    combined_config_dict[k] = v

patience_count = 0
total_train_time = 0.0

run_log_dir = "run_logs"
run_log_fname = "{}.json".format(p_args.dataset)
run_log_fpath = os.path.join(run_log_dir, run_log_fname)

ppo_buffer = {
    'states': [], 'actions': [], 'rewards': [], 'advantages': [],
    'dones': [], 'old_log_probs': []
}

for epoch in range(1, config.num_epochs + 1):
    epoch_start_time = time.time()

    ppo_buffer = {
        'states': [], 'actions': [], 'rewards': [], 'advantages': [],
        'dones': [], 'old_log_probs': []
    }

    for step, (batch_edge_indices, batch_edges) in enumerate(train_edges_data_loader):
        batch_edges = batch_edges.to(device)
        batch_edge_indices = batch_edge_indices.to(device)
        model.train()
        with g.local_scope():
            new_g = g

            if method == "one_game_base1":
                user_h, item_h = forward(new_g)
                mf_losses = compute_info_bpr_loss(user_h, item_h, batch_edges, num_negs=config.num_negs,
                                                  reduction="none")
                l2_loss = compute_l2_loss([user_h, item_h])
                loss = mf_losses.sum() + l2_loss * config.l2_coef
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            else:
                if method == "one_game":
                    outputs = forward(new_g, batch_edges=batch_edges,
                                      batch_indices=batch_edge_indices, return_all=True)
                    (user_h, item_h, user_emb_h, item_emb_h, user_t_h, item_t_h,
                     user_v_h, item_v_h, encoded_t, encoded_v, z_memory_h,
                     meanflows_loss, reward, advantage, state, action, log_prob) = outputs
                else:
                    outputs = forward(new_g, batch_edges=batch_edges,
                                      batch_indices=batch_edge_indices, return_all=True)
                    (user_h, item_h, user_emb_h, item_emb_h, user_t_h, item_t_h,
                     user_v_h, item_v_h, encoded_t, encoded_v, z_memory_h, meanflows_loss) = outputs

                mf_losses = compute_info_bpr_loss(user_h, item_h, batch_edges, num_negs=config.num_negs,
                                                  reduction="none")
                l2_loss = compute_l2_loss([user_h, item_h])


                loss = mf_losses.sum() \
                       + l2_loss * config.l2_coef \
                       + p_args.meanflows_weight * meanflows_loss

                if method != "one_game_base1":
                    pos_user_h = user_h[batch_edges[:, 0]]
                    pos_z_memory_h = z_memory_h[batch_edges[:, 1] + num_users]
                    unsmooth_logits = (pos_user_h.unsqueeze(1) @ pos_z_memory_h.permute(0, 2, 1)).squeeze(1)
                    unsmooth_loss = F.cross_entropy(
                        unsmooth_logits,
                        torch.zeros([batch_edges.size(0)], dtype=torch.long, device=device),
                        reduction="none"
                    ).sum()
                    loss = loss + p_args.unsmooth_weight * unsmooth_loss


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if method == "one_game":
                    batch_size = batch_edges.size(0)
                    total_samples = batch_size * 2
                    ppo_buffer['states'].append(state.detach().cpu())
                    ppo_buffer['actions'].append(action.detach().cpu())
                    ppo_buffer['rewards'].append(reward.detach().cpu())
                    ppo_buffer['advantages'].append(advantage.detach().cpu())
                    ppo_buffer['old_log_probs'].append(log_prob.detach().cpu())
                    ppo_buffer['dones'].extend([0] * total_samples)

                    max_buffer_steps = 50
                    if len(ppo_buffer['states']) > max_buffer_steps:
                        ppo_buffer['states'] = ppo_buffer['states'][-max_buffer_steps:]
                        ppo_buffer['actions'] = ppo_buffer['actions'][-max_buffer_steps:]
                        ppo_buffer['rewards'] = ppo_buffer['rewards'][-max_buffer_steps:]
                        ppo_buffer['advantages'] = ppo_buffer['advantages'][-max_buffer_steps:]
                        ppo_buffer['old_log_probs'] = ppo_buffer['old_log_probs'][-max_buffer_steps:]
                        ppo_buffer['dones'] = ppo_buffer['dones'][-max_buffer_steps * total_samples:]

                    if step % 4 == 0 and len(ppo_buffer['states']) >= 32:
                        states_cat = torch.cat(ppo_buffer['states'], dim=0).to(device)
                        actions_cat = torch.cat(ppo_buffer['actions'], dim=0).to(device)
                        rewards_cat = torch.cat(ppo_buffer['rewards'], dim=0).to(device)
                        adv_cat = torch.cat(ppo_buffer['advantages'], dim=0).to(device)
                        dones_cat = torch.tensor(ppo_buffer['dones'], device=device)
                        old_log_probs_cat = torch.cat(ppo_buffer['old_log_probs'], dim=0).to(device)
                        model.update_generator(states_cat, actions_cat, old_log_probs_cat,
                                               rewards_cat, dones_cat, adv_cat)

                        for key in ppo_buffer:
                            ppo_buffer[key] = []

                        torch.cuda.empty_cache()

    update_learning_rate(optimizer)
    epoch_end_time = time.time()
    total_train_time += epoch_end_time - epoch_start_time

    print(
        "epoch = {}\tloss = {:.4f}\tmf_loss = {:.4f}\tl2_loss = {:.4f}\tmeanflows_loss = {:.4f}\tupdated_lr = {:.4f}\tepoch_time = {:.4f}s\tpcount = {}"
        .format(epoch, loss.item(), mf_losses.mean().item(), l2_loss.item(), meanflows_loss.item(),
                optimizer.param_groups[0]['lr'], epoch_end_time - epoch_start_time, patience_count))

    if epoch % config.validation_freq == 0:
        print("\nEvaluation before epoch {} ......".format(epoch))
        valid_results_dict = evaluate(valid_user_items_dict, valid_mask_user_items_dict)
        print("valid_results_dict = ", valid_results_dict)

        current_score = valid_results_dict[early_stop_metric]
        if current_score > best_valid_score:
            test_results_dict = evaluate(test_user_items_dict, test_mask_user_items_dict)
            print("test_results_dict = ", test_results_dict)
            best_valid_score = current_score
            best_epoch = epoch
            early_stop_valid_results_dict = valid_results_dict
            early_stop_test_results_dict = test_results_dict
            print("updated early_stop_test_results_dict = ", early_stop_test_results_dict)
            patience_count = 0
        else:
            print("old early_stop_test_results_dict = ", early_stop_test_results_dict)
            patience_count += config.validation_freq
            if patience_count >= config.patience:
                print("Early stopping at epoch {} ......".format(epoch))
                break