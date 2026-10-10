# coding=utf-8

import torch
import torch.nn as nn
import dgl
import numpy as np
import dgl.function as fn


class GNNBackbone(nn.Module):
    CACHE_KEY = "gnn_backbone_weight"

    def __init__(self, k, alpha, beta, x_drop_rate, edge_drop_rate, z_drop_rate, *args, **kwargs):


        super().__init__(*args, **kwargs)

        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.gamma = torch.tensor(GNNBackbone.compute_gamma(alpha, beta, k)).float()

        self.gammas = [torch.tensor(GNNBackbone.compute_gamma(alpha, beta, i)).float() for i in range(1, k + 1)]

        self.x_drop_rate = x_drop_rate
        self.edge_drop_rate = edge_drop_rate
        self.z_drop_rate = z_drop_rate

        self.x_dropout = nn.Dropout(x_drop_rate)
        self.edge_dropout = nn.Dropout(edge_drop_rate)
        self.z_dropout = nn.Dropout(z_drop_rate)

    @classmethod
    def compute_gamma(cls, alpha, beta, k):
        return np.power(beta, k) + alpha * np.sum([np.power(beta, i) for i in range(k)])


    @classmethod
    def build_sorted_homo_graph(cls, user_item_edges, num_users=None, num_items=None):

        user_index, item_index = user_item_edges.T

        if num_users is None:
            num_users = np.max(user_index) + 1

        if num_items is None:
            num_items = np.max(item_index) + 1

        user_index = torch.tensor(user_index)
        item_index = torch.tensor(item_index)

        num_homo_nodes = num_users + num_items
        homo_item_index = item_index + num_users

        src = torch.concat([user_index, homo_item_index, torch.arange(num_homo_nodes)], dim=0)
        dst = torch.concat([homo_item_index, user_index, torch.arange(num_homo_nodes)], dim=0)

        g = dgl.graph((src, dst), num_nodes=num_homo_nodes)

        assert g.num_edges() == src.size(0)



        return g

    @classmethod
    @torch.no_grad()
    def norm_adj(cls, g):

        CACHE_KEY = GNNBackbone.CACHE_KEY



        degs = g.in_degrees()
        src_norm = degs.pow(-0.5)
        dst_norm = src_norm

        with g.local_scope():
            g.ndata["src_norm"] = src_norm
            g.ndata["dst_norm"] = dst_norm
            g.apply_edges(fn.u_mul_v("src_norm", "dst_norm", CACHE_KEY))
            gcn_weight = g.edata[CACHE_KEY]

        g.edata[CACHE_KEY] = gcn_weight

    def forward(self, g, x, return_all=False):

        CACHE_KEY = GNNBackbone.CACHE_KEY
        GNNBackbone.norm_adj(g)

        edge_weight = g.edata[CACHE_KEY]
        dropped_edge_weight = self.edge_dropout(edge_weight)


        h0 = self.x_dropout(x)
        h = h0

        if return_all:
            h_list = []

        with g.local_scope():
            g.edata[CACHE_KEY] = dropped_edge_weight

            for _ in range(self.k):

                g.ndata["h"] = h
                g.update_all(fn.u_mul_e("h", CACHE_KEY, "m"), fn.sum("m", "h"))
                h = g.ndata.pop("h")

                h = h * self.beta + h0 * self.alpha

                if return_all:
                    h_list.append(h)

        if not return_all:
            h = h / self.gamma
            h = self.z_dropout(h)
            return h
        else:
            h_list = [h / gamma for h, gamma in zip(h_list, self.gammas)]
            h_list = [self.z_dropout(h) for h in h_list]
            return h_list


