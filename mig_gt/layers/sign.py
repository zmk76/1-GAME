import torch
import dgl
import torch.nn.functional as F
import dgl.function as fn
from torch import nn


def l2_normalize(x, dim=-1):
    return x / (torch.norm(x, dim=dim, keepdim=True) + 1e-8)

def random_project(x, units):
    
    weight_shape = (x.shape[-1], units)
    weight = torch.randn(weight_shape)
    h = x @ weight
    h = l2_normalize(h, dim=-1)
    return h


def compute_gcn_weight(g, norm="both"):
    in_degs = g.in_degrees().float()
    out_degs = g.out_degrees().float()
    
    if isinstance(norm, (tuple, list)):
        src_norm = torch.pow(out_degs, norm[1])
        dst_norm = torch.pow(in_degs, norm[2])
    elif norm == "both":        
        src_norm = torch.pow(out_degs, -0.5)
        dst_norm = torch.pow(in_degs, -0.5)
    elif norm == "right":
        src_norm = torch.ones_like(out_degs)
        dst_norm = torch.pow(in_degs, -1.0)
    elif norm == "left":
        src_norm = torch.pow(out_degs, -1.0)
        dst_norm = torch.ones_like(in_degs)
    else:
        raise NotImplementedError()
    
    with g.local_scope():
        g.ndata['src_norm'] = src_norm
        g.ndata['dst_norm'] = dst_norm
        g.apply_edges(fn.u_mul_v('src_norm', 'dst_norm', 'gcn_weight'))
        gcn_weight = g.edata['gcn_weight']

    return gcn_weight








def sign_pre_compute(g, x, k, include_input, alpha=0.0, norm="both", remove_self_loop=False, input_drop_rate=0.0, edge_drop_rate=0.0
):
    
    if remove_self_loop:
        g = dgl.remove_self_loop(g)

    if isinstance(alpha, torch.Tensor):
        alpha = alpha.to("cpu")

    with torch.no_grad():

        if input_drop_rate > 0.0:
            x = F.dropout(x, input_drop_rate, training=True)
            print("sign: input drop rate {}".format(input_drop_rate))

        cpu_x = x.to("cpu")
        h_list = []

        if k > 0:

            g = g.to("cpu")
            h = cpu_x

            with g.local_scope():
                
                gcn_weight = compute_gcn_weight(g, norm=norm).to(x.dtype)

                if edge_drop_rate > 0.0:
                    gcn_weight = F.dropout(gcn_weight, edge_drop_rate, training=True)
                    print("sign: edge drop rate {}".format(edge_drop_rate))

                g.edata['gcn_weight'] = gcn_weight


                for k_ in range(k):
                    print("sign: start propagation {}".format(k_))

                    g.ndata['h'] = h 
                    g.update_all(fn.u_mul_e('h', 'gcn_weight', 'm'), fn.sum('m', 'h'))
                    h = g.ndata.pop('h')


                    h = (1.0 - alpha) * h + alpha * cpu_x
                    h_list.append(h)
                    print(h.mean())

            del g

    if include_input:
        h_list = [cpu_x.to("cpu")] + h_list

    return h_list
